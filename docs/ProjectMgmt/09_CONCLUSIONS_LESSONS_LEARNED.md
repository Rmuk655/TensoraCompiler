# 9. Conclusions and Lessons Learned

## 9.1 Language Design

The declarative, axis-named tensor-type design (`Tensor[batch=B,
feat=784]` rather than bare `Tensor[B,784]`) earned its complexity. The
axis-name check is not cosmetic: it catches a real class of bugs a
size-only shape checker (and, in practice, PyTorch) misses entirely —
combining a `[batch, feat]` tensor with a `[feat, batch]` tensor of the
same sizes. Every operator contract in the reference manual (Sections
3.7–3.17) ended up needing exactly this name-plus-size compatibility
rule, so the design generalized cleanly rather than needing per-operator
exceptions. The one place complexity crept in legitimately was symbolic
dimensions (Section 3.4): restricting dimension expressions to a
concrete integer, a symbolic variable, or an affine product with a
constant kept the solver a union-find plus constant arithmetic, instead
of a general constraint solver — a deliberate, documented restriction
rather than an oversight.

## 9.2 Semantic Analysis

Every operator contract (matmul, linear, elementwise+broadcast, relu,
softmax, reshape, flatten, transpose, sum/mean, conv2d, maxpool) is
statically checked, including the conv2d/maxpool spatial-arithmetic
rules (exact-division requirement on the output-size formula) and the
reshape element-count rule. Symbolic dimensions are supported as long
as reasoning about them stays inside the affine-in-one-variable subset
described above; anything outside that subset is rejected with
`SemaUnresolvableShape` rather than silently accepted or crashing later.
The axis-name mismatch diagnostic (`SemaAxisNameMismatch`) and the
reshape-count diagnostic (`SemaReshapeCount`) were, in practice, the
most immediately useful during development — both point at the exact
value and the exact numeric mismatch, not just "shape error."

## 9.3 Intermediate Representations

The four-stage separation (AST -> Typed Computation Graph -> MLIR ->
LLVM) held up well under real construction pressure. The clearest
payoff: every backend test in this project builds its `TypedGraph`
directly in C++ (bypassing the lexer/parser/semantic analyzer
entirely), which let lowering, optimization-pass, and tiling-pass work
proceed and get tested without ever blocking on frontend completeness —
exactly the parallel-development pattern the project's own instructions
call for. The TCG's axis names are deliberately erased at the MLIR
boundary (ARCH-005): concrete axis sizes become static tensor dimensions
and symbolic ones become `?`, with nothing else surviving the crossing.
That erasure boundary being *lossless* for the property tiling actually
needs (concrete M/N/K) is why the Level-2 tiling pass could treat MLIR's
static tensor dims as authoritative without re-deriving anything sema
had already proven.

## 9.4 Optimization

All six frozen Level-1 passes (shape folding, dead-op elimination,
reshape cancellation, transpose cancellation, elementwise fusion,
linear-bias-activation fusion) are implemented and each is tested with
a positive case and a deliberate near-miss case. (A seventh pass,
constant folding, was implemented and then retired — see
`docs/architecture/DECISIONS.md` ARCH-014.) Numerical-equivalence
evidence
for the linear-bias-activation fusion specifically comes from
`LOW-E2E-MLP`/`NATIVE-001`, which apply the fusion and then check the
JIT/native-executable result against a hand-computed reference — not
just "the graph has fewer nodes."

The Level-2 project-authored pass (shape-guided matmul tiling) is kept
structurally distinct from the fusion passes above: it operates on
lowered `linalg.matmul`, not on the TCG, and is applied as an ordinary
MLIR pass in the pipeline rather than a graph-level rewrite. Its
performance evidence (`LOOP-004`, plus the dedicated
`tools/benchmark_matmul_tiling` sweep) is reported honestly, and it
changed direction once measured at real scale: at a small 8x8x8
workload, tiling overhead measurably *loses* to the untiled path, since
the matrix is far too small for tiling to pay off. At a realistic
1024x1024x1024 workload, tiling wins clearly instead — roughly 2x over
untiled, consistently across the 16x16x16, 32x32x32, and 64x64x64 tile
sizes tried (`docs/testing/TEST_MATRIX.md`'s "Measured tile-size
evidence" has the exact numbers). The lesson is not "tiling is always
worth it" or "tiling is never worth it" — it is that a correctly-applied
optimization's payoff depends on problem size, so a course project
should measure at the size that matters instead of asserting a speedup
from structure alone, and should report a measured loss just as
plainly as a measured win.

## 9.5 MLIR/LLVM Integration

Standard MLIR/LLVM infrastructure reused, unmodified: the `func`,
`arith`, `linalg`, `tensor`, `scf`, `math`, and `LLVM` dialects; One-Shot
Bufferize; the generic linalg-to-loops, control-flow-to-LLVM,
arith-to-LLVM, memref-to-LLVM, and func-to-LLVM conversion passes;
`ExpandStridedMetadata` and `LowerAffine`; and `llvm::TargetMachine` for
object-code emission. Team-authored: `MlirLowerer` (TCG -> MLIR for
every implemented operator), the `GraphPass`/`PassManager`
infrastructure and its six passes, `TilingDecision` (the shape-guided
tiling *decision*, pure C++), and `MatmulTilingPass` (the MLIR pass that
*applies* that decision via MLIR's `TilingInterface`/`scf::tileUsingSCF`
— this is the project's Level-2 / USP deliverable, and its header
documents name, invariants, legality, decision, and failure behavior
explicitly so the distinction from standard infrastructure is never
ambiguous in the code itself, not just in prose).

Three genuinely non-obvious MLIR/LLVM integration bugs were found and
fixed during this phase, each worth recording precisely because none of
them were config typos — they required real investigation:

1. **Accumulate-into-`outs`, not initialize.** `linalg.matmul`,
   `linalg.conv_2d_nchw_fchw`, and `linalg.pooling_nchw_max` all
   *accumulate* into their `outs` operand rather than initializing it.
   An uninitialized `tensor.empty()` accumulator produced a genuinely
   intermittent test failure — sometimes the correct answer, sometimes
   garbage — that only repeated runs surfaced. Fixed by explicitly
   zero-filling (or, for maxpool's running max, `-inf`-filling) the
   accumulator via `linalg.fill` before the op. This is now a
   consistently-applied pattern across every lowering that uses one of
   these ops.
2. **A missing dialect registration produced a confusing downstream
   error.** `scf.for`'s bufferizable-op-interface was never registered
   in the dialect registry, so tiling-introduced nested loops with
   tensor `iter_args` could not bufferize. The resulting error ("op was
   not bufferized") gave no indication that the actual cause was a
   missing `scf::registerBufferizableOpInterfaceExternalModels` call —
   finding it required dumping IR at each pipeline stage and reasoning
   about which op's type genuinely never changed, not just tweaking
   bufferization options.
3. **A two-pass ordering bug, not a missing pass.** Tiling introduces
   `memref.subview`, which does not lower to LLVM via
   `FinalizeMemRefToLLVMConversion` alone — it needs
   `ExpandStridedMetadataPass` first. That pass in turn emits
   `affine.apply`/index arithmetic that must be lowered by
   `LowerAffinePass` *before* `ArithToLLVM` runs, not after; getting the
   order backward left `affine.apply` and stray index/i64 casts stranded
   in the final module.

Two further integration lessons were about the execution harness itself,
not the compiler pipeline:

- `ExecutionEngine::invoke()`'s templated `Result<T>` convention is
  unreliable for a plain scalar-returning JIT function (an ABI mismatch
  that manifests as intermittent garbage, not a hard failure); the
  reliable pattern used throughout this project is a raw
  `engine->lookup(name)` cast directly to a function pointer.
- A function that returns a tensor by value gets its result from a
  *new* allocation the ciface wrapper mallocs internally; the caller's
  pre-supplied output descriptor's fields (`data`/`basePtr`) get
  overwritten with that new pointer. Reading the caller's original
  buffer instead of the (now-overwritten) descriptor fields gives a
  silently wrong answer, not a crash — this was caught only by hand-
  verifying the numeric result against a computed reference, which is
  why every backend test in this project checks an exact expected value
  rather than just "did it run."

Finally: ahead-of-time native compilation (object-file emission via
`llvm::TargetMachine`, linked by the system linker into a real ELF
executable — confirmed with `file` and, via `ldd`, shown to have zero
MLIR/LLVM runtime dependency at link time) turned out to be a
materially more convincing "this is a real compiler, not an
interpreter" deliverable than JIT execution, and was not difficult to
add once the JIT path was already numerically correct — the two paths
share the same MLIR-to-LLVM-IR lowering; only the final codegen step
differs.

## 9.6 Team Engineering

The role-boundary discipline in this project's own instructions (Language
Guru / System Architect / System Integrator / Tester / Developer, each
with a defined authority order) mapped cleanly onto how the backend
phase actually proceeded: implementation work consistently deferred to
already-frozen language/architecture decisions (e.g., ARCH-005's
static/dynamic-dim erasure rule, the frozen six-pass Level-1 set)
rather than re-litigating them mid-implementation, and the one new
toolchain decision this phase required (pinning MLIR/LLVM 21 via apt)
was recorded as an integration decision (INTEG-003) rather than made
silently.

## 9.7 Scope Lessons

One item stays deferred deliberately, recorded as such rather than
silently skipped: general (non-flatten) `reshape`. `tensor.reshape`'s
runtime shape-operand form has weaker bufferization support than
`tensor.collapse_shape`, and `flatten`'s fixed, batch-preserving
collapse covered every workload actually attempted, so the team did
not build the more general, riskier form speculatively. This was not
discovered late — it is a documented scope boundary
(`MlirLowerer`'s unhandled-op case is `llvm_unreachable`, not a silent
fallthrough) recorded in `docs/testing/TEST_MATRIX.md` alongside the
tests that *are* passing.

Two items once deferred for the same reason are now done: batched
matmul (needed for multi-head attention; single-head attention, which
needs only rank-2 matmul, transpose, and softmax, had already proven
the same lowering/optimization/backend machinery end to end), and a
general-purpose CLI driver (`src/driver/main.cpp`) that takes any
`.tns` file to a native executable. Every end-to-end *test* still
constructs its `TypedGraph` directly, which was enough to validate the
backend independent of the driver.

## 9.8 Use of Code-Generation Assistants

AI coding assistance was used throughout this phase. It did not
establish correctness by itself: every lowering was individually JIT-
executed and checked against a hand-computed (or, for two workloads,
Python-double-precision-computed, introducing no runtime Python/NumPy
dependency into the C++ suite) numeric reference before being combined
into a larger workload, and every JIT test was run multiple times
specifically because this phase's own history included a real
intermittent failure (Section 9.5, item 1) that a single passing run
would have missed. The three MLIR/LLVM integration bugs in Section 9.5
were found through iterative IR inspection and targeted hypothesis-
testing (dumping the module at each pipeline stage, checking `ldd` and
`file` on emitted binaries), not accepted on faith from a first attempt
that happened to compile.

## 9.9 Open-Source Lessons

The reproducibility story is concrete, not aspirational: the exact apt
package versions needed for a from-scratch environment are recorded
(INTEG-003, `docs/development/INTEG_DECISIONS.md`), and the full
regression suite runs right after a clean `cmake .. && cmake --build .`
with no manual steps. A general-purpose CLI now exists (Section 9.7);
what is still missing for a genuine external-contributor experience is
a populated top-level `README` quick-start.

## 9.10 Research Lessons

The most concretely measured question this phase actually answers is
narrow and specific, not "does the compiler work": *at what problem size
does shape-guided matmul tiling stop being worth its own overhead?* The
answer, measured (not asserted) on this codebase's JIT path, is that an
8x8x8 matmul is well below that threshold, and a 1024x1024x1024 matmul
is well above it — tiling wins by roughly 2x there (Section 9.4). A
follow-on question this phase also answered, not just scoped: does the
Level-2 tiling decision actually use the Level-1 liveness estimate, or
only in an isolated unit test? `MlirLowerer` now computes a real
`LivenessResult` for every graph it lowers and feeds it into
`TilingDecision` for that graph's own matmul nodes (not a hand-built
fixture), confirmed by `LOOP-005-A..D`. A question that is still open,
with the same benchmark harness already able to answer it: exactly
where between "too small to tile" and "clearly worth tiling" does the
crossover happen, and does the answer change as the matrix grows large
enough to genuinely exceed cache — this needs a size sweep the current
benchmark tool does not yet run, not new infrastructure.

## 9.11 Final Conclusion

What was built: a complete Tensora pipeline from a hand-built (not yet
source-text-driven) typed computation graph through team-authored
Level-1 optimization, a project-authored Level-2 MLIR tiling pass, and
MLIR/LLVM lowering to both JIT execution and a genuine ahead-of-time
native executable — covering 14 of the 15 operators in
`docs/testing/TEST_MATRIX.md`'s coverage roll-up (every one except
general reshape), validated end to end on five real workloads (MLP,
tiled matmul, CNN, Transformer FFN, single-head attention). What was
measured: exact numeric correctness against hand- or reference-computed
values for every workload, and honest (not cherry-picked) timing
showing tiling overhead losing at small problem sizes. What worked:
the phase-by-phase, test-each-lowering-before-combining discipline,
which caught three real integration bugs and one real flaky-test root
cause before they could compound into something harder to diagnose in
a combined workload. What did not (yet) work, by scope choice rather
than failure: general reshape, and a multi-head attention *example*
specifically (batched matmul itself is implemented, so nothing blocks
writing one — it simply has not been written). The remaining technical
question — exactly where the tiling profitability crossover point
falls — is scoped precisely enough to answer with the infrastructure
already built, not a vague direction for future work.
