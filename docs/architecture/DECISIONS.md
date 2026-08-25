# Tensora Architecture Decision Records

> Owned by the System Architect. Each record: Problem, Options, Decision,
> Rationale, Affected modules, Consequences. Language-level decisions are
> frozen in the whitepaper (Section 12); these are *compiler* decisions.

---

## ARCH-001 — Typed Computation Graph as the central IR

**Problem.** What representation sits between semantic analysis and the
backend?

**Options.** (a) Lower the AST directly to MLIR; (b) an untyped graph;
(c) a typed, SSA computation graph.

**Decision.** (c) A typed, SSA computation graph with named-axis types
(see [IR.md](IR.md)).

**Rationale.** The backend must not re-check semantics. Graph passes
need verified types and producer/consumer structure. Team members need
a serializable, arena-owned IR they can build by hand for mocks.
Lowering straight to MLIR would push the MLIR dependency upstream and
lose axis names before the team's passes run.

**Affected.** `ir/`, `sema/`, `passes/`, `lowering/`.

**Consequences.** One extra IR to maintain, offset by clean phase
separation and testability.

---

## ARCH-004 — No custom MLIR dialect in v1

**Problem.** Define a `tensora` MLIR dialect, or lower directly to
existing dialects?

**Decision.** Lower directly to `func`/`linalg`/`tensor`/`arith`/`scf`.

**Rationale.** Mirrors whitepaper Decision 6. Nothing in the frozen v1
operator set needs a high-level op to survive lowering; a dialect is a
multi-week detour (TableGen, verifiers, a custom conversion pass) with no
v1 payoff. The TCG already retains high-level/semantic information
upstream of MLIR, so the option is not foreclosed.

**Affected.** `lowering/`.

**Consequences.** Simpler backend; a dialect can be added later as an
extension if a pass needs pattern preservation through lowering.

---

## ARCH-005 — Axis names live in the surface language and TCG, erased at the MLIR boundary

**Problem.** Where do named axes stop existing?

**Decision.** Names are carried on `TensorType` through the TCG (used by
passes, diagnostics, and `explain`) and dropped when lowering to MLIR,
where tensor dimensions are positional (concrete -> static dim, symbolic
-> dynamic `?`).

**Rationale.** MLIR has no notion of axis names; forcing one would require
the custom dialect ARCH-004 rejects. All name-based checking is a
frontend/TCG concern and is complete before lowering.

**Affected.** `types/`, `ir/`, `lowering/`.

**Consequences.** A single, well-defined erasure point; lowering never
needs name information.

---

## ARCH-006 — Bounded dimension solver (no SMT)

**Problem.** How much symbolic shape reasoning?

**Decision.** Union-find over symbolic dimension ids plus integer
arithmetic, implementing exactly the affine-in-one-variable subset of
RM 3.4. Element-count equality is decided by matching symbolic-factor
multisets and comparing constant products.

**Rationale.** This method is decidable, small, and enough for every
frozen operator and example. A general nonlinear/SMT solver is
research-scale and not needed here. The compiler rejects a program
that needs one, with `SemaUnresolvableShape`.

**Affected.** `types/DimSolver`, `sema/`.

**Consequences.** Checking stays predictable and easy to explain. The
compiler rejects, by design, a documented class of programs, instead
of trying to solve them.

---

## ARCH-007 — Error recovery by type poisoning

**Problem.** Stop at the first semantic error, or report many?

**Decision.** On an error, the compiler emits a diagnostic, gives the
offending value `InvalidType`, and continues. An operator that gets an
`InvalidType` operand does not report its own error, but it still
passes `InvalidType` onward.

**Rationale.** One `--check` run should find most errors in a file.
Poisoning stops a single mistake from cascading into misleading
follow-on errors. `GraphVerifier` guarantees that `InvalidType` never
reaches the backend in a graph.

**Affected.** `sema/`, `ir/` (verifier).

**Consequences.** Better ergonomics for developers and tests. Every
contract checker in sema must handle an `InvalidType` operand.

---

## ARCH-008 — GraphBuilder + GraphVerifier as the mock enablers

**Problem.** How do downstream owners work before upstream phases exist?

**Decision.** All TCG construction goes through `GraphBuilder`. All TCG
consumers can assume `GraphVerifier`-clean input. Both are public.

**Rationale.** M4 (lowering) and M5 (optimization/testing) build
`TypedProgram` fixtures by hand, and work independently of sema. The
verifier guarantees that hand-built fixtures follow the same rules as
sema-produced ones.

**Affected.** `ir/`, all downstream tests.

**Consequences.** Real parallel work (ARCHITECTURE.md §8), at a small
cost: the team must keep the builder and verifier authoritative.

---

## ARCH-009 — Two optimization levels + one data-flow analysis

**Problem.** What optimization/analysis does V0.1 need to be a strong
*compilers-course* project, given the syllabus (SSA, data-flow analysis,
loop transformations, polyhedral)?

**Decision.** V0.1 implements: (1) Level-1 pattern/graph passes over the
SSA typed graph; (2) one real **data-flow analysis** — tensor liveness,
which yields a peak-memory estimate that a decision or report uses; and
(3) a Level-2 **structured-loop tiling** decision on matmul, where
Tensora picks tile sizes from verified shapes and MLIR applies the
rewrite.

**Rationale.** The course grades SSA, data-flow analysis, and loop
transformation. Pattern rewrites alone do not cover this well. An
analysis that feeds a decision, plus a shape-driven loop
transformation, connects the project directly to the graded material.
Trading operator breadth for this depth is the core of the rescope
(V0.1 scope decision, roadmap §8).

**Boundary.** The team owns the analysis and the tiling *decision*.
MLIR performs the mechanical tiling. The report must state this split.
The team does not claim MLIR/LLVM's standard passes as its own work.

**Affected.** `analysis/`, `passes/`, `lowering/` (`TilingDecision`), tests.

**Consequences.** More compiler-engineering depth on a smaller operator
set. This creates a benchmarking obligation (tiled vs. untiled) to
back up the Level-2 claim.

---

## ARCH-010 — Matrix multiplication is the end-to-end reference computation

**Problem.** How to make the compiler's depth legible and demonstrable?

**Decision.** `C = A @ B` runs through every level — source, semantic
analysis (attribute grammar), SSA graph, MLIR `linalg.matmul`, `scf`
loops, tiled loops, LLVM, CPU — as the project's flagship example
(whitepaper §8.3).

**Rationale.** One computation, traced end to end, teaches and shows
more than many shallow operators. It anchors the viva and lines up with
the course's SSA, data-flow, and loop-transformation units.

**Affected.** Tutorial, whitepaper §8, tests (the matmul E2E and tiling
cases), demos.

**Consequences.** Matmul-heavy workloads get first-class treatment in
evaluation. The project shows other operators, but does not trace them
as deeply.

---

## ARCH-011 — ANTLR4-based frontend

**Problem.** How should the lexer and parser be implemented, to keep
the frontend easy for a new student to read while preserving the
approved lexical and syntactic grammar (RM 3.1-3.3, 3.24)?

**Options.** (a) a handwritten recursive-descent lexer and parser; (b)
ANTLR4, with one combined grammar file.

**Decision.** (b) ANTLR4, one combined grammar (`grammar/Tensora.g4`).

**Rationale.** One `.g4` file defines both the lexer and the parser,
so there are few representations between "source text" and "AST." AST
construction lives entirely in project-owned C++
(`src/parser/AstBuilder.cpp`): a plain recursive-descent walk over the
parse tree, not embedded grammar actions. This is easy for a new
reader to follow. ANTLR's generated lexer and parser stay separate
from hand-written code (only `src/lexer/` and `src/parser/` include the
generated headers). `Parser`'s public interface exposes only
`SourceBuffer` and `DiagnosticEngine`, never an ANTLR type.

**Consequences.**

- The build needs a Java runtime at configure time, to run the ANTLR
  generator jar. CMake `FetchContent` fetches the C++ runtime and the
  jar reproducibly, so no path ever depends on one machine (see
  `docs/development/BUILD.md`, INTEG-004).
- ANTLR uses adaptive LL(*) parsing. The grammar itself (RM 3.24) is
  LL(1) (see the RM's own note on this). The frontend shows
  grammar-driven code generation, lexical and syntactic rule design,
  and generator-produced error recovery.
- `Parser` lexes and parses source text in one call
  (`Parser(const SourceBuffer&, DiagnosticEngine&)`), rather than
  taking a pre-lexed `Token` vector. `Lexer`/`Token` still support the
  lexer's own standalone demonstration (`--emit-tokens`, lexer tests).
- ANTLR's default error strategy repairs some broken input by inserting
  or deleting one token, instead of dropping it outright.
  `ParserErrorStrategy` turns this off, so recovery stays panic-mode:
  one diagnostic per independent syntax error, and the whole broken
  program still rejected (PAR-028, PAR-029).

**Affected.** `grammar/`, `lexer/`, `parser/`, `ast/`, CMake,
`docs/development/BUILD.md`,
`docs/development/07_DEVELOPMENT_ENVIRONMENT.md`,
`docs/testing/TEST_MATRIX.md`, `docs/testing/08_TEST_PLAN.md`.

**Tests.** Lexer (`LEX-*`, `BND-LEX-*`) and parser (`PAR-*`,
`BND-PAR-*`) tests all pass. Full suite: 156/156 CTest cases pass.

---

## ARCH-013 — Three RM 3.3/3.21 specification holes resolved

**Problem.** A migration-driven specification audit (Stage 9 of the
Tensora rename task) found three places where the Reference Manual and
the implementation disagreed, or where the RM said nothing and the
implementation's actual behavior was arbitrary rather than designed.
The team presented each one to the project owner, with the current
behavior, the RM, the whitepaper, and a recommendation, before making
any change.

**Decisions.**

1. **Duplicate axis names within one tensor type** (e.g.
   `Tensor[batch=4, batch=8]`) were silently accepted. Decision: reject
   with a new `SemaDuplicateAxisName` diagnostic (RM 3.3).
2. **Zero-sized dimensions** (e.g. `Tensor[batch=0, feat=8]`) were
   silently accepted despite RM 3.3 already saying axis sizes are
   "concrete positive integers" (never enforced). Decision: enforce RM
   3.3 as written with a new `SemaNonPositiveDimension` diagnostic.
3. **Constant folding's scalar-combination precision**: `ConstantFoldingPass`
   combined chained scalar attributes in C++ `double`, while every
   scalar constant lowers to `f32` at runtime (`MlirLowerer.cpp`,
   `getF32FloatAttr`). Decision: combine in `float` precision instead,
   so the fold matches the runtime element type; documented as not a
   full bit-exactness guarantee (floating-point non-associativity is a
   separate, unavoidable fact -- RM 3.21).

**Rationale.** (1) and (2) make the implementation match language
rules that already existed (2), or fill an unstated gap in a way that
fits the language's "be explicit and statically checked" design (1).
(3) keeps the team-authored optimization's arithmetic precision
consistent with the backend it targets. This is what RM 3.21's
"preserve the observable output" rule really asks for, at the level of
individual scalar constants.

**Affected.** `include/tensora/shared/Diagnostic.h` (two new `DiagCode`s),
`src/semantic/SemanticAnalyzer.cpp` (`validateAxes`, called from
`convertTensorType` and the `reshape` target axis list),
`src/optimization/GraphPasses.cpp` (`ConstantFoldingPass`),
`docs/language/03_LANGUAGE_REFERENCE_MANUAL.md` §3.3, §3.21.

**Consequences.** No previously-passing test or example used a
duplicate axis name, a zero-sized dimension, or double-precision
constant-folding rounding, so no existing behavior regressed. Three
new semantic tests (`SEM-DECL-004`, `SEM-DIM-003`, `SEM-DECL-005` — a
reshape-target duplicate) cover the newly-rejected cases.

**Tests.** `SEM-DECL-004` (duplicate axis name in a tensor type),
`SEM-DIM-003` (zero-sized dimension), `SEM-DECL-005` (duplicate axis
name in a `reshape` target). Full suite verified green after the change
(158/158 -> 161/161 with these three cases added).

---

## ARCH-014 — Sema/backend rank gap closed; unsafe cast, retired
constant folding, TilingDecision wiring

**Problem.** An external review, comparing the source listing against
the Reference Manual, found five confirmed defects. The team checked
each one independently against the source before making any change:

1. `resolveOperand` (`SemanticAnalyzer.cpp`) always ran
   `static_cast<const TensorType*>` on a global symbol's type. That
   type is `types.invalid()` when the declaration itself failed RM 3.3
   validation. This is real undefined behavior, reachable through any
   use of a badly-declared global (for example `input x: Tensor[a=0]`)
   in a graph body.
2. `linear`'s contract (`Contracts.cpp`) accepts `rank(x) >= 1` (RM
   3.9), but `MlirLowerer` always built a rank-2 `linalg.matmul`. So
   rank 1 or rank >= 3 input passed sema but could not lower.
3. The same gap existed for `@`: the contract accepts `rank >= 2` with
   broadcastable leading axes (RM 3.7), but lowering handled rank 2
   only.
4. `ConstantFoldingPass` combined chained same-kind scalar ops
   (`(x*s1)*s2 -> x*(s1*s2)`) at compile time. No one can prove this
   keeps `return`'s observable output the same under IEEE-754 f32
   non-associativity — a real breach of RM 3.21's preservation rule.
   Both the code and the RM already noted this as a caveat, but the
   team had not fixed it.
5. `TilingDecision::decide()` (`LoopTiling.cpp`) had zero call sites.
   `MatmulTilingPass` computed its own tile sizes from raw MLIR shapes
   and its own `TilingPolicy`. The two stayed "consistent" only because
   their defaults happened to match, not because the code enforced it.

**Decisions.**

1. `resolveOperand` now returns the poisoned `{0, sym->type}` pair
   before the cast, when the symbol's type is `Kind::Invalid`. This
   matches every other poison-and-continue path in the analyzer
   (ARCH-007).
2. `linear`'s rank(x) 1 and rank(x) >= 3 cases now lower to a
   `linalg.generic` batched vector-matrix contraction
   (`lowerLinearContraction`). Every leading axis of `x` passes
   straight through to the output — there is no separate M axis the
   way `@` has one, since `W` is always exactly rank 2 and shared
   across the whole batch. So batching falls out of the same
   affine-map machinery already used for elementwise broadcasting.
   Rank 2 keeps the existing dedicated `linalg.matmul` path unchanged —
   that is the one path `MatmulTilingPass` tiles. The new paths give
   correctness only; v1 sets no performance requirement for them
   (project decision: only the rank-2 reference computation gets
   team-authored tiling).
3. `@`'s rank >= 2 case (any batch rank on either operand, including
   size-1 broadcasting, matching `checkMatMul`'s `broadcastAxes` rule
   exactly) now lowers through `lowerBatchedMatMul`, a `linalg.generic`
   contraction with M/N parallel, K reduction, and batch dims mapped
   through a broadcast-aware affine map that mirrors
   `broadcastOperandMap`'s elementwise rule. Rank 2 again keeps the
   dedicated tiled path.
4. The team retired `ConstantFoldingPass` instead of fixing it: no
   scalar-chain reassociation over an arbitrary runtime tensor can be
   proven bit-exact in general, so this transformation can never meet
   RM 3.21's "exactly, not merely up to rounding" rule. The frozen v1
   pass set is now six passes, not seven (RM 3.21, whitepaper 6.2).
5. `MlirLowerer` now owns a `TilingDecision` (built from the same
   `TilingPolicy` the driver gives `createMatmulTilingPass`), and it
   calls `decide()` at the TCG level for every rank-2 `@`/`linear`
   node. It stamps the resulting tile sizes onto the produced
   `linalg.matmul` as a `tensora.tile_sizes` attribute.
   `MatmulTilingPass` now only reads and applies that attribute; it no
   longer computes its own policy from raw MLIR shapes.

**Rationale.** (1)-(3) restore a core guarantee: a semantically valid
Tensora program always reaches a correct lowering, or, where the
project has not yet built support, a clean rejection.
(4) fixes a real breach of the preservation rule by removing the
transformation, instead of leaving it as a documented exception. (5)
turns "the two share one policy" into a rule the code enforces,
instead of just a claim in a comment.

**Affected.** `src/semantic/SemanticAnalyzer.cpp` (`resolveOperand`),
`src/codegen/MlirLowerer.cpp`/`.h` (`lowerLinearContraction`,
`lowerBatchedMatMul`, `TilingDecision` plumbing), `src/codegen/
MatmulTilingPass.cpp`, `src/optimization/GraphPasses.cpp`
(`ConstantFoldingPass` removed), `include/tensora/optimization/
GraphPass.h`, `src/driver/main.cpp`, `docs/language/
03_LANGUAGE_REFERENCE_MANUAL.md` §3.21, `docs/language/00_WHITEPAPER.md`
§6.2, `docs/architecture/INTERFACES.md`, `docs/architecture/
06_COMPILER_ARCHITECTURE.md`, `docs/testing/TEST_MATRIX.md`.

**Consequences.** `OPT-CF-001`/`OPT-CF-002` (ConstantFolding's own unit
tests) are removed along with the pass. No other passing test depended
on constant folding running. Four new tests (`RANK-001`..`004`,
`test_rank_e2e.cpp`) check, through JIT execution, that rank-1/rank>=3
`linear` and rank>=2 (with and without broadcasting) `@` match
hand-computed references. Full suite green after the change (162/162
with `RANK-E2E` added, then 160/160 after removing the two retired
`OPT-CF-*` cases).

**Tests.** `RANK-001` (linear, rank(x)==1), `RANK-002` (linear,
rank(x)==3, batched, shared weight), `RANK-003` (matmul, rank 3, equal
batch dims), `RANK-004` (matmul, rank 3, broadcasting batch dim).
`SEM-DIM-003` (already existing) continues to cover the unsafe-cast
path end to end: it declares an invalid global and references it in a
graph body, and continues to report a clean diagnostic with no crash.
`LOOP-002` continues to pass, confirming `MatmulTilingPass` still tiles
correctly when reading `TilingDecision`'s stamped attribute instead of
its own independently recomputed policy.

---

## ARCH-015 — `PassManager::runAll` runs the pass list to a fixed point

**Problem.** `PassManager::runAll` ran the six Level-1 passes exactly
once each, in a fixed order. This misses a real optimization case: a
pass can create the exact condition an earlier pass in the list is
looking for. Example: `linear` feeds two `relu` nodes, one dead (never
reaches `return`) and one live. `LinearBiasActivationFusion` requires
`linear`'s output to have exactly one consumer (RM 3.9's fused-region
legality needs the `linear` result read once), so with two consumers it
does not tag the pair. `DeadOpElimination` then removes the dead `relu`,
leaving `linear` with exactly one consumer — but that pass already ran
earlier in the same sweep, so the now-fusable pair was never found. A
real MLP or Transformer graph can easily have unused branches like this,
left over from an earlier transformation or from how the source program
was written, so this was a real gap, not just a hypothetical one.

**Decision.** `runAll` now repeats the full ordered sweep as long as the
last sweep changed the graph, up to a defensive cap of 10 sweeps
(`kMaxSweeps` in `GraphPass.h`). `PassStats` gained an `iterations`
field so a caller or test can see how many sweeps a `runAll` call took.
This required making the two fusion passes idempotent first
(`hasFactPrefix`, `GraphPasses.cpp`): before, a fusion pass had no check
for "already tagged," so it would re-tag the same pair with a new fact
name every sweep, which would report `changed` forever and never let
`runAll` reach a real fixed point. Every structural pass (dead-op
elimination, reshape/transpose cancellation, shape folding) is already
idempotent by construction — the pattern it looks for is gone once it
fires — so only the two fusion passes needed this fix.

**Rationale.** RM 3.21 requires every pass to preserve `return`'s
observable output; it does not require the pipeline to run each pass
exactly once. Running the six passes to a fixed point finds strictly
more of the optimizations RM 3.21 already permits, using the exact same
passes and the exact same per-pass legality arguments — no pass's own
precondition changed. The 10-sweep cap is defensive: every pass in this
project is monotonic (it only removes, merges, or tags nodes, and never
re-tags a node it already tagged), so a correct pass set converges
quickly and should never come close to the cap. The cap exists so a
future pass bug (one that reports `changed` without making real
progress) fails safe, instead of hanging the compiler.

**Affected.** `include/tensora/optimization/GraphPass.h`
(`PassStats::iterations`, `PassManager::runAll`),
`src/optimization/GraphPasses.cpp` (`hasFactPrefix`,
`ElementwiseFusionPass`, `LinearBiasActivationFusionPass`),
`docs/architecture/06_COMPILER_ARCHITECTURE.md` §6.10,
`docs/testing/TEST_MATRIX.md`.

**Consequences.** No existing test's expected final graph changed —
every prior test scenario already converged in one sweep, so the new
loop just confirms that and stops. Two new tests
(`OPT-FIX-001`/`OPT-FIX-002`) cover the fixed-point case directly: one
confirms the dead-consumer-then-fuse interaction now happens and takes
more than one sweep, the other confirms a single sweep (the old
behavior) genuinely misses it. Full suite green after the change
(172/172).

**Tests.** `OPT-FIX-001` (multi-sweep interaction: `DeadOpElimination`
enables `LinearBiasActivationFusion`), `OPT-FIX-002` (near-miss: one
sweep only, pair stays untagged).

---

## ARCH-016 — Runtime data I/O for `--run` (`.npy` load/store)

**Problem.** `--run` filled every input and weight with synthetic
all-ones data, so a compiled program produced no meaningful result.
There was no way to run a real trained model on real inputs, which the
real-world inference demos (MLP / CNN / attention) require.

**Options.**

- (a) Add a tensor-literal syntax to the language so values live in
  `.tns` source. Rejected: megabytes of trained floats do not belong in
  a text grammar, and it would be a language change (LANG-XXX) for a
  purely runtime concern.
- (b) Bake weights into the compiled artifact as `arith.constant`s.
  Rejected for now: needs new `MlirLowerer` logic to intercept weight
  params before lowering; larger than the demos need.
- (c) Load real values at run time from files, keeping `.tns` source
  value-free (shapes only, as the language already intends).

**Decision.** (c). Two additive driver flags on `--run`: `--data-dir
DIR` loads one float32 `.npy` file per declared input/weight
(`DIR/<name>.npy`, matched by the param's declared name), and `--output
FILE.npy` writes the full result tensor. A small reader/writer
(`src/driver/Npy.{h,cpp}`) handles the NumPy v1.0 format, little-endian
float32 only (the one element type Tensora uses, RM 3.23).

**Rationale.** File paths are environment-specific values, so they stay
out of `.tns` source and off at the CLI boundary — the same separation
ONNX Runtime / TVM / TensorRT use (the model is static; inputs are
supplied per call). The fill site already iterated `graph.params` with
each param's declared name, shape, and Origin, so binding a file to the
right buffer needed no new IR. All-ones stays the default, so behavior
without `--data-dir` is unchanged and no existing test is affected.
This is additive and backward-compatible, so it is recorded as an ADR
rather than a language change.

**Affected.** `src/driver/main.cpp` (flag parsing, the `--run` fill
loop, full-output writer), new `src/driver/Npy.{h,cpp}`, `CMakeLists.txt`
(driver sources, four `INT-DATALOAD-*` tests), `docs/development/BUILD.md`,
`docs/language/02_LANGUAGE_TUTORIAL.md`.

**Consequences.** stdout is unchanged whether or not `--data-dir` is
given (the full tensor goes only to `--output`), so no large-tensor
dumps and every `INT-FULL-*-RUN` test still matches. `-o` (AOT) still
emits only the graph function and supplies no data; real-data AOT would
need a separate per-graph runtime-argument convention and is left as
future work.

**Fixed-arity call limitation removed (packed invocation).** `--run`
originally called `_mlir_ciface_<graph>` by casting the function pointer
to a concrete `void(*)(void*, ...)` chosen from a `switch` on argument
count, which capped a runnable graph at 7 inputs/weights. That cap is
gone: the driver now calls MLIR's own packed C-interface entry via
`ExecutionEngine::invokePacked("_mlir_ciface_" + name, argAddrs)` (the
same path the library's `invoke()` helper uses), where `argAddrs[i]`
points to the i-th descriptor pointer in `packedArgs` (result first,
then params in declared order). There is no per-arity code and no
ceiling. A `>7`-param graph (a two-layer attention classifier head,
`examples/attn_text_classifier.tns`) now runs and matches its numpy
reference exactly. Verified: full suite green, and existing example
results (`INT-FULL-*-RUN`) unchanged.

**Tests.** `INT-DATALOAD-EQUIV` (all-ones `.npy` reproduces the built-in
all-ones result — the loader is transparent), plus `INT-DATALOAD-MISSING`,
`INT-DATALOAD-BADSHAPE`, `INT-DATALOAD-BADDTYPE` (each diagnostic
rejects bad input). The three real-world demos are the end-to-end
evidence that genuine trained parameters produce PyTorch-matching
results.

---

## ARCH-017 — CNN demo dataset: STL-10 (native 96x96), not upscaled CIFAR-100

**Problem.** The CNN real-world demo (`examples/cnn_stl10.tns`,
`tools/models/prepare_stl10_demo.py`) originally trained on CIFAR-100,
whose images are natively 32x32, upscaled (bilinear interpolation) to
96x96 so the displayed picture would be large enough to see clearly.
That upscaling does not add information: it interpolates the existing
32x32 pixels, so the shown image is a physically larger but still
blurry picture, not a genuinely higher-resolution photograph. For a
demo whose point is "show a recognizable image, run it through the
compiled CNN," a fake-resolution image undercuts exactly that goal.

**Decision.** Switch to **STL-10** (`torchvision.datasets.STL10`), which
is natively 96x96x3 with 10 recognizable object classes (airplane,
bird, car, cat, deer, dog, horse, monkey, ship, truck). No resize or
interpolation step is needed; the displayed image is genuinely that
resolution. The existing CNN architecture (four `[conv -> relu ->
maxpool]` blocks, 16→32→64→64 channels, 96→48→24→12→6 spatial) needed
no change — only the dataset loader and the final class count
(20 CIFAR-100 coarse superclasses → 10 STL-10 classes).

**Tradeoff.** STL-10's download (~2.5 GB) is larger than CIFAR-100's
(~170 MB), and its labeled training split is smaller (5000 images vs
50000), so `prepare_stl10_demo.py` trains for more epochs (40 vs 12) to
compensate. Both are one-time, cached costs (`demos/artifacts/`,
`cnn-model.pt`), not a per-run cost.

**Tests.** Unaffected — the CNN demo is not a CTest (needs PyTorch to
prepare artifacts, per ARCH-016); `E2E-007` in TEST_MATRIX.md documents
it as before, now against STL-10. `docs/development/BUILD.md`,
`README.md`, and `demos/cnn_image_classifier_demo.sh` were updated to
match; the CNN's multi-channel conv2d/maxpool and >7-input packed-JIT
coverage (ARCH-016) is unchanged, since the architecture did not
change.

---

## ARCH-018 — `--polyhedral` driver flag: an optional MLIR affine lowering path

**Problem.** v0.1's standard lowering path sends `linalg` straight to
`scf` loops (`convert-linalg-to-loops`). This path never builds an
MLIR affine loop nest, so no MLIR pass that needs the affine dialect
can run on Tensora's output. A research spike (`v0.3-research` branch)
found that a small, optional change lets Tensora exercise the MLIR
affine dialect and one real polyhedral pass, with no change to the
language or to any existing behavior.

**Decision.** Add a `--polyhedral` flag to the `tensora` driver. It is
a modifier flag, like `--data-dir`, not a new pipeline stage. When set,
the lowering pipeline runs `convert-linalg-to-affine-loops`, then
`affine-loop-tile`, then `lower-affine`, then the normal `scf`-to-LLVM
path, in place of the default `convert-linalg-to-loops`. Affine loop
tiling is a standard, size-preserving polyhedral transform. The flag
needs no CMake change, since the `tensora` target already links the
full `MLIR` library.

**Rationale — this uses no Tensora semantics.** The `--polyhedral` flag
does not read any Tensora-specific fact (no axis role, no verified
extent, no operator contract). It only runs stock MLIR passes on the
same `linalg` IR v0.1 already emits. So it carries no research risk to
v0.1: it is additive infrastructure, not a new claim about what Tensora
knows. The open research question — whether a semantic fact from
Tensora's type system can change a polyhedral decision that generic
MLIR analysis cannot make on its own — stays a V0.3 question, tracked
in `research/polyhedral_spike/SEMANTIC_POLYHEDRAL.md` on the
`v0.3-research` branch, and is not part of this decision.

**A rejected alternative — loop fusion.** The first version of this
flag used `affine-loop-fusion`, the dependence-driven MLIR pass, in
place of tiling. It gave a wrong result on the larger CNN example
(`examples/cnn_stl10.tns`): a real miscompilation in upstream MLIR
(`affine-loop-fusion` reorders a `tensor.pad` fill loop across the
`memref.copy` that fills the padded region). The bug is filed upstream
against LLVM/MLIR, with a minimal, dependency-free reproducer (see
`research/polyhedral_spike/mlir_bug_report/` on the `v0.3-research`
branch). Loop tiling does not use this MLIR code path, so it is not
affected, and it is what this flag ships.

**Consequences.** `INT-POLY-MLP-RUN`, `INT-POLY-MM-RUN`,
`INT-POLY-CNN-RUN`, `INT-POLY-ATTN-RUN`, and `INT-POLY-CNNSTL10-RUN`
(`tests/cli/cli_tests.txt`, documented in TEST_MATRIX.md) each reuse
the exact expected value of their `INT-FULL-*-RUN` twin. A match proves
the `--polyhedral` path gives the same numeric result as the default
path, on every example that runs today, not only that it runs without
error. The generated LLVM IR does differ under the flag (more branches,
more lines), so this is a real transform, not a silent no-op. Without
the flag, behavior and output are unchanged (`INT-FULL-*` still pass).
