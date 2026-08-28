# 12. Project FAQ and Design Defense

## 12.1 Format

This doc is a self-contained walkthrough for demonstrating and
defending the compiler: two non-trivial **working** examples, two
**non-working** (compilation-fail) examples, and a set of anticipated
design questions with answers. Useful for a live demo, a design
review, or just re-orienting after time away from the code.

**Update: this caveat used to say there was no general-purpose CLI —
that gap is now closed.** `tensora` (`src/driver/main.cpp`) is a real,
tested, multi-phase driver: `--emit-tokens` (lexer), `--emit-ast`
(parser), `--check`/`--emit-graph`/`--explain` (semantic analysis),
`--emit-mlir` (lowering + Level-1 optimization + Level-2 tiling),
`--emit-llvm` (backend translation), `--run` (JIT execution), and `-o`
(ahead-of-time native object-file emission) all take a real `.tns` file
and work end to end. `demos/` has one script per phase
(`demos/01_lexer_demo.sh` .. `demos/06_final_compiler_demo.sh`), each a
thin wrapper around the real driver or the CTest binaries, not a
reimplementation.

One real, honestly-stated limitation remains: `--emit-mlir`/
`--emit-llvm`/`--run`/`-o` require **concrete** (non-symbolic) shapes.
Lowering a symbolic dimension past type-checking has a genuine bug
(`tensor.empty()` isn't given dynamic-size operands for a symbolic dim
— found by running this very driver against `examples/mlp.tns`, which
has a symbolic batch dimension); the driver detects this and rejects it
with a clear message rather than crashing or miscompiling. Use
`examples/mlp_concrete.tns` (concrete batch=4) for those stages instead
of `examples/mlp.tns`. `--emit-tokens`/`--emit-ast`/`--check`/
`--emit-graph`/`--explain` work on any well-formed program, symbolic or
not, since they stop before lowering.

## 12.2 One-Sentence Pitch

"Tensora is a small declarative language for ML computation graphs
that statically verifies tensor shapes *and axis names* — catching a
class of bugs a size-only checker (and PyTorch) misses — then lowers a
verified graph through hand-authored optimization passes and a real
MLIR/LLVM backend to either JIT execution or a genuine ahead-of-time-
compiled native executable."

## 12.3 Two Working Examples (Live Demo)

Build first, from the repo root:

```sh
cd build && cmake --build . -j4
```

Both examples below are real `.tns` source files under `examples/`,
run through the actual `tensora` CLI — not a C++ test harness. Show
the source (`cat`), then compile and run it live.

### Example 1 — CNN classifier (conv2d -> relu -> flatten -> linear -> softmax)

```sh
cat ../examples/cnn.tns
./tensora --check   ../examples/cnn.tns
./tensora --emit-mlir ../examples/cnn.tns | head -20
./tensora --run     ../examples/cnn.tns
```

Five operators in one pipeline, two of which (`conv2d`, `softmax`)
needed real MLIR lowering work this phase (zero-filled/`-inf`-filled
accumulators, `tensor.pad`, a 4-step numerically-stable softmax
expansion). `--run` supplies synthetic all-ones data for every declared
input/weight (there is no tensor-literal syntax in Tensora source);
expected output:

```text
ran 'cnn' with synthetic all-ones inputs
output shape: [1, 2], 2 elements; element[0] = 0.5
```

0.5/0.5 is hand-verifiable: an all-ones conv+bias produces equal
values at every spatial position, so both logits end up equal, and
softmax of two equal values is exactly 0.5 each — no rounding to argue
about. (`tests/codegen/test_cnn_e2e.cpp` is the from-C++-fixture
counterpart, checked against a *different*, Python-double-precision
hand-picked input, 0.539902 — useful if asked "how do you know it's
numerically correct, not just that it runs".)

### Example 2 — single-head scaled dot-product attention

```sh
cat ../examples/attention.tns
./tensora --check ../examples/attention.tns
./tensora --run   ../examples/attention.tns
```

`transpose`, `matmul`, scalar `*`, `softmax`, `matmul` — rank-2,
single-head (the tutorial's Section 2.11 shows the rank-4 multi-head
version, which type-checks and now also lowers, since batched matmul
is implemented; see Section 12.5). Expected output:

```text
ran 'attention' with synthetic all-ones inputs
output shape: [2, 2], 4 elements; element[0] = 1
```

Also hand-verifiable: all-ones Q/K/V make every attention weight
exactly 0.5 (softmax of two equal scores), so the output is the mean
of two all-ones rows = 1 everywhere.

### If asked for a third or fourth

- **Level-2 tiling, live**: `./tensora --emit-mlir ../examples/matmul_large.tns | grep scf.for` shows 3
  nested loops (256x256x256 is above `MatmulTilingPass`'s default
  tiling threshold); `./tensora --run ../examples/matmul_large.tns`
  gives `element[0] = 256` — exactly the contracted dimension for
  all-ones inputs, hand-verifiable.
- **Ahead-of-time native compilation**: `./tensora -o /tmp/cnn.o ../examples/cnn.tns` then
  `file /tmp/cnn.o` (a real ELF object) — or `./test_mlp_native` +
  `ldd test_mlp_native` (zero MLIR/LLVM runtime dependency, only
  `libc`) for the fully-linked-executable version of the same story.

## 12.4 Two Non-Working (Compilation-Fail) Examples

Both are shown two ways: live via the real `tensora --check` CLI (the
*rendered* diagnostic text a user would actually see, per whitepaper
Section 4.7's diagnostic-format contract), and as a registered `ctest`
case asserting the specific `DiagCode` programmatically
(`tests/semantic/test_semantic.cpp`) — so both "does the compiler
reject this" and "with the right message" are demonstrated, not just
one.

### Non-working example 1 — axis-name mismatch (equal sizes, wrong meaning)

```sh
cat > /tmp/bad1.tns <<'EOF'
input a: Tensor[batch=32, feat=64]
input b: Tensor[feat=64, batch=32]
graph g {
    y = a + b
    return y
}
EOF
./tensora --check /tmp/bad1.tns
```

Both tensors are 32x64, but the axis *names* are swapped — `a`'s axis 0
is `batch`, `b`'s axis 0 is `feat`. A size-only checker would accept
this; Tensora rejects it (whitepaper Section 10.8; Reference Manual
Section 3.8). Expected rendered diagnostic (exact wording, verified
live):

```text
/tmp/bad1.tns:4:9: error: axis 1 is 'feat' on one operand but 'batch' on the other, and neither side is a size-1 broadcast axis
```

`DiagCode::SemaAxisNameMismatch`; also asserted programmatically by
`./test_semantic SEM-EW-003`.

### Non-working example 2 — reshape element-count mismatch

```sh
cat > /tmp/bad2.tns <<'EOF'
input x: Tensor[feat=768]
graph g {
    y = reshape(x, [head=10, dim=64])
    return y
}
EOF
./tensora --check /tmp/bad2.tns
```

`10 * 64 = 640 != 768` — the reshape does not preserve the element
count (Reference Manual Section 3.12). Expected rendered diagnostic
(exact wording, verified live):

```text
/tmp/bad2.tns:3:9: error: reshape: element count 640 != 768 (source Tensor[feat=768] -> target [head=10, dim=64])
```

`DiagCode::SemaReshapeCount`; also asserted programmatically by
`./test_semantic SEM-RS-002`.

## 12.5 Anticipated Questions

**Why MLIR instead of hand-rolled codegen or a different IR?** Existing
dialects (`linalg`, `tensor`, `arith`, `scf`) already model tensor
operations and structured loops correctly; Tensora's job is verifying
*ML-domain* semantics (axis names, operator contracts) before lowering,
not reinventing loop/tensor infrastructure. No custom Tensora dialect
was needed or built.

**Why these six Level-1 passes specifically?** They are the frozen v1
set (whitepaper Section 6.2): shape folding, dead-op elimination,
reshape/transpose cancellation, elementwise fusion, linear-bias-
activation fusion. (A seventh pass, constant folding, was built and
then retired — `docs/architecture/DECISIONS.md` ARCH-014 — since its
scalar-chain reassociation cannot be proven to preserve `return`'s
observable output under IEEE-754 f32 non-associativity.) Each pass has
a documented legality condition, a positive test, and a deliberate
near-miss test that must *not* transform.

**What's the Level-2 tiling pass, and why does it count as
project-authored work rather than "using MLIR"?** `MatmulTilingPass` is
a real, hand-written MLIR
`PassWrapper`. It applies the tile sizes `TilingDecision`
chose from the TCG's *verified* M/N/K shapes (and, under memory
pressure, a liveness estimate), using MLIR's own
`TilingInterface`/`scf::tileUsingSCF`. `TilingDecision` itself is pure
C++, independently unit-tested. The mechanical rewrite is MLIR's; the
*decision* of what to tile and by how much is project-authored, and it
traces directly to sema's output, not re-derived from the IR.

**How do you know the optimization passes preserve behavior, not just
"look smaller"?** Every pass has a positive-transformation test and a
near-miss test. The linear-bias-activation fusion, specifically, is
checked end to end (`LOW-E2E-MLP`/`NATIVE-001`): apply the fusion,
then JIT/native-execute, then compare against a hand-computed numeric
reference. Correctness is never inferred from node count alone.

**What's not implemented, and why?** General (non-flatten) `reshape`.
`flatten`'s fixed, batch-preserving collapse covered every real
workload attempted, so the riskier general form (a
runtime shape operand with weaker bufferization support) was not built
speculatively. This is recorded as a scope boundary in
`docs/testing/TEST_MATRIX.md`, and `MlirLowerer`'s lowering switch
hits `llvm_unreachable` (not a silent fallthrough) for anything
unhandled. Batched matmul, once also unimplemented, is now done.

**What was the hardest bug found?** A few candidates: (1)
`linalg.matmul`/`conv_2d_nchw_fchw`/`pooling_nchw_max` all *accumulate*
into their output operand rather than initializing it — an
uninitialized accumulator caused a genuinely intermittent (not always
reproducing) wrong-answer bug, only caught by running the same JIT test
many times; (2) tiling-introduced `scf.for` loops with tensor
`iter_args` failed to bufferize with a generic, unhelpful error, which
traced back to `scf`'s bufferizable-op-interface never being registered
in the dialect registry — a one-line fix that took real IR-dumping
investigation to find; (3) building the CLI driver's `--run` surfaced
two more: `ExecutionEngine::invokePacked` needs the `_mlir_ciface_`
symbol name prefixed manually (the templated `invoke()` does this
automatically, `invokePacked` doesn't — using it unprefixed compiled
fine and then segfaulted at runtime, with no diagnostic pointing at the
cause); and a real, still-open bug where `tensor.empty()` isn't given
dynamic-size operands for a symbolic dimension, found only because the
driver was tested against `examples/mlp.tns`'s real symbolic-batch
source rather than only hand-built, always-concrete C++ fixtures — the
strongest argument in this project for why testing the *whole*
pipeline against real source, not just each phase's unit tests, matters.

**JIT vs. native — why do both?** JIT (`ExecutionEngine`) is faster to
iterate with during development. Ahead-of-time native compilation
(`llvm::TargetMachine` -> object file -> system linker) proves the
pipeline produces a genuinely standalone executable, not something that
only runs inside an MLIR host process — confirmed via `ldd` showing no
MLIR/LLVM runtime dependency at all.

## 12.6 Pre-Demo Checklist

- [ ] `cd build && cmake --build . -j4` — clean build, no errors.
- [ ] `ctest` — all cases passing. Run it twice; this project's own
      history includes a real intermittent JIT failure, so a single
      green run is weaker evidence than two.
- [ ] Terminal open in `build/`, both working-example commands
      (`./tensora --run ../examples/cnn.tns`,
      `./tensora --run ../examples/attention.tns`) tested and their
      output visible/scrollback-able. Also try `demos/06_final_compiler_demo.sh`
      once end to end — it runs all four workloads (MLP, tiled matmul,
      CNN, attention) plus native object-file emission in one go.
- [ ] Both non-working-example files (`/tmp/bad1.tns`, `/tmp/bad2.tns`)
      created and `./tensora --check` run against each, with the exact
      rendered diagnostic text visible (Section 12.4) — this is now the
      real rendered message, not just a `DiagCode` assertion.
- [ ] Know, without looking it up: the exact `DiagCode` for each
      non-working example (`SemaAxisNameMismatch`, `SemaReshapeCount`),
      and the one-line reason each program is invalid.
- [ ] Know, without looking it up, the one real remaining limitation:
      `--run`/`-o`/`--emit-mlir`/`--emit-llvm` require concrete shapes
      (say so before being asked — Section 12.1), and why (LOW-005).
- [ ] `./tensora --emit-mlir ../examples/matmul_large.tns | grep scf.for`
      ready to show the Level-2 tiling pass actually firing, live.
- [ ] `ldd test_mlp_native` and `file mlp_native.o` ready as a backup
      "third example" if there's time (Section 12.3).
