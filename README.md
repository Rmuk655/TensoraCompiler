# Tensora Project Instructions

This repository contains the Tensora compiler project.

Tensora is a small, standalone declarative language — a teaching
compiler's DSL, not a production framework — for describing
machine-learning computation graphs over a fixed, closed set of
built-in tensor operators (`matmul`, `linear`, `relu`, `softmax`,
`conv2d`, etc.). It is not a PyTorch front-end: it does not read
PyTorch or any other existing framework's code, has no loops or
user-defined functions, and cannot express anything outside that fixed
operator set. What it does do for a `.tns` program written directly in
this DSL: statically check the computation, keep verified semantic
information in a typed graph IR, optimize that graph, and lower valid
programs through MLIR and LLVM for native execution.

**This is not an isolated toy — it interops with PyTorch today, at the
data boundary, to run genuine end-to-end programs.** Train a model in
PyTorch as usual, export its weights to `.npy` (one file per
input/weight), and load them into a compiled Tensora program with
`--data-dir` (see [docs/development/BUILD.md](docs/development/BUILD.md)).
Tensora then statically verifies, optimizes, and runs that exact
computation through MLIR/LLVM, and its result matches the PyTorch
reference within floating-point tolerance. This is not a synthetic
claim: all three demos under `demos/` (a tabular MLP, a CNN image
classifier, a self-attention text classifier) are genuine PyTorch-trained
models, on real data, running end to end through Tensora this way —
see [Real-World Inference Demos](#real-world-inference-demos) below. A
deeper, compile-time integration (for example registering Tensora as a
`torch.compile` backend, or converging with PyTorch's own MLIR path) is
design work, not yet built — see
[docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md](docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md)
§9.

The expected high-level pipeline is:

``` text
Tensora Source
      ↓
Lexer / Parser
      ↓
AST
      ↓
Semantic Analysis
      ↓
Typed Computation Graph
      ↓
Graph Optimization
      ↓
MLIR
      ↓
LLVM
      ↓
Native Execution
```

The implementation targets forward computation for inference.
Training, automatic differentiation, backpropagation, optimizers,
distributed training, and production serving possible future extensions.

------------------------------------------------------------------------

## 0. Current Repository State


**Identity.** Source files use the `.tns` extension. The compiler binary
is `tensora` (built at `build/tensora`).

**Repository architecture.** `include/tensora/` and `src/` are organized
by compiler phase, with cross-phase representations in one `shared/`
location. See [§7.6 Repository](docs/development/07_DEVELOPMENT_ENVIRONMENT.md#76-repository)
for the actual current tree.

**Compiler pipeline (as implemented):**

``` text
source (.tns)
  -> ANTLR lexer (Tensora.g4)      -> token stream
  -> ANTLR parser (Tensora.g4)     -> parse tree
  -> AstBuilder (src/parser/)      -> AST
  -> SemanticAnalyzer              -> Typed Computation Graph (TCG)
  -> Level-1 graph passes          -> optimized TCG
  -> MlirLowerer                   -> MLIR (func/linalg/tensor/arith/scf)
  -> Level-2 matmul tiling pass    -> tiled MLIR
  -> standard MLIR/LLVM pipeline   -> LLVM IR
  -> JIT execution or native object file
```

**One-time setup after cloning.** Install the system packages first
(`setup-dev.sh` deliberately does not `sudo`-install them), then run it
to verify the toolchain against
[docs/development/DEPENDENCIES.md](docs/development/DEPENDENCIES.md) and
print the next commands:

``` bash
sudo apt install -y build-essential cmake default-jre llvm-21-dev libmlir-21-dev mlir-21-tools
./tools/setup-dev.sh          # add --demos to also set up the PyTorch demo venv (large)
```

**Canonical build and test:**

``` bash
cmake -S . -B build                 # Configure project and generate build files
cmake --build build -j "$(nproc)"   # Compile and link, one job per core (default is 1 job -- serial)
ctest --test-dir build --output-on-failure # Run tests and show output for failed tests
```

(or `make build` / `make test`). First configure needs internet access:
CMake fetches the ANTLR jar and C++ runtime via `FetchContent`. See
[docs/development/BUILD.md](docs/development/BUILD.md) for the full
build/test commands, including building or testing one compiler phase
at a time (`make test-lexer`, `make test-parser`, etc.), and
[docs/development/07_DEVELOPMENT_ENVIRONMENT.md](docs/development/07_DEVELOPMENT_ENVIRONMENT.md)
for the toolchain policy (platform, compiler, LLVM/MLIR version,
warnings, debugging, git workflow).

# Tensora Phase Demos

There are six scripts: one per compiler phase, plus one that runs the
whole pipeline together. Each is a thin wrapper around the real,
tested artifacts — the `tensora` CLI driver
([`src/driver/main.cpp`](src/driver/main.cpp)) and the `test_passes`
CTest binary
([`tests/optimization/test_passes.cpp`](tests/optimization/test_passes.cpp))
— not a separate copy of any compiler logic.

**Demo commands** (one per phase; the table below has the full map):

``` bash
demos/01_lexer_demo.sh          # --emit-tokens
demos/02_parser_demo.sh         # --emit-parse-tree, --emit-ast, --emit-ast-dot
demos/03_semantic_demo.sh       # --check, --emit-symbols, --emit-graph, --emit-graph-dot, --explain
demos/04_optimization_demo.sh   # --emit-optimized-graph, the 6 Level-1 passes
demos/05_codegen_demo.sh        # --emit-mlir, --emit-llvm
demos/06_final_compiler_demo.sh # --run and -o across MLP / tiled matmul / CNN / attention
```

Run from the repository root after building (`cmake --build build`).
`make example` (alias `make examples`) runs all six in order; each is
also its own target (`make lexer-example` … `make codegen-example`,
`make final-example`).

Each script accepts an optional `.tns` file as its first argument
(defaults to a sensible example under `examples/`) and an optional
build-directory path as its second argument (defaults to `build`).

| Script | Phase | What it runs |
|---|---|---|
| `01_lexer_demo.sh` | Lexer | `tensora --emit-tokens` |
| `02_parser_demo.sh` | Parser | `tensora --emit-parse-tree`, `--emit-ast`, `--emit-ast-dot` |
| `03_semantic_demo.sh` | Semantic analysis | `tensora --check`, `--emit-symbols`, `--emit-graph`, `--emit-graph-dot`, `--explain` |
| `04_optimization_demo.sh` | Level-1 graph optimization | `tensora --emit-optimized-graph`; `test_passes` on named pass cases (positive + near-miss) |
| `05_codegen_demo.sh` | Lowering + Level-2 tiling | `tensora --emit-mlir`, `--emit-llvm` |
| `06_final_compiler_demo.sh` | Full pipeline | `tensora --run` and `-o` across MLP / tiled matmul / CNN / attention |

You can inspect every intermediate form the compiler produces
directly, one flag per stage: tokens (`--emit-tokens`), the raw ANTLR
parse tree (`--emit-parse-tree`), the AST (`--emit-ast`/`--emit-ast-dot`),
the symbol table (`--emit-symbols`, RM 3.6 — global declarations, each
graph's local scope, and its name-resolution log), the Typed
Computation Graph before and after optimization
(`--emit-graph`/`--emit-graph-dot`, `--emit-optimized-graph`), MLIR
(`--emit-mlir`), and LLVM IR (`--emit-llvm`). This is useful for
debugging one phase, or for a demo, without running the whole
pipeline.

# Visualization, Independent Pipeline, and Benchmarks

Three more demos sit alongside the per-phase ones. Each is its own
`make` target, and `make demo` (alias `make demos`) runs them together
with the real-world demos below (skipping any whose artifacts or tools
are absent):

``` bash
make viz              # demos/08 -- render AST / TCG (before+after opt) / MLIR DAG to PNGs (needs Graphviz `dot`)
make demo-independent # demos/07 -- feed Tensora's emitted MLIR/LLVM IR to upstream mlir-opt/llc/lli, same math
make benchmark        # measure Level-2 matmul-tiling speedup (not a CTest; see 07_DEVELOPMENT_ENVIRONMENT.md)
```

`make viz` gives the clearest picture of optimization: the Typed
Computation Graph *before* and *after* the Level-1 passes, side by side.
`make demo-independent` is separate from the tests on purpose — the
CTest suite *asserts* correctness (e.g. `test_passes`, and the gated
`INT-IR-*-STD` cases that check emitted IR parses under external tools),
while these demos *visualize* it; both read the same
`examples/optimizations/*.tns`, and demo 04 ends by running the
authoritative pass tests rather than re-checking anything itself.

# Real-World Inference Demos

Three demos take recognizable real inputs, use genuine PyTorch-trained
parameters, run the whole computation through Tensora (source → TCG →
optimization → MLIR → LLVM → native execution), and check the result
against a PyTorch reference. **Each needs its weights and inputs
prepared once first**, on a machine with PyTorch
([`tools/models/requirements.txt`](tools/models/requirements.txt)); the
demo runners themselves need only numpy:

``` bash
python3 tools/models/prepare_mlp_demo.py        # trains the tabular MLP, exports weights + held-out students
python3 tools/models/prepare_stl10_demo.py       # trains the STL-10 CNN, exports weights + held-out images
python3 tools/models/prepare_text_attn_demo.py   # trains the attention text classifier, exports weights + sentences
```

Only once that exists (`demos/artifacts/{mlp,cnn,attn}/`):

``` bash
make demo-mlp-placement        # student placement-eligibility (tabular MLP)
make demo-cnn-image-classifier # STL-10 image classification, 10 classes (CNN, native 96x96, shows the image)
make demo-attn-text-classifier # weather / food / sports topic (self-attention)
```

`make demo` runs all three plus the visualization, independent-pipeline,
and benchmark demos above; it skips (with a message) any real-world demo
whose artifacts have not been prepared, so it still succeeds on a fresh
clone.

Real values reach the compiled graph through `tensora --run --data-dir`
(one float32 `.npy` per input/weight); see
[docs/development/BUILD.md](docs/development/BUILD.md).

## Known limitations

- `--emit-mlir`/`--emit-llvm`/`--run`/`-o` need **concrete**
  (non-symbolic) shapes (see `docs/testing/TEST_MATRIX.md` LOW-005).
  Use `examples/mlp_concrete.tns`, not `examples/mlp.tns`, for those
  stages.
- The Transformer demo is a self-attention **classifier** built from
  Tensora's existing operators — a full pretrained BERT/DistilBERT QA
  model needs operators Tensora does not yet have (embedding/gather,
  layernorm, GELU, integer tensors), tracked in
  [docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md](docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md)
  §8.9a/§8.9.
- The only way to get data into a compiled Tensora program today is
  `.npy` files (`--data-dir`) — there is no in-memory way to hand
  Tensora a live tensor from another process. Making Tensora a
  first-class interop target (for example, a PyTorch custom op that
  passes tensors directly, no file round-trip) is needed for Tensora to
  be useful across wider use cases; see
  [docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md](docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md)
  §9.
