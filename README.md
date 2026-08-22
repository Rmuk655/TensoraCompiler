# Tensora Project Instructions

This repository contains the Tensora compiler project.

Tensora is a small declarative language for describing machine-learning
computation graphs. The compiler statically checks ML
computation, keeps verified semantic information in a typed graph IR,
optimizes that graph, and lowers valid programs through MLIR and LLVM
for native execution.

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
by compiler phase, with cross-phase representations in one
`shared/` location:

``` text
include/tensora/            src/
├── lexer/                  ├── lexer/          Token, Lexer (wraps ANTLR)
├── parser/                 ├── parser/         Parser, AstBuilder, error strategy/listener
├── semantic/                ├── semantic/       SemanticAnalyzer, Symbol, Contracts
├── optimization/            ├── optimization/   GraphPass, Level-1 passes, Liveness
├── codegen/                  ├── codegen/        MlirLowerer, DialectSetup, tiling pass
└── shared/                  └── shared/         SourceSpan, Diagnostic, AST, IR, Type

grammar/Tensora.g4    one ANTLR4 grammar (lexer + parser rules)
src/driver/main.cpp   the `tensora` CLI: one flag per compiler phase
tests/                mirrors the phases above, plus smoke/ and support/
```

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

**One-time setup after cloning** (checks the toolchain against
[docs/development/DEPENDENCIES.md](docs/development/DEPENDENCIES.md)
and prints the next commands):

``` bash
./tools/setup-dev.sh
```

**Canonical build and test:**

``` bash
cmake -S . -B build           # Configure project and generate build files
cmake --build build -j4       # Compile and link using 4 parallel jobs
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
tested artifacts — the `tensora` CLI driver (`src/driver/main.cpp`)
and the `test_passes` CTest binary
(`tests/optimization/test_passes.cpp`) — not a separate copy of any
compiler logic.

**Demo commands** (one per phase; the table below has the full map):

``` bash
demos/01_lexer_demo.sh          # --emit-tokens
demos/02_parser_demo.sh         # --emit-parse-tree, --emit-ast, --emit-ast-dot
demos/03_semantic_demo.sh       # --check, --emit-symbols, --emit-graph, --emit-graph-dot, --explain
demos/04_optimization_demo.sh   # --emit-optimized-graph, the 6 Level-1 passes
demos/05_codegen_demo.sh        # --emit-mlir, --emit-llvm
demos/06_final_compiler_demo.sh # --run and -o across MLP / tiled matmul / CNN / attention
```

Run from the repository root after building (`cmake --build build`):


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

## Known limitations these demos will hit if misused

- `--emit-mlir`/`--emit-llvm`/`--run`/`-o` need **concrete**
  (non-symbolic) shapes (see `docs/testing/TEST_MATRIX.md` LOW-005).
  Use `examples/mlp_concrete.tns`, not `examples/mlp.tns`, for those
  stages.
- `--run` always uses synthetic all-ones input data (Tensora source
  declares shapes, not values). This is still a checkable choice, not
  an arbitrary one — for example, a matmul's result is exactly the
  contracted dimension.
- `examples/attention.tns` is single-head (rank-2); batched matmul is
  lowered (RANK-003/004), so a rank-4 multi-head version is simply not
  yet written as an example, not blocked on missing lowering support.
