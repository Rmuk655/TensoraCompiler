# 7. Development Environment

## 7.0 Canonical Platform (WSL2 + Ubuntu)

The canonical build and test platform is **WSL2 running Ubuntu** (or a
native Linux/Ubuntu machine). This choice is frozen (see INTEG-001).
Reason: a JRE, CMake, and LLVM/MLIR are all first-class on Linux, and
the whole toolchain installs with one `apt` command. A Windows-native
build fights the ANTLR and MLIR integration. Everyone develops and
runs tests on this platform, so results stay reproducible.

Toolchain install (one line):

```bash
sudo apt update && sudo apt install -y build-essential cmake default-jre
```

**Work inside the WSL native filesystem**, not on a Windows/OneDrive
path: clone or copy the repo to, for example, `~/tensora` (ext4), not
`/mnt/c/...`. Building under `/mnt/c` is slow, and OneDrive sync fights
the generated `build/` directory. Keep `build/` out of git and out of
OneDrive.

## 7.1 Language

C++20, frozen team-wide. The team bumped this from C++17 in INTEG-002,
so the Parser boundary in `INTERFACES.md` §2 can use `std::span`.

## 7.2 Frontend

Frozen choice: **ANTLR4** generates both the lexer and parser from one
grammar (see architecture ADR ARCH-011 in
`docs/architecture/DECISIONS.md`). This course teaches lexer/parser
generators and grammar-driven code generation, so an ANTLR frontend
shows graded frontend-generator theory directly, with a small amount of
hand-integrated generator glue for a new student to trace through
before reaching the AST.

The single `grammar/Tensora.g4` specification lives in `grammar/`, and
the CMake build generates it reproducibly. This keeps it separate from
hand-written compiler logic, which lives in
`src/parser/AstBuilder.cpp` and `src/parser/ParserErrorListener.cpp`.
The AST interface (`docs/architecture/INTERFACES.md` §2) stays the
same and does not depend on the generator, so a future switch would
not disturb downstream phases.

The team freezes the ANTLR4 version (`ANTLR4_VERSION` in
`CMakeLists.txt`) along with the compiler and CMake versions (Section
7.3), so generated code stays reproducible across machines. CMake
`FetchContent` fetches both the ANTLR jar and its C++ runtime, instead
of assuming they are already installed system-wide.

## 7.3 Compiler Infrastructure

**Pinned: LLVM/MLIR 21.1.8, installed via apt packages** (INTEG-003 in
`docs/development/INTEG_DECISIONS.md`), not a from-source build:

```bash
sudo apt install -y llvm-21-dev libmlir-21-dev mlir-21-tools
```

This installs `llvm-config-21`, `mlir-opt`, `mlir-translate`, and `llc`
under `/usr/lib/llvm-21/bin/`; the CMake config packages at
`/usr/lib/llvm-21/lib/cmake/{llvm,mlir}`; and the runner-support shared
libraries (`libmlir_c_runner_utils.so`, `libmlir_runner_utils.so`).
JIT-executed tests need these libraries whenever the lowered IR calls
MLIR's own `memrefCopy` runtime helper — any tiling, reshape, or
conv2d-padding step that produces a `memref.copy` on a non-identity-
layout memref does this. `CMakeLists.txt` adds both cmake directories
to `CMAKE_PREFIX_PATH` before it calls `find_package(MLIR REQUIRED
CONFIG)` and `find_package(LLVM REQUIRED CONFIG)`. Both packages
export combined shared-library imported targets named `MLIR` and
`LLVM`. Every lowering and codegen target links against these directly,
instead of listing each per-dialect library. Reproducibility record:
exact package version `1:21.1.8-6ubuntu1`; `llvm-config-21 --version`
reports `21.1.8`.

This pin needs no source changes to the frontend or semantic analyzer,
and allows none — neither has any MLIR/LLVM dependency
(`docs/architecture/06_COMPILER_ARCHITECTURE.md` §6.3/§6.11). Only the
`lowering/` module and its tests and tools depend on MLIR/LLVM.

## 7.4 Build

CMake with reproducible configure/build/test commands. A from-scratch
clone builds and passes the full suite with:

``` bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build
```

`ctest` currently reports **all tests passing, none failing**, across
every phase: lexer, parser, semantic analysis, boundary and negative
cases, the six Level-1 optimization passes, tensor liveness, the
Level-2 matmul tiling pass, MLIR lowering for every implemented
operator, and two ahead-of-time-compiled native executables. The team
ran the suite repeatedly during development, not just once, to catch
JIT and bufferization flakiness — see
`docs/ProjectMgmt/09_CONCLUSIONS_LESSONS_LEARNED.md` for the
intermittent-failure lesson that motivated this.

## 7.5 Warnings

At minimum:

``` text
-Wall -Wextra
```

Treat additional warnings as errors in CI where practical.

## 7.6 Repository

The actual current tree (verified, not aspirational):

``` text
tensora/
├── grammar/                 Tensora.g4 (ANTLR4: lexer + parser rules)
├── include/tensora/
│   ├── lexer/               Token, TokenAdapter (ANTLR token -> Token), Lexer
│   ├── parser/              Parser (runs the ANTLR lexer+parser over source text)
│   ├── semantic/            SemanticAnalyzer, Symbol/SymbolTable, Contracts
│   ├── optimization/        GraphPass, PassManager, the 6 Level-1 passes, Liveness
│   ├── codegen/             MlirLowerer, DialectSetup, LoopTiling (TilingDecision),
│   │                        MatmulTilingPass (Level-2, project-authored MLIR pass)
│   └── shared/              SourceSpan, SourceBuffer, Diagnostic, Ast, Ir, Type,
│                            DimSolver, Result -- cross-phase representations
├── src/                     mirrors include/tensora/, plus:
│   ├── parser/AstBuilder.{h,cpp}          parse tree -> AST (private to src/parser/)
│   ├── parser/ParserErrorListener.{h,cpp} ANTLR syntax errors -> Diagnostic
│   ├── parser/ParserErrorStrategy.{h,cpp} panic-mode recovery strategy
│   └── driver/main.cpp      the `tensora` CLI: one flag per compiler phase, plus
│                            --run (JIT-execute) / -o (native object file)
├── tests/
│   ├── lexer/  parser/  semantic/  optimization/  codegen/  smoke/  support/
├── tools/
│   ├── codegen_mlp_native.cpp      ahead-of-time object-file codegen tool (not the JIT path)
│   └── benchmark_matmul_tiling.cpp measured tile-size timing (untiled/16/32/64), not a CTest case
├── examples/
│   └── mlp.tns, mlp_concrete.tns, cnn.tns, attention.tns, matmul_large.tns
├── docs/
└── CMakeLists.txt
```

`benchmarks/` and `scripts/` directories do not exist. The one real
benchmark tool (`tools/benchmark_matmul_tiling.cpp`) lives under
`tools/`, is built by the normal CMake build, and is run directly
(`build/benchmark_matmul_tiling`), not through CTest -- see
`docs/testing/TEST_MATRIX.md`'s "Measured tile-size evidence" section
for what it actually measured on the development machine. This
repository is not under git version control for this submission
(`git rev-parse HEAD` fails with "not a git repository"). See
`docs/development/10_APPENDIX_SOURCE_LISTING.md` for how the team
generated the source appendix without a commit hash to anchor it to.

## 7.7 CI

CI should build, run unit tests, semantic regression tests, lowering
tests, and end-to-end smoke tests.

## 7.8 Formatting and Static Analysis

Use `clang-format`; optionally use `clang-tidy` with a documented
configuration.

## 7.9 Debugging

Support debug builds with assertions and symbols. MLIR dumps should be
available behind compiler flags.

## 7.10 Benchmarking

`tools/benchmark_matmul_tiling.cpp` measures real wall-clock time for a
1024x1024x1024 matmul, untiled and at three candidate tile sizes
(16/16/16, 32/32/32, 64/64/64), best of 5 runs each. It records the
workload size and reports the measured seconds and the speedup over
untiled directly to stdout; it does not yet record machine/CPU
information, compiler version, or a warmup/repetition policy beyond
"best of 5" -- a fuller benchmark harness recording those fields is
future work. See `docs/testing/TEST_MATRIX.md`'s "Measured tile-size
evidence" section for the actual numbers this tool produced on the
development machine.

## 7.11 Reproducibility

A fresh machine should be able to follow `docs/development/BUILD.md`
and reproduce the examples.

## 7.12 Open-Source Hygiene

Before public release include:

-   LICENSE;
-   README;
-   CONTRIBUTING;
-   CODE_OF_CONDUCT if desired;
-   architecture document;
-   examples;
-   test instructions;
-   issue templates;
-   known limitations.

Do not commit generated binaries, credentials, local paths, or
proprietary datasets.
