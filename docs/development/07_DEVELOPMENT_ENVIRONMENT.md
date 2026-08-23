# 7. Development Environment

## 7.0 Canonical Platform (WSL2 + Ubuntu)

The canonical build and test platform is **WSL2 running Ubuntu** (or a
native Linux/Ubuntu machine). This choice is frozen (see INTEG-001).
Reason: a JRE, CMake, and LLVM/MLIR are all first-class on Linux, and
the whole toolchain installs with one `apt` command. A Windows-native
build fights the ANTLR and MLIR integration. Everyone develops and
runs tests on this platform, so results stay reproducible.

[BUILD.md](BUILD.md) has the toolchain install command and the actual
build/test commands. [DEPENDENCIES.md](DEPENDENCIES.md) lists every
dependency (OS packages, LLVM/MLIR, ANTLR) with its pinned version, and
[`./tools/check-deps.sh`](../../tools/check-deps.sh) checks a machine
against that list.

## 7.1 Language

C++20, frozen team-wide. The team bumped this from C++17 in INTEG-002,
so the Parser boundary in `INTERFACES.md` §2 can use `std::span`.

## 7.2 Frontend

Frozen choice: **ANTLR4** generates both the lexer and parser from one
grammar (see architecture ADR ARCH-011 in
[`docs/architecture/DECISIONS.md`](../architecture/DECISIONS.md)). This course teaches lexer/parser
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
[`docs/development/INTEG_DECISIONS.md`](INTEG_DECISIONS.md)), not a from-source build:

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

CMake, with reproducible configure/build/test commands and one target
per compiler phase -- see `docs/development/BUILD.md` for the actual
commands, including how to build or test one phase at a time.

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

`CMakeLists.txt` sets `-Wall -Wextra` for every project-owned target
(placed right after the ANTLR `FetchContent`/generate setup, so it does
not apply to the vendored ANTLR runtime). This is enforced by the build
itself, not just recommended here. A from-scratch build currently
compiles with zero warnings from project code (MLIR/LLVM system headers
are included with `SYSTEM PRIVATE`, so their own warnings do not show up
in project targets' output either). Treat any new warning as a defect
to fix, not something to silence.

## 7.6 Repository

The actual current tree:

```text
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
│   ├── benchmark_matmul_tiling.cpp measured tile-size timing (untiled/16/32/64), not a CTest case
│   ├── check-deps.sh               checks the toolchain against DEPENDENCIES.md, installs nothing
│   ├── setup-dev.sh                one-time bootstrap: runs check-deps.sh, prints next steps
│   └── regen_appendix.py           regenerates 10_APPENDIX_SOURCE_LISTING.md (`make appendix`)
├── examples/
│   └── mlp.tns, mlp_concrete.tns, cnn.tns, attention.tns, matmul_large.tns
├── docs/
└── CMakeLists.txt
```

The one benchmark tool ([`tools/benchmark_matmul_tiling.cpp`](../../tools/benchmark_matmul_tiling.cpp))
is a plain CMake executable, not a CTest case; run it with
`make benchmark` -- see `docs/testing/TEST_MATRIX.md`'s "Measured
tile-size evidence" section for what it actually measured on the
development machine.

This repository is under git version control, hosted at
[github.com/Rmuk655/TensoraCompiler](https://github.com/Rmuk655/TensoraCompiler)
(`origin`, branch `main`). [docs/development/10_APPENDIX_SOURCE_LISTING.md](10_APPENDIX_SOURCE_LISTING.md)
records the commit the appendix was generated against.

## 7.7 Developer Git Workflow

`main` is the stable, tested, integrated branch. No one commits to it
directly. Every change goes through a personal branch and a pull
request. Assume the repository is already cloned and `main` is checked
out.

**1. First-time setup for a new piece of work.**

```bash
cd ~/Tensora
git fetch origin
git checkout main
git pull origin main

# Create a personal development branch, e.g.:
#   <username>/semantic-analysis, <username>/parser, <username>/mlir-lowering
git checkout -b <username>/my-feature
git push -u origin <username>/my-feature
```

**2. While developing**, commit to the personal branch, never to `main`
directly:

```bash
git status
git add .
git commit -m "Add semantic validation for reshape"
git push
```

**3. Regularly bring `main` into the branch**, so it does not drift far
before the PR:

```bash
git fetch origin
git checkout <username>/my-feature
git merge origin/main
```

Resolve any conflicts in the affected files, then `git add .`,
`git commit`, `git push`.

**4. Before declaring the branch integration-ready**, merge `main` in
one more time, then build and run the full suite locally:

```bash
git fetch origin
git checkout <username>/my-feature
git merge origin/main

cmake --build build -j4
ctest --test-dir build
```

Also confirm: new tests exist for the change, existing regression
tests pass, area-specific tests pass, and the designated
tester/reviewer has tested and signed off. Commit any final changes
and push.

**5. Raise the pull request** (`<username>/my-feature` -> `main`) on
GitHub. Merge only after review and a clean local build/test run (step
4) -- there is no CI gate; a human reviewer is the check.

**6. After the PR merges**, update local `main` and branch again from
the new tip for the next piece of work -- do not keep committing to
the now-merged branch:

```bash
git checkout main
git pull origin main
git checkout -b <username>/next-feature
git push -u origin <username>/next-feature
```

The rule this workflow enforces:

```text
main                       = stable, tested, integrated code
<username>/<feature>       = work in progress
origin/main -> personal branch   = do this often
personal branch -> main          = only through an integration-ready PR
```

## 7.8 Formatting and Static Analysis

There is no enforced formatting/lint gate (no pre-commit hook, no CI):
for a team this size, that machinery costs more setup and maintenance
than it saves. Match the existing style by eye (`src/` and
`include/tensora/` already use a consistent Google-style, 2-space,
~120-column layout) and rely on code review to catch drift, the same
way review already catches everything else. `clang-format`/`clang-tidy`
may still be run by hand or from an editor if a developer wants to, but
nothing in the repository depends on it.

## 7.9 Debugging

A debug build (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`) adds `-g`
and does not define `NDEBUG`, so any `assert()` and debug symbols work
through ordinary CMake, with no project-specific setup needed (this is
generic CMake behavior, not something the project configures itself).

Tensora's own intermediate representations already have dedicated dump
flags on the `tensora` CLI, always available, not hidden behind a
build-time flag: `--emit-tokens`, `--emit-ast`/`--emit-ast-dot`,
`--emit-symbols`, `--emit-graph`/`--emit-graph-dot`, `--explain`,
`--emit-optimized-graph`, `--emit-mlir`, `--emit-llvm` (see
`docs/language/02_LANGUAGE_TUTORIAL.md` §2.12). MLIR's own per-pass IR
printing (`mlir::PassManager::enableIRPrinting`) is not wired into the
driver -- that is future work, not a currently available flag.

## 7.10 Benchmarking

[`tools/benchmark_matmul_tiling.cpp`](../../tools/benchmark_matmul_tiling.cpp)
(run with `make benchmark`) measures real wall-clock time for a
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

A fresh machine should be able to follow [BUILD.md](BUILD.md) and
reproduce the examples.

## 7.12 Open-Source Hygiene

Before public release include:

- LICENSE;
- README;
- CONTRIBUTING;
- CODE_OF_CONDUCT if desired;
- architecture document;
- examples;
- test instructions;
- issue templates;
- known limitations.

Do not commit generated binaries, credentials, local paths, or
proprietary datasets.
