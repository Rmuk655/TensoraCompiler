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

`CMakeLists.txt` sets `-Wall -Wextra` for every project-owned target
(placed right after the ANTLR `FetchContent`/generate setup, so it does
not apply to the vendored ANTLR runtime). This is enforced by the build
itself, not just recommended here. A from-scratch build currently
compiles with zero warnings from project code (MLIR/LLVM system headers
are included with `SYSTEM PRIVATE`, so their own warnings do not show up
in project targets' output either). Treat new warnings as errors in CI
where practical.

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
│   ├── benchmark_matmul_tiling.cpp measured tile-size timing (untiled/16/32/64), not a CTest case
│   └── setup-dev.sh                one-time developer bootstrap (installs the pre-commit hook)
├── examples/
│   └── mlp.tns, mlp_concrete.tns, cnn.tns, attention.tns, matmul_large.tns
├── docs/
├── .github/workflows/ci.yml  pull-request CI (build, test, format/lint)
├── .clang-format / .prettierrc.json / .markdownlint.json  format/lint configs
├── .pre-commit-config.yaml   what tools/setup-dev.sh registers as the git hook
└── CMakeLists.txt
```

The one benchmark tool (`tools/benchmark_matmul_tiling.cpp`) is built
by the normal CMake build and is run directly
(`build/benchmark_matmul_tiling`), not through CTest -- see
`docs/testing/TEST_MATRIX.md`'s "Measured tile-size evidence" section
for what it actually measured on the development machine.

This repository is under git version control, hosted at
[github.com/Rmuk655/TensoraCompiler](https://github.com/Rmuk655/TensoraCompiler)
(`origin`, branch `main`). `docs/development/10_APPENDIX_SOURCE_LISTING.md`
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
GitHub. Merge only after review/testing and the required CI checks
(§7.8) pass.

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

## 7.8 Continuous Integration (CI)

`.github/workflows/ci.yml` runs on every pull request to `main` (not
every push -- see §7.9 for why), using only GitHub-hosted runners (free
for a public repository; see the GitHub Actions pricing page for
private-repository minute quotas). Two jobs:

- **Format and lint check**: `clang-format --dry-run --Werror` over
  every `.cpp`/`.h` file, plus `prettier --check` and `markdownlint`
  over every `.md` file.
- **Build and run the full test suite**: installs the toolchain
  (`build-essential`, `cmake`, `default-jre`, LLVM/MLIR 21 via
  `apt.llvm.org`'s install script -- not assumed already present on the
  runner), then `cmake -S . -B build`, `cmake --build build`, `ctest`.

`./tools/setup-dev.sh`'s pre-commit hook (§7.9) is the primary,
day-to-day mechanism and catches almost everything before a commit
even happens, for free. This workflow is the final independent check
before a merge (§7.7 step 5): a PR with a formatting or test
regression fails CI, regardless of whether the author ran the hook at
all -- but it is not the first line of defense for a small team
watching Actions minutes.

## 7.9 Formatting and Static Analysis

Formatting rules are defined by committed config files, but a config
file does not format anything by itself -- something has to invoke the
tool. This project uses three layers. For a small team, the first
layer below is the primary, day-to-day mechanism (it costs no CI
minutes and catches almost everything before a commit even happens);
CI is the final independent check before a merge, not the first line
of defense:

1. **`pre-commit` hook (primary mechanism, local, skippable).** Run
   once per clone, right after cloning:
   ```bash
   ./tools/setup-dev.sh
   ```
   This installs `pre-commit` and registers it as this clone's actual
   git hook. From then on, a normal `git add` / `git commit` runs
   `clang-format` on C/C++ and Prettier + `markdownlint` on Markdown
   automatically, with no extra command to remember. `git clone` alone
   cannot do this step itself -- letting a clone auto-run code would
   let any repository execute arbitrary commands on a developer's
   machine just by being cloned, so this one explicit bootstrap step
   is required. This layer is cooperative, not guaranteed: it can be
   skipped with `git commit --no-verify`, or never installed at all if
   a developer skips `./tools/setup-dev.sh`.
2. **Editor integration (optional, per developer).** Any editor that
   supports `clang-format`/Prettier can additionally format on save
   using the same committed configs. Not required, and not assumed --
   nothing in the repository depends on a specific editor.
3. **CI (§7.8, the final independent check).** Runs on every pull
   request to `main` (not every push, to conserve Actions minutes for
   a small team) and blocks the merge on failure, independent of
   whether any developer ran `./tools/setup-dev.sh` at all.

Configs: `.clang-format` (`BasedOnStyle: Google`, 2-space indent, a
120-column limit, matching the style already used throughout `src/`
and `include/tensora/`), `.prettierrc.json` (`proseWrap: preserve`, so
Prettier does not rewrap this project's already hand-wrapped prose),
and `.markdownlint.json` (default ruleset, with the line-length,
inline-HTML, and first-line-heading rules turned off -- they conflict
with this project's existing wide tables and note/blockquote-first
files, not with genuine problems). Vale (English prose style, the
`clang-tidy` analog for Markdown) is documented as an option in
`.pre-commit-config.yaml` but is not enabled yet -- it needs a house
style pack this repository does not have configured. `clang-tidy` is
likewise optional and not set up yet.

Neither the format/lint configs nor the CI workflow above have been
run against this codebase's actual current formatting as of writing
this section -- the tools were not available in the environment that
authored these configs. Expect the first real CI run to surface
pre-existing formatting to fix, which is normal first-adoption work,
not a sign the configuration is wrong.

## 7.10 Debugging

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

## 7.11 Benchmarking

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

## 7.12 Reproducibility

A fresh machine should be able to follow `docs/development/BUILD.md`
and reproduce the examples.

## 7.13 Open-Source Hygiene

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
