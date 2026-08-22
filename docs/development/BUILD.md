# Building Tensora

This is the canonical reference for the actual build/test/run commands.
[docs/development/07_DEVELOPMENT_ENVIRONMENT.md](07_DEVELOPMENT_ENVIRONMENT.md)
covers environment policy and process (why WSL2, why C++20, warnings
policy, git workflow, debugging, benchmarking) and links back here
instead of repeating these commands.

## Canonical platform: WSL2 + Ubuntu (INTEG-001)

Build and run on **WSL2 running Ubuntu** (or native Linux). Install the
whole toolchain in one command:

```bash
sudo apt update && sudo apt install -y build-essential cmake default-jre
```

Then run [`./tools/check-deps.sh`](../../tools/check-deps.sh). See
[docs/development/DEPENDENCIES.md](DEPENDENCIES.md) for the full
pinned-version reference (including LLVM/MLIR 21) and what each
dependency is for.

**Important:** work inside the WSL native filesystem. Copy the repo
to, for example, `~/tensora` (ext4), and build there. Do **not** build
from `/mnt/c/.../OneDrive/...`. `/mnt/c` is slow, and OneDrive fights
the generated `build/` directory. This is why `build/` is git-ignored.

macOS works too (`brew install cmake openjdk`). Windows-native is not
supported (see INTEG-001).

## Build everything

```bash
cmake -S . -B build           # Configure project and generate build files
cmake --build build -j4       # Compile and link using 4 parallel jobs
```

Or use the convenience wrapper (same thing, less typing):

```bash
make build     # configure + build everything
make test      # build + run the full ctest suite
make example   # build + run the full pipeline (demos/06_final_compiler_demo.sh)
make clean
```

## Build one phase at a time

`CMakeLists.txt` defines one CMake target per compiler phase, each
depending only on what that phase needs — building `lexer`, for
example, does not build the MLIR/LLVM backend. Useful when submitting
or demoing one phase at a time.

```bash
cmake --build build --target lexer          # lexer only
cmake --build build --target parser         # parser only
cmake --build build --target semantic       # semantic analysis only
cmake --build build --target optimization   # the 6 Level-1 passes + liveness
cmake --build build --target backend        # MLIR/LLVM codegen + tensora CLI
cmake --build build --target full           # everything (same set `make build` builds)
```

("backend", not "codegen": CMake reserves the target name `codegen`
for its own `CODEGEN` custom-command feature, so the MLIR/LLVM backend
phase target is named `backend` instead.)

Or with the `make` wrapper, which also configures first:

```bash
make lexer
make parser
make semantic
make optimization
make codegen      # builds the `backend` CMake target
make full
```

Every CTest case also carries a phase label matching this same
breakdown (see "Run the tests" below), so a phase can be tested in
isolation the same way it is built in isolation.

## Run the lexer on an example

```bash
./build/tensora --emit-tokens examples/mlp.tns
```

Each printed line is `line:col  kind` (it also shows the text for
identifiers and literals). The compiler reports an illegal character
on stderr and sets a non-zero exit code. `docs/language/02_LANGUAGE_
TUTORIAL.md` §2.12 lists every other `--emit-*` flag, one per phase.

`make example` runs the whole pipeline, not one phase. For a single
phase's demo (source in, phase output out — the same `demos/*.sh`
scripts README.md's Phase Demos section describes), use the matching
`-example` target instead:

```bash
make lexer-example         # demos/01_lexer_demo.sh
make parser-example        # demos/02_parser_demo.sh
make semantic-example      # demos/03_semantic_demo.sh
make optimization-example  # demos/04_optimization_demo.sh
make codegen-example       # demos/05_codegen_demo.sh
```

## Run the tests

```bash
ctest --test-dir build --output-on-failure # Run tests and show output for failed tests
```

Filter to one phase with `-L <label>` (`lexer`, `parser`, `semantic`,
`optimization`, `codegen`), or to one test-kind with `-L <label>`
(`unit`, `boundary`, `smoke`, `integration`):

```bash
ctest --test-dir build --output-on-failure -L semantic
```

Or with the `make` wrapper, which builds only that phase first:

```bash
make test-lexer         # build + ctest -L lexer
make test-parser        # build + ctest -L parser
make test-semantic      # build + ctest -L semantic
make test-optimization  # build + ctest -L optimization
make test-codegen       # build + ctest -L codegen
```

## Notes / known build caveats

- ANTLR generates `TensoraLexer`/`TensoraParser` from
  `grammar/Tensora.g4` into the build tree (`build/antlr4_generated/`).
  A CMake custom command does this (the `antlr4_generate` target), so a
  clean `cmake --build build` always regenerates it. No one commits or
  hand-edits generated files. `grammar/Tensora.g4` is one *combined*
  grammar file (lexer and parser rules together), so this one ANTLR
  invocation always produces both generated files together, even when
  building only the `lexer` target -- there is no way to generate just
  one from a combined grammar. This step itself is fast; it is not the
  reason a from-scratch build takes a while (see below).
- CMake `FetchContent` fetches the ANTLR jar and C++ runtime on first
  configure, which needs internet access. Both are pinned to one
  ANTLR4 version (`ANTLR4_VERSION` in `CMakeLists.txt`), for
  reproducibility. Compiling that C++ runtime from source (~100 files)
  is the real cost of a from-scratch build -- but it is a one-time cost
  per `build/` directory: CMake will not recompile it again on a later
  incremental build unless `build/` is removed. `make`'s targets pass
  `-j$(JOBS)` (default: `nproc`) to `cmake --build` for exactly this
  step, since those ~100 files compile independently and building them
  one at a time wastes every idle core on a multi-core machine.
- A from-scratch build reports progress as a percentage of the *whole*
  build graph (every library, executable, and the vendored ANTLR4
  runtime), not per compiler phase — so the percentage can jump
  straight to (for example) 67% as soon as the one large vendored
  dependency finishes, well before any Tensora-owned code has
  compiled. This is normal `cmake --build`/`make` behavior, not a sign
  anything is wrong. Building one phase at a time (above) reports
  progress against that phase's much smaller target set instead, which
  reads more sensibly.

## Regenerate the source-listing appendix

[docs/development/10_APPENDIX_SOURCE_LISTING.md](10_APPENDIX_SOURCE_LISTING.md)
is a mechanical concatenation of the real source tree, not a
hand-maintained document (see its own Section 10.10). After changing a
file it already lists, run:

```bash
make appendix
```

(or [`./tools/regen_appendix.py`](../../tools/regen_appendix.py)
directly). This is a doc-maintenance step, run by hand when needed --
it is not part of `make build`/`make test`, since it rewrites a
committed file and should not fire on every build.

The script only refreshes dumps of files it already has an entry for;
it does not decide where a *new* file's dump goes. Sections 10.3-10.8
group files by compiler phase (a curated breakdown, not something a
manifest diff can infer), so if you add a new source file, add its
`### \`path\`` entry to the right section by hand once, then re-run
the script -- it will keep that entry's content in sync from then on.
The script's own output tells you exactly which files still need this
(a new file with no entry yet, or an entry whose file was deleted).
