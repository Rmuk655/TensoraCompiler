# Building Tensora

## Canonical platform: WSL2 + Ubuntu (INTEG-001)

Build and run on **WSL2 running Ubuntu** (or native Linux). Install the
whole toolchain in one command:

```bash
sudo apt update && sudo apt install -y build-essential cmake default-jre
```

This gives you a C++20 compiler (g++), CMake, and a Java runtime. You
need the Java runtime to run the ANTLR4 generator (ARCH-011,
INTEG-004). CMake fetches the ANTLR jar and the ANTLR C++ runtime
itself, reproducibly, through `FetchContent` — you do not need a
separate ANTLR install step.

**Important:** work inside the WSL native filesystem. Copy the repo
to, for example, `~/tensora` (ext4), and build there. Do **not** build
from `/mnt/c/.../OneDrive/...`. `/mnt/c` is slow, and OneDrive fights
the generated `build/` directory. This is why `build/` is git-ignored.

macOS works too (`brew install cmake openjdk`). Windows-native is not
supported (see INTEG-001).

## Build

```bash
cmake -S . -B build
cmake --build build
```

Or use the convenience wrapper (same thing, less typing):

```bash
make build     # configure + build
make test      # build + run ctest
make example   # build + run the lexer on examples/mlp.tns
make clean
```

## Run the lexer on an example

```bash
./build/tensora --emit-tokens examples/mlp.tns
```

Each printed line is `line:col  kind` (it also shows the text for
identifiers and literals). The compiler reports an illegal character
on stderr and sets a non-zero exit code.

## Run the tests

```bash
ctest --test-dir build --output-on-failure
```

## Files added in Phase 1 (lexer)

```text
grammar/Tensora.g4                        ANTLR4 grammar (lexer + parser rules)
include/tensora/lexer/Token.h             TokenKind + Token
include/tensora/lexer/TokenAdapter.h      ANTLR token -> project Token conversion
include/tensora/lexer/Lexer.h             Lexer interface
include/tensora/support/                  SourceSpan, SourceBuffer
include/tensora/diagnostics/              Diagnostic + DiagnosticEngine
src/lexer/Lexer.cpp                        drives the ANTLR-generated scanner
src/lexer/TokenAdapter.cpp                 the one ANTLR-token-to-Token conversion
src/driver/main.cpp                        tensora --emit-tokens
tests/lexer/test_lexer.cpp                 unit + boundary + diagnostic tests
```

## Notes / known build caveats

- ANTLR generates `TensoraLexer`/`TensoraParser` from
  `grammar/Tensora.g4` into the build tree (`build/antlr4_generated/`).
  A CMake custom command does this (the `antlr4_generate` target), so a
  clean `cmake --build build` always regenerates it. No one commits or
  hand-edits generated files.
- CMake `FetchContent` fetches the ANTLR jar and C++ runtime on first
  configure, which needs internet access. Both are pinned to one
  ANTLR4 version (`ANTLR4_VERSION` in `CMakeLists.txt`), for
  reproducibility.
