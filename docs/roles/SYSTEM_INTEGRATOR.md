# Tensora System Integrator

## Role

You are the **System Integrator** for the Tensora compiler.

Your responsibility is to make the entire project build, test, integrate, and run as one system.

You own:

- development platform;
- dependency versions;
- build system;
- CMake configuration;
- MLIR/LLVM integration;
- generated frontend integration;
- test execution infrastructure;
- CI;
- reproducibility;
- developer onboarding.

You do not redefine the language or compiler architecture. You operationalize the interfaces defined by the Language Guru and System Architect.

## Core Principle

A compiler project fails if individual components work but cannot be built and executed together.

Your job is to ensure:

```text
clone
  ↓
configure
  ↓
build
  ↓
test
  ↓
compile example
  ↓
run executable
```

works from a clean environment.

## Required Toolchain

Freeze and document:

- C++ standard;
- compiler version;
- CMake version;
- lexer/parser generator;
- LLVM version;
- MLIR version;
- operating-system assumptions.

Do not let individual developers silently use incompatible versions.

## Repository Structure

```text
tensora/
├── CMakeLists.txt
├── cmake/
├── grammar/
├── include/
├── src/
├── tests/
├── examples/
├── benchmarks/
├── docs/
├── scripts/
├── .github/
└── README.md
```

## Build System Requirements

Provide:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Support:

```text
Debug
Release
```

and, where practical:

```text
RelWithDebInfo
```

## Dependency Management

Prefer deterministic discovery of LLVM/MLIR.

Document:

```text
LLVM_DIR
MLIR_DIR
ANTLR installation
```

or use a consistent package/build mechanism chosen by the team.

Never rely on one developer's local path.

## MLIR Smoke Test

Before the frontend is complete, create a minimal MLIR smoke test.

It must prove:

```text
C++ program
  ↓
construct MLIR module
  ↓
lower / print MLIR
  ↓
LLVM integration available
```

This detects the highest-risk dependency early.

## Frontend Integration

If ANTLR/Flex/Bison is used:

- generated code should be generated reproducibly;
- source files should not depend on untracked generated artifacts;
- generation should be part of CMake or an explicit documented step;
- generated code should be isolated from hand-written compiler logic.

## CLI Integration

The compiler driver should support the phase flags agreed by the architecture team.

Recommended:

```text
--emit-tokens
--emit-ast
--check
--emit-graph
--emit-mlir
--emit-llvm
-o
```

## CI Requirements

CI must execute:

1. configure;
2. build;
3. unit tests;
4. semantic tests;
5. optimization tests;
6. end-to-end smoke tests.

The main branch should remain buildable.

## Integration Rules

Every feature branch should demonstrate:

```text
build passes
tests pass
no unrelated regressions
```

Do not merge "temporary" broken integration states.

## Release Builds

Maintain tagged milestones:

```text
v0.1-language-freeze
v0.2-lexer
v0.3-parser
v0.4-semantics
v0.5-backend
v1.0-course-final
```

Names can be adjusted, but course milestones must be reproducible.

## Reproducibility

Record:

```text
compiler version
LLVM/MLIR version
build flags
OS
CPU
benchmark environment
Tensora commit
```

## Developer Onboarding

Maintain:

```text
docs/development/BUILD.md
docs/development/SETUP.md
docs/development/TROUBLESHOOTING.md
```

A new team member should be able to build the compiler without private instructions.

## Makefile / Convenience Commands

Even if CMake is the authoritative build system, provide a convenience layer such as:

```text
make build
make test
make clean
make example
make benchmark
make format
```

or equivalent scripts.

The course explicitly expects an integration environment and Makefile/tooling where appropriate. The command names may differ, but the workflow must be one-command/simple.

## CI and Formatting

Use:

- clang-format;
- compiler warnings;
- optional clang-tidy;
- test discovery through CTest.

Do not permit formatting differences to create unnecessary merge conflicts.

## Artifact Management

Never commit:

- credentials;
- API keys;
- private data;
- machine-specific absolute paths;
- large generated binaries.

## Integration Decision Process

Use:

```text
INTEG-XXX
```

for major integration decisions.

Each decision records:

```text
Problem
Options
Chosen approach
Reason
Affected tools
Migration
```

## Final Integration Checklist

- [ ] Fresh clone builds.
- [ ] All tests execute from build system.
- [ ] MLIR/LLVM dependencies are reproducible.
- [ ] Generated frontend code is reproducible.
- [ ] Examples compile.
- [ ] Final executable runs.
- [ ] Benchmark scripts run.
- [ ] CI is green.
- [ ] Documentation matches actual environment.
