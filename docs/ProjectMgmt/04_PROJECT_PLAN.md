# 4. Project Plan

## 4.1 Objectives

Deliver a complete Tensora academic compiler by its deadline while
concentrating engineering effort on semantic analysis and code
generation/optimization.

Required result:

``` text
source -> AST -> typed graph -> optimization -> MLIR -> LLVM -> executable
```

## 4.2 Five-Person Ownership

  -----------------------------------------------------------------------
  Owner                   Primary area            Deliverable boundary
  ----------------------- ----------------------- -----------------------
  M1                      Language + frontend     Source -\> AST

  M2                      Type system +           shared types, symbols,
                          architecture            typed-IR contracts

  M3                      Semantic analysis       AST -\> verified Typed
                                                  Graph

  M4                      MLIR/LLVM + integration Typed Graph -\>
                                                  executable

  M5                      Optimization + testing  passes, corpus,
                                                  benchmarks
  -----------------------------------------------------------------------

All members contribute tests, documentation, reviews, videos, and final
integration.

## 4.3 Parallel Development

Do not use a waterfall.

-   M1 builds grammar/parser.
-   M2 freezes AST/type/IR interfaces.
-   M3 tests semantic rules using hand-built ASTs.
-   M4 tests MLIR lowering using hand-built typed graphs.
-   M5 builds graph-pass/test infrastructure using synthetic graphs.

## 4.4 Milestones

### Language Freeze

Freeze core syntax, operator set, semantic rules, valid examples, and
invalid examples.

### Lexer

Tokenization, source positions, lexical diagnostics, token dump.

### Parser

Stable AST and syntax-error diagnostics.

### Semantic Phase

Symbol table, shape propagation, operator contracts, diagnostics,
explain mode.

### Backend

MLIR lowering, LLVM lowering, CPU execution.

### Optimization

At least one meaningful team-authored pass plus structural/performance
evaluation.

### Final

Complete documentation, tests, demos, report, source listing.

## 4.5 Scope Tiers

### Tier A -- Must Ship

-   MLP;
-   core tensor semantics;
-   matmul/linear/ReLU/reshape/transpose;
-   at least basic CNN operations;
-   semantic diagnostics;
-   typed graph;
-   MLIR/LLVM CPU execution;
-   one meaningful custom optimization;
-   comprehensive tests.

### Tier B -- Strong Final Project

-   richer broadcasting with axis-name checking;
-   convolution/pooling;
-   graph fusion;
-   explain mode;
-   100-case semantic corpus;
-   Transformer feed-forward block;
-   multi-head attention (whitepaper Section 10.4 -- frozen core-operator
    form; this is part of the graded v1 language, not a stretch goal).

### Tier C -- Research/Open-Source Extensions

-   grouped-query / multi-query attention (head-grouping contracts);
-   layout propagation;
-   memory analysis;
-   GPU lowering;
-   a custom Tensora MLIR dialect (not needed for v1; only if a future
    pass requires preserving a high-level pattern through lowering);
-   more sophisticated fusion;
-   a dtype/mixed-precision system;
-   PyTorch import/export;
-   real-framework bug corpus;
-   benchmarking against reference implementations.

Code-generation assistance can accelerate implementation, but Tier C is
never allowed to compromise Tier A correctness.

## 4.5a Status as of Final Submission

The project completes all Tier A items: MLP, core tensor semantics,
matmul/linear/relu/reshape/transpose, basic CNN operations, semantic
diagnostics, the typed graph, MLIR/LLVM CPU execution, one meaningful
custom optimization (in fact all six frozen Level-1 passes), and a
full, comprehensive test suite, stable across repeated runs.

Tier B is also complete: richer axis-name-checked broadcasting,
convolution/pooling (including padding), graph fusion, the six-pass
Level-1 suite, a Transformer feed-forward block, batched matmul, and
attention (whitepaper Section 10.4's frozen core-operator form) —
including multi-head attention, demonstrated end to end against a real
pretrained ViT-Base checkpoint's own 12-head, 768-dim self-attention
block (`demo-vit-attention`, verified independently at all 12 encoder
layers; see [README.md](../../README.md#real-world-inference-demos)),
not only the single-head hand-built case.

Beyond Tier B, the project also delivers a Tier-B/C-adjacent item not
originally itemized in Section 4.5: a project-authored **Level-2 MLIR
pass** (shape-guided matmul tiling via MLIR's own `TilingInterface`),
plus an **ahead-of-time native-executable** backend path (object-file
emission via `llvm::TargetMachine`, linked by the system linker — not
just JIT execution).

Milestones (Section 4.4): Language Freeze, Lexer, Parser, Semantic
Phase, Backend, and Optimization are all complete. `src/driver/main.cpp`
is a general-purpose CLI driver that takes any `.tns` file through the
full pipeline (`--run`/`-o`); every MLIR-lowering and codegen *test*,
though, still builds each `TypedGraph` directly in C++ test/tool code
and exercises the real lowering, optimization, and backend from there
(the mock-enabler pattern `GraphBuilder`/`GraphVerifier` make possible
— see [08_TEST_PLAN.md](../testing/08_TEST_PLAN.md) and
[ARCH-008](../architecture/DECISIONS.md#arch-008-graphbuilder--graphverifier-as-the-mock-enablers)).
A small number of Tier C items (general reshape, GPU lowering) remain
future work by design; batched matmul, originally a Tier C item, is
now done (see [01_INTRODUCTION.md §1.5](01_INTRODUCTION.md#15-scope)).

Toolchain risk (Section 4.6's "MLIR setup consumes weeks") materialized
in a smaller form than feared: MLIR/LLVM 21 C++ dev headers were not
preinstalled and had to be added via `apt` (`llvm-21-dev`,
`libmlir-21-dev`, `mlir-21-tools`; recorded as INTEG-003 in
`docs/development/INTEG_DECISIONS.md`), after which the smoke test
(`SMOKE-MLIR`) came up working within the same session.

## 4.6 Risk Register

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  MLIR setup consumes weeks           Build a scalar/matmul MLIR smoke
                                      test immediately

  language keeps changing             formal language freeze + change log

  symbolic reasoning grows too large  support a documented constraint
                                      subset

  optimization is only cosmetic       select at least one transformation
                                      with measurable structural benefit

  generated code outruns              require owner review, tests, and
  understanding                       design notes for every subsystem

  five-person integration failure     stable interfaces + mocks + CI

  GPU becomes a time sink             CPU required; GPU only after
                                      end-to-end CPU path
  -----------------------------------------------------------------------

## 4.7 Open-Source Readiness

Original plan: the repository should include a clear README, a
quick-start example, an architecture diagram, a language
specification, a contributor guide, a reproducible build, examples,
regression tests, benchmark scripts, issue templates, a chosen
license, and a tagged release.

Status: most of the substance is done, but the repository is not yet
actually published, so three items below stay open by omission, not by
oversight.

Done:

-   **README** — comprehensive: project description, pipeline diagram,
    one-time setup, canonical build/test commands, and a full demo
    catalog ([README.md](../../README.md)).
-   **Quick-start example** — the "Canonical build and test" and
    "Compiler Demos" sections of the README double as this (`cmake`
    configure/build/test, then `make demo-mlp` and friends).
-   **Architecture diagram** — as a text pipeline diagram in the
    README and as the module map in
    [06_COMPILER_ARCHITECTURE.md §2](../architecture/06_COMPILER_ARCHITECTURE.md),
    not (yet) as a rendered image file.
-   **Language specification** —
    [03_LANGUAGE_REFERENCE_MANUAL.md](../language/03_LANGUAGE_REFERENCE_MANUAL.md),
    plus the whitepaper and tutorial.
-   **Reproducible build** — CMake `FetchContent` pins the ANTLR jar
    and C++ runtime; see
    [BUILD.md](../development/BUILD.md)/[07_DEVELOPMENT_ENVIRONMENT.md](../development/07_DEVELOPMENT_ENVIRONMENT.md).
-   **Examples** — `examples/*.tns` plus the `demos/` end-to-end
    PyTorch-interop programs.
-   **Regression tests** — the CTest suite plus
    [REGRESSION_LOG.md](../testing/REGRESSION_LOG.md) and
    [KNOWN_BUGS.md](../testing/KNOWN_BUGS.md).
-   **Benchmark scripts** — `tools/benchmark_matmul_tiling.cpp`
    (tiled-vs-untiled timing, backing the Level-2 claim in
    [ARCH-009](../architecture/DECISIONS.md#arch-009-two-optimization-levels-one-data-flow-analysis)).

Not yet done:

-   **Contributor guide** — no `CONTRIBUTING.md`.
-   **Issue templates** — no `.github/ISSUE_TEMPLATE/`.
-   **License and a tagged release** — no `LICENSE` file and no git
    tag exist yet; these are the actual publish-time gate, since a
    public repository with no license grants no one any rights to use
    it. The remaining work here is a licensing decision (owned by the
    team/institution, not a compiler-engineering task) followed by
    adding the three items above and cutting a tag — not further
    compiler work.
