# 1. Introduction

## 1.1 Motivation

Machine-learning computation has structure that a compiler can use:
tensor ranks and dimensions, operator-specific constraints, explicit
data dependencies, and recurring graph patterns. Tensora makes this
structure explicit in a small declarative language.

A simple model can be represented as:

``` text
Input -> Linear -> ReLU -> Linear -> Output
```

Every edge carries a tensor. Every node sets semantic requirements and
determines properties of its own result. Tensora checks those
properties before execution, and keeps them in a typed computation
graph for optimization.

## 1.2 Project Objective

The project implements a complete compiler:

``` text
Tensora
 -> Lexer/Parser
 -> AST
 -> Semantic Analysis
 -> Typed Computation Graph
 -> Optimization
 -> MLIR
 -> LLVM
 -> Native CPU Execution
```

This academic compiler's backend targets forward computation suitable for inference.
Training and automatic differentiation are outside the required scope.

## 1.3 Why a Domain-Specific Language?

A general-purpose language lets a programmer describe almost any
behavior. Tensora instead makes ML graph concepts first-class. This
limit gives the compiler a direct representation of operators and
dependencies, and it allows domain-specific diagnostics and
transformations.

The intended analogy: Tensora does not replace Python, any more than
SQL replaces general-purpose programming. A DSL's value is this — the
programmer states a domain computation using domain concepts, and the
implementation takes responsibility for verification and execution
planning.

## 1.4 Main Contributions

The completed project provides:

1.  a declarative ML computation-graph language;
2.  a complete frontend (lexer, parser, AST);
3.  tensor-aware semantic analysis, with symbolic shape propagation and
    a symbol table with observable output
4.  a typed graph Intermediate Representation (Typed Computation Graph) with a verifier;
5.  six project-authored graph transformations (shape folding, dead-op
    elimination, reshape/transpose cancellation, elementwise fusion,
    linear-bias-activation fusion — a seventh, constant folding, was
    built and then retired; see `docs/architecture/DECISIONS.md`
    ARCH-014);
6.  MLIR/LLVM lowering covering matmul (rank 2, and rank >= 2 with
    batching/broadcasting), linear (rank 1, rank 2, rank >= 3), relu,
    elementwise (with broadcasting), softmax, transpose, flatten,
    conv2d (including padding), sum, mean, and maxpool;
7.  a project-authored Level-2 MLIR pass (shape-guided matmul tiling,
    built on MLIR's own `TilingInterface`), kept distinct throughout
    from the standard MLIR/LLVM infrastructure it runs alongside;
8.  both JIT execution and ahead-of-time native compilation (a real
    object file, emitted through `llvm::TargetMachine`, linked by the
    system linker into a standalone executable with zero MLIR/LLVM
    runtime dependency);
9.  end-to-end numerical validation on five real workloads: an MLP, a
    tiled matmul-heavy model, a small CNN classifier, a Transformer
    feed-forward block, and single-head scaled dot-product attention;
10. a full passing regression suite (`ctest`), stable across repeated
    runs.

## 1.5 Scope

The project implements and numerically verifies, end to end, the
required workloads (MLP and a small CNN) and the two advanced examples
(a Transformer feed-forward block and attention). Attention is
demonstrated for a single head. Batched matmul is implemented (Section
1.7), so a multi-head attention example is possible; the project has
not yet written one.

Required target: CPU through MLIR/LLVM. The project implements both a
JIT path and an ahead-of-time native-executable path.

Potential extensions: GPU lowering, layout analysis, memory analysis,
more fusion, general (non-flatten) reshape, PyTorch interoperability,
and a larger real-world failure corpus.

## 1.7 Known Limitations

One operator remains unlowered by design, not by oversight: general
`reshape` (arbitrary axis regrouping — `flatten`'s fixed,
batch-preserving collapse covered every workload attempted). This is
recorded as a scope boundary in `docs/testing/TEST_MATRIX.md`, not
silently miscompiled: `MlirLowerer`'s lowering switch hits
`llvm_unreachable` for any op it does not yet handle. `src/driver/
main.cpp` is a general-purpose CLI driver: it takes any `.tns` source
file through the full pipeline to an executable (`--run`/`-o`). Every
MLIR-lowering and codegen *test*, though, still builds its `TypedGraph`
directly in C++ (the same parallel-development pattern the project's
own instructions call for — a downstream phase uses hand-built
fixtures instead of waiting on the frontend), and exercises the real
`MlirLowerer`, optimization passes, and backend from that point on.

## 1.6 Positioning

Tensora does not claim that tensor shape inference or ML graph
compilation is new. Existing frameworks and languages already hold
substantial work in these areas. The project's contribution is the
complete educational compiler, and the explicit connection from
semantic facts to graph transformations and lowering.
