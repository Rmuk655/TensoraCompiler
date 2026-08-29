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
9.  end-to-end numerical validation on six real workloads: an MLP, a
    tiled matmul-heavy model, a small CNN classifier, a Transformer
    feed-forward block, and single- and multi-head scaled
    dot-product attention;
10. a full passing regression suite (`ctest`), stable across repeated
    runs.

## 1.5 Scope

The project implements and numerically verifies, end to end, six
workloads: the two required ones (an MLP and a small CNN classifier)
and four others exercising batched matmul, a Transformer
feed-forward block, and single- and multi-head scaled dot-product
attention (see Section 1.4, item 9, for the full list).

Required target: CPU through MLIR/LLVM. The project implements both a
JIT path and an ahead-of-time native-executable path.

Potential extensions: GPU lowering, layout analysis, memory analysis,
more fusion, general (non-flatten) reshape, PyTorch interoperability,
and a larger real-world failure corpus.

## 1.6 Known Limitations

Two are worth calling out explicitly (the full list of what v1
deliberately excludes is the whitepaper's
[Section 12](../language/00_WHITEPAPER.md#12-core-language-v1-versus-future-extensions),
not repeated here):

- **General `reshape`** (arbitrary axis regrouping, as opposed to
  `flatten`'s fixed batch-preserving collapse) is not lowered — a
  documented scope boundary
  ([TEST_MATRIX.md](../testing/TEST_MATRIX.md#lowering-rm-322)),
  not a silent miscompilation (`MlirLowerer`'s lowering switch hits
  `llvm_unreachable` for any op it does not yet handle). It was never
  needed: `flatten`'s collapse covered every workload attempted.

The more consequential gap, in practice, is symbolic-dimension
lowering: a program with any symbolic axis passes semantic analysis
but cannot yet be lowered to MLIR (`tensor.empty()` and similar
allocation-introducing lowering steps do not thread through a dynamic
size). The driver detects this and rejects such a program before
lowering, with a clear message, rather than crashing or miscompiling —
see `docs/testing/TEST_MATRIX.md`'s LOW-005 entry.

## 1.7 Positioning

Tensora does not claim that tensor shape inference or ML graph
compilation is new. Existing frameworks and languages already hold
substantial work in these areas. The project's contribution is the
complete educational compiler, and the explicit connection from
semantic facts to graph transformations and lowering.
