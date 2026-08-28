# 5. Language Evolution

## 5.1 Initial Idea: Shape-Typed Tensor Language

The first design put tensor dimensions in the type. The compiler
checked these dimensions at compile time.

This design gave strong semantics. But it overlapped a lot with prior
work in array and tensor languages, and with modern ML frameworks.

## 5.2 Semantic Contracts

The next design step put the focus on operator-specific rules, not just
shape syntax. Matrix multiplication, reshape,
convolution, and other operations were treated as contracts. Each contract states
its preconditions and the output it produces.

## 5.3 Pipeline-Level Exploration

A wider design was also explored: a verified ML pipeline. This
pipeline would cover datasets, preprocessing, models, loss, and
evaluation.

This design gave useful cross-stage checks. But it pushed the project
toward MLOps and workflow orchestration, away from compiler work.

## 5.4 Inference and Training Separation

Model training was split from forward execution. Training needs
autograd, backward graphs, optimizers, and parameter updates.

This narrowed the academic compiler's scope to forward ML computation for
inference only.

## 5.5 Computation Graph as the Core Abstraction

The language became a declarative ML computation-graph language.

This gives a direct path from source to native code:

``` text
source semantics -> typed graph -> graph optimization -> MLIR
```

This path also fits the academic compiler's semantic-analysis and code-generation
requirements better.

## 5.6 Final Core Philosophy

The core design rules are:

-   Use declarative dataflow.
-   State tensor and operator semantics explicitly.
-   Check statically wherever the compiler can prove a rule.
-   Keep semantic information after checking, for later compiler
    passes to use.
-   Run graph transformations before low-level lowering.
-   Keep the required language small. Keep the compiler architecture
    open to extension.

## 5.7 Change-Control Policy

Every language change must state:

1.  the motivation for the change;
2.  the syntax;
3.  the static semantics;
4.  the run-time meaning;
5.  the lowering strategy;
6.  positive tests;
7.  negative tests;
8.  the impact on compatibility.

The language guru owns the language specification and the change log.
