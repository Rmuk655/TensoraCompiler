# 5. Language Evolution

## 5.1 Initial Idea: Shape-Typed Tensor Language

The first design put tensor dimensions in the type. The compiler
checked these dimensions at compile time.

This design gave strong semantics. But it overlapped a lot with prior
work in array and tensor languages, and with modern ML frameworks.

## 5.2 Semantic Contracts

The next design step put the focus on operator-specific rules, not just
shape syntax. Shape typing alone can only ask one question:

``` text
are these dimensions compatible?
```

That question is already answered by prior array/tensor languages and
by shape-checking utilities in existing ML frameworks; it does not, by
itself, justify a new language. A semantic contract asks a stronger,
operator-specific set of questions instead:

``` text
what must be true for THIS operation, specifically?
what can the compiler prove about it before running it?
what of that proof survives, for later compiler passes to use?
```

For example, `@` (matrix multiplication) and `conv2d` are both
"just" tensor operations under shape typing, but their contracts are
very different in what they require:

``` text
@ contract                        conv2d contract
-----------------------------      -----------------------------
A[..., M, K]                       image[batch, in_channel, H, W]
B[..., K, N]                       weight[out_channel, in_channel, kh, kw]

requires:                          requires:
  contracted sizes unify             channel counts compatible
  leading axes match (name+size)     stride > 0
                                     spatial output arithmetic
                                       resolves to a concrete size

produces:                          produces:
  C[..., M, N]                       out[batch, out_channel, H', W']
```

`conv2d`'s contract (Section 3.16) is rejected statically when the
spatial output arithmetic cannot be resolved concretely — a case
shape typing alone has no vocabulary to express, since it is a
property of one specific operator's formula, not of tensor shapes in
general. This is the point at which the project stopped being "a
language with tensor shapes in its type system" and became "a
language whose static semantics are one contract per operator."

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
8.  the impact on compatibility;
9.  the normative specification changes required to keep this file and
    the whitepaper synchronized — which sections of this file change,
    which sections of `00_WHITEPAPER.md` change, whether the grammar
    (Section 3.24) changes, which diagnostics are new or changed, and
    which rows are added to `docs/testing/TEST_MATRIX.md`.

The language guru owns the language specification and the change log.

## 5.8 Language Change vs. Implementation Change

Not every compiler improvement is a language change, and this policy
does not apply to all of them. The distinction is whether a program's
*observable semantics* — its accepted/rejected status, its inferred
types, and its output values — can change:

``` text
LANGUAGE CHANGE                    IMPLEMENTATION CHANGE
--------------------------------   --------------------------------
add gelu/layernorm operator        faster matmul tiling
change what a program type-checks  better register allocation
change Tensor[...] syntax          faster MLIR lowering
change an operator's contract      new optimization pass ordering
                                      (same output, different route)
```

A language change goes through Section 5.7's full policy. An
implementation change does not, provided it satisfies the requirement
already stated in Section 3.21: optimization must preserve a program's
observable output *exactly*, not merely within a floating-point
tolerance — this is also why one tempting scalar constant-folding
optimization was rejected in Section 3.21, since it could not meet
that bar. If a proposed "implementation change" would alter a
program's accepted/rejected status or its output, it is a language
change, and Section 5.7 applies instead.
