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

## 5.9 Accepted Language Changes

Every change that went through Section 5.7's policy is recorded here,
in one place, instead of as separate per-change files. This keeps the
policy (5.7) and the record of what it has actually approved together.

### LANG-001 — Bound `intLiteral` to a signed 64-bit integer

**Motivation.** RM 3.1 defines `intLiteral ::= [0-9]+` with no width
bound. The lexer (`src/lexer/Lexer.cpp`) converted every literal with
`std::stoll`, which throws `std::out_of_range` for a literal wider
than `int64_t` (for example, a 40-digit number) — an uncaught
exception, not the diagnostic `Lexer.h`'s contract promises. Every
statically compiled language with fixed-width integers makes the same
choice (C, Java, Rust, Go): an over-wide literal is a compile-time
error, not silent wraparound, truncation, or a crash.

**Syntax.** Unchanged; `intLiteral ::= [0-9]+` stays the same.

**Semantics.** An `intLiteral` token's decimal value must fit in a
signed 64-bit integer (`0` to `9223372036854775807`). A literal outside
this range is rejected with a new diagnostic code,
`DiagCode::LexIntegerOverflow`, at the literal's span; scanning
continues afterward, the same way it already continues past
`LexIllegalChar`, and the rejected token still takes an `Error` slot in
the token stream. This bound comfortably fits every real Tensora
program: axis sizes, `stride`, `padding`, `size`, and reduction-axis
indices never approach 2^63.

**Affected.** `include/tensora/shared/Diagnostic.h` (added
`DiagCode::LexIntegerOverflow`); `src/lexer/Lexer.cpp` (replaced the
unchecked `std::stoll` call with a checked parse); `tests/lexer/
test_lexer.cpp` `LEX-011` (changed from a "does not crash" case to a
real negative case asserting the diagnostic, span, `Error`-kind token,
and continued scanning, plus a min/max boundary pair);
`docs/language/03_LANGUAGE_REFERENCE_MANUAL.md` RM 3.1 (the bound);
`docs/architecture/INTERFACES.md` `DiagCode` list.

**Migration impact.** None for realistic programs. Only a pathological
literal (tens of digits) is affected, and it moves from "crashes" to "a
normal, located compile error." No accepted program's behavior changes.

**Decision.** ACCEPTED, 2026-08-18 (fixed as `REG-001` — see
`docs/testing/REGRESSION_LOG.md` and `docs/testing/KNOWN_BUGS.md`
`BUG-001`).
