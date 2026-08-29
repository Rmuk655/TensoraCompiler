# 3. Tensora Language Reference Manual

> This file is the normative baseline for Tensora v1. `00_WHITEPAPER.md`
> Section 12 freezes this baseline. This file must stay in sync with the
> whitepaper. A change to one file requires a matching change to the
> other. See the change-control policy in `05_LANGUAGE_EVOLUTION.md`.

## 3.0 Operator Quick Reference

This table is a lookup index, not a substitute for the normative
sections it points to — every static requirement below is stated
precisely, and completely, only in the linked section.

| Operator | Input -> output (rank) | Main static requirement |
|---|---|---|
| [`@` (3.7)](#37-matrix-multiplication-) | `[...,M,K] x [...,K,N] -> [...,M,N]` | contracted sizes equal/unifiable; leading axes match in name and size |
| [`+ - * /` (3.8)](#38-elementwise-arithmetic-and-broadcasting) | `T x T -> T` (broadcast) | aligned axes equal/unifiable/size-1 in size, **and** matching in name unless broadcasting from size 1 |
| [`linear` (3.9)](#39-linear) | `[...,K], W[K,N], b[N] -> [...,N]` | `x`'s last axis == `W`'s input axis; `b`'s axis == `W`'s output axis |
| [`relu` (3.10)](#310-relu) | `T -> T` | none (shape-preserving) |
| [`softmax` (3.11)](#311-softmax) | `T -> T` | named axis must exist in the input |
| [`reshape` (3.12)](#312-reshape) | `T -> T'` (any rank) | product of target sizes == product of source sizes |
| [`flatten` (3.13)](#313-flatten) | rank N -> rank 2 | trailing dimensions being merged must be concrete |
| [`transpose` (3.14)](#314-transpose) | `T -> T'` (same rank) | `perm` names every axis position exactly once |
| [`sum` / `mean` (3.15)](#315-sum--mean) | rank N -> rank N-1 | named axis must exist in the input |
| [`conv2d` (3.16)](#316-conv2d) | NCHW -> NCHW | channel counts compatible; spatial output arithmetic must resolve concretely |
| [`maxpool` (3.17)](#317-maxpool) | NCHW -> NCHW | window/stride produce a valid, concrete output size |

## 3.1 Lexical Structure

### Identifiers

``` text
identifier ::= [A-Za-z_][A-Za-z0-9_]*
```

Identifiers name inputs, weights, graphs, intermediate values, and
tensor axes. An axis name is an ordinary identifier. It is not part of
a reserved namespace.

### Literals

``` text
intLiteral   ::= [0-9]+
floatLiteral ::= [0-9]+ "." [0-9]+
```

The language uses integer literals for concrete axis sizes and for
integer operator arguments (`stride`, `padding`, `size`, and `axis`
index where one applies). An `intLiteral` value must fit in a signed
64-bit integer (`0` to `9223372036854775807`). The compiler rejects a
literal outside this range with a `LexIntegerOverflow` diagnostic. It
does not truncate or wrap the value (LANG-001). The language uses
floating-point literals as scalar operands in elementwise expressions
(for example, `scores * 0.125`).

### Core Keywords

``` text
input  weight  graph  return  Tensor
```

### Core Operators and Delimiters

``` text
+ - * / @ = : [ ] { } ( ) , #
```

`#` starts a line comment. The comment ends at the end of the line.

## 3.2 Program Structure

``` text
program     ::= declaration* graphDecl+
declaration ::= inputDecl | weightDecl
inputDecl   ::= "input" identifier ":" tensorType
weightDecl  ::= "weight" identifier ":" tensorType
graphDecl   ::= "graph" identifier "{" statement* "return" identifier "}"
statement   ::= identifier "=" expression
```

An `expression` is one of: an operator call (`opName "(" argList ")"`),
a matmul expression (`identifier "@" identifier`), an elementwise
binary expression (`identifier ("+"|"-"|"*"|"/") (identifier |
floatLiteral)`), or a bare identifier. `return` names an identifier
that a statement already bound. A graph must have exactly one
`return`. It must be the graph's final statement.

## 3.3 Tensor Types

``` text
tensorType ::= "Tensor" "[" axis ("," axis)* "]"
axis       ::= identifier "=" dimSize
dimSize    ::= intLiteral | identifier
```

A tensor type is an ordered, non-empty list of named axes. Every axis
has a name (an identifier) and a size. The size is either a concrete
positive integer or a symbolic identifier. The rank equals the number
of axes. Every tensor holds 32-bit floating-point elements. There is no
element-type annotation in v1 (Section 3.21).

The compiler rejects a concrete axis size of `0` or less. It reports
`SemaNonPositiveDimension`. (A symbolic axis is not checked for a
positive value here, because its concrete value is not known until the
`DimSolver` resolves it.) Axis names must be unique within one tensor
type. The compiler rejects a repeated name with
`SemaDuplicateAxisName` (ARCH-013). This rule applies to every
`tensorType`, and also to `reshape`'s target axis list (Section 3.12).

## 3.4 Symbolic Dimensions

A symbolic dimension is an identifier used as a `dimSize`. Every
occurrence of the same symbolic identifier within one graph names the
same size. The compiler unifies these occurrences. The compiler
considers only these forms of dimension expression: a concrete
integer; a symbolic variable; or a product of a dimension expression
with an integer constant (affine in one variable). The compiler solves
equalities with union-find plus constant arithmetic. If a program's
shape correctness needs reasoning outside this set of forms, the
compiler rejects the program with a static diagnostic. The compiler
never substitutes a run-time shape check instead.

## 3.5 Axis Names

Every axis has a name, in addition to its size. Two axes are
**name-compatible** when their names are the identical identifier.
Sections 3.7 through 3.15 state where the compiler requires
name-compatibility, in addition to size-compatibility. Lowering to
MLIR erases axis names (Section 3.20). Axis names exist for
surface-language checking, for diagnostics, and for `explain` mode.

## 3.6 Name and Scope Rules

An identifier must be declared (`input`/`weight`) or bound (by an
assignment) before its use. The compiler rejects two things: a
redeclared top-level name, and a rebind, in the *same* graph, of a name
already bound to a different declaration kind. The compiler permits
reassigning an already-bound value name within a graph. This
reassignment shadows the prior binding for every later reference
(Section 2.2 of the whitepaper).

There are exactly two scopes: a flat **global** scope (every
`input`/`weight` declaration, visible to every graph in the program)
and one flat **local** scope per graph (that graph's value bindings,
reset at the start of each graph — a value bound in one graph is not
visible in another, Section 2.2). The compiler keeps this environment
in a symbol table. The table maps each declared or bound name to its
kind (`input`/`weight`/a graph-local value), its resolved tensor type,
and its declaration site. Semantic analysis resolves every name
reference against this table (it checks the local scope first, then
the global scope). Every error this section describes — an unresolved
identifier, a rejected redeclaration, a rejected cross-kind rebind — is
exactly a symbol-table lookup or insertion that failed. The command
`tensora --emit-symbols file.tns` prints this table: the global
declarations, each graph's local scope, and the resolution outcome for
every name referenced in that graph's body.

## 3.7 Matrix Multiplication (`@`)

``` text
Tensor[..., i=M, k=K] @ Tensor[..., k'=K, j=N] -> Tensor[..., i=M, j=N]
```

The contracted axes are the last axis of the left operand and the
second-to-last axis of the right operand. Their sizes must be
unifiable. Their names do not need to match. Any leading axes must be
name-compatible and size-compatible under the broadcasting rule of
Section 3.8. The result carries the left operand's row-axis name and
the right operand's column-axis name. Both operands must have rank
`>= 2`.

## 3.8 Elementwise Arithmetic and Broadcasting

`+`, `-`, `*`, `/` accept two tensors, or one tensor and one scalar
literal. For a tensor-tensor operation, the compiler aligns the two
shapes at their trailing axis. For each aligned pair of axes:

-   the axis names must be identical, **unless** one of the two axes
    has concrete size `1` (a broadcast axis) — in that case, the
    compiler does not check names for that pair;
-   the sizes must be equal, must unify, or one of them must be the
    concrete integer `1`.

A shape shorter in rank is treated as having leading axes of size `1`.
These leading axes take their names from the longer operand. Breaking
either the name rule or the size rule is a static error (whitepaper
Section 4.7, `AxisNameError` / `ShapeError`). A tensor-scalar operation
applies the scalar to every element. It places no name or size
constraint on the tensor.

## 3.9 `linear`

``` text
linear(Tensor[..., k=K], Tensor[in=K, out=N], Tensor[out=N]) -> Tensor[..., out=N]
```

All three arguments are required. The bias is not optional in the core
language. The `K` contraction is positional: the sizes must unify, and
the names are unconstrained. The bias tensor's axis must be name- and
size-compatible with the weight's `out` axis. The input's leading axes
stay, with their names unchanged. The result's trailing axis takes the
weight's `out` axis name. The compiler computes this as `x @ W + b`.
The weight layout is `[in, out]`.

## 3.10 `relu`

``` text
relu(Tensor[a1=d1, ..., an=dn]) -> Tensor[a1=d1, ..., an=dn]
```

This computes elementwise `max(x, 0)`. The result type (its names and
sizes) equals the input type exactly.

## 3.11 `softmax`

``` text
softmax(Tensor[...], axis=name) -> Tensor[...]
```

`axis` names an existing axis of the input, by identifier. The result
type equals the input type. The values are normalized along the named
axis.

## 3.12 `reshape`

``` text
reshape(Tensor[...], "[" axis ("," axis)* "]") -> Tensor[axis, ...]
```

The second argument is the complete list of result axes (names and
sizes), written the same way as a `tensorType`. The reshape is valid
only if the product of the result sizes equals the product of the
source sizes. The compiler decides this within the dimension-expression
subset of Section 3.4. A reshape does not require any name
relationship between a source axis and a result axis — it is
explicitly a re-labeling operation.

## 3.13 `flatten`

``` text
flatten(Tensor[a0=d0, a1=d1, ..., an=dn]) -> Tensor[a0=d0, flat=(d1*...*dn)]
```

Axes `a1..an` must all have concrete sizes. `a0` may be symbolic. It
keeps its name. The flattened trailing axis is named `flat`. `flatten`
requires an input of rank 2 or higher — there is no trailing axis to
flatten out of a rank-1 tensor, since a rank-0 (scalar) result is not
a representable `TensorType` (Section 3.3).

## 3.14 `transpose`

``` text
transpose(Tensor[...], perm="[" intLiteral ("," intLiteral)* "]") -> Tensor[...]
```

`perm` must be a permutation of `0 .. rank-1`. It must name every
source axis position exactly once. The result reorders both the sizes
and the names of the input's axes, following `perm`.

## 3.15 `sum` / `mean`

``` text
sum(Tensor[...], axis=name)  -> Tensor[...]   (axis removed)
mean(Tensor[...], axis=name) -> Tensor[...]   (axis removed)
```

`axis` names an existing axis, by identifier. The result is the input
type with that axis deleted. There is no `keepdim` option in v1. The
input must have rank 2 or higher: reducing a rank-1 tensor's only axis
would produce a rank-0 result, and `TensorType` (Section 3.3) cannot
represent one.

## 3.16 `conv2d`

``` text
conv2d(
  Tensor[batch=B, channel=Cin,  height=H, width=Wd],
  Tensor[out_channel=Cout, in_channel=Cin, kh=kH, kw=kW],
  Tensor[out_channel=Cout],
  stride=s, padding=p
) -> Tensor[batch=B, channel=Cout, height=Hout, width=Wout]
```

The layout is fixed to NCHW. The weight layout is fixed to
`[out_channel, in_channel, kh, kw]`. `x`'s `channel` axis size must
unify with the weight's `in_channel` axis size. `stride` (`s`) and
`padding` (`p`) are integer literals, but the two are not symmetric:
`padding` may be zero or positive (the grammar alone guarantees
non-negative, since integer literals have no unary minus), while
`stride` must be strictly positive — `stride <= 0` is a static error.
Dilation is fixed at 1, and groups is fixed at 1.

``` text
Hout = (H  + 2*p - kH) / s + 1
Wout = (Wd + 2*p - kW) / s + 1
```

Both divisions must be exact: integer, with zero remainder. A
configuration that does not divide exactly is a static error. This
arithmetic needs `H`, `Wd`, `kH`, and `kW` to be concrete (not
symbolic) sizes. `conv2d`'s spatial divisibility check falls outside
the bounded symbolic-dimension subset (Section 3.4). So the compiler
rejects a symbolic height, width, or kernel size with
`SemaUnresolvableShape`. It does not defer this check to run time.
`batch` and the channel axes may still be symbolic.

## 3.17 `maxpool`

``` text
maxpool(Tensor[batch=B, channel=C, height=H, width=Wd], size=k, stride=t)
  -> Tensor[batch=B, channel=C, height=Hout, width=Wout]
```

This uses `conv2d`'s spatial formula, with kernel `k`, padding `0`,
and stride `t` (default `t = k`). `k` and `t` must each be strictly
positive — a zero or negative pooling window or stride is a static
error (`SemaConvArithmetic`), the same rule `conv2d`'s `stride` follows
(Section 3.16). It has the same requirement that `H`
and `Wd` must be concrete (Section 3.16). v1 defines only max-pooling.
Average pooling is a future extension.

## 3.18 Graph Semantics

A graph is a directed acyclic dataflow computation. Every assignment
defines a value from its right-hand-side expression. Every use of a
name that a statement already bound creates a dependency edge from
that producing statement. The core language has no mutation and no
control flow — no conditionals, and no loops.

## 3.19 Static Semantic Failure

Compilation of a graph fails when any of the following holds: the
program uses an undefined name; the program illegally redeclares a
name; a declared or inferred axis size is not a positive integer
(`SemaNonPositiveDimension`) or a `tensorType`/reshape target list
repeats an axis name (`SemaDuplicateAxisName`, Section 3.3); an
operator call has the wrong argument count, the wrong rank, or
a type that breaks its contract (Sections 3.7–3.17); an elementwise
operation breaks broadcasting size or axis-name compatibility (Section
3.8); a `reshape` does not keep the element count; a `transpose`
permutation is invalid; a `sum`/`mean`/`softmax` axis name does not
exist on its operand; or a `conv2d`/`maxpool` configuration has
mismatched channels, a non-positive `stride`/pooling `size`, or spatial
arithmetic that does not divide evenly. A distinct, separately named
failure, `SemaUnresolvableShape`, covers cases the compiler cannot
decide either way from the bounded symbolic-dimension subset (Section
3.4) — an unrelated symbolic reshape factor (Section 3.12) or a
symbolic spatial/kernel size reaching `conv2d`/`maxpool` (Sections
3.16–3.17) — as opposed to the failures above, all of which the
compiler does decide, and rejects because the decision came out
negative. The compiler reports every failure in the diagnostic form of
whitepaper Section 4.7.

## 3.20 Typed Computation Graph

A successfully checked graph becomes a typed computation graph. Each
node records: its operator kind; its operand and result value
identifiers; the verified result type (ordered named axes with sizes);
its source location; and any semantic facts kept for later passes
(whitepaper Section 6.1). Axis names are a property of this typed
graph and of the surface language. Lowering the graph to MLIR erases
axis names — axes become positional (whitepaper Section 7.2).

## 3.21 Optimization Semantics

Every optimization pass must keep `return`'s observable output the
same, for every input that satisfies the checked program's semantics —
exactly the same, not just the same up to floating-point rounding. The
frozen v1 pass set is exactly the six passes of whitepaper Section 6.2
(shape folding, dead-operation elimination, reshape cancellation,
transpose cancellation, elementwise fusion, linear-bias-activation
fusion). A pass beyond this set is a future extension. It is not part
of the frozen v1 language.

A constant-folding pass was considered for v1. This pass would
combine a chain of same-kind scalar-attribute elementwise ops (for
example, `(x * 2.0) * 3.0` -> `x * 6.0`) into one node. v1
rejects this pass. Combining the two scalars at compile time,
then applying the result in one run-time multiply, is not guaranteed
to equal the unfolded two-step computation. IEEE-754 f32 multiplication
is not associative: the unfolded path rounds the intermediate tensor
value after the first operation, and the folded path does not round
it. This is a real break of this section's requirement. It is not a
rounding difference inside a tolerance. So v1 excludes this pass. It is
not part of v1 with a documented exception.

## 3.22 Execution Model

The required path is exactly:

``` text
Tensora source -> Typed Computation Graph -> MLIR (linalg/tensor/arith/math/func/scf) -> LLVM -> native CPU executable
```

v1 defines no custom Tensora MLIR dialect (whitepaper Section 12).
Lowering targets existing MLIR dialects directly. GPU lowering is a
future extension. v1 never requires it.

## 3.23 Excluded Semantics

The core language defines no automatic differentiation, no
backpropagation, no optimizer updates, no distributed execution, no
dataset loading, no dynamic control flow, and no element type other
than f32. Whitepaper Section 12 lists these as future extensions. They
are not optional core features.

## 3.24 Grammar Summary

This section collects the complete v1 grammar in one place, in the
spirit of Appendix A.13 of Kernighan and Ritchie. It is the
authoritative syntax. Sections 3.1–3.3 give the prose form. Notation:
`x*` means zero or more, `x+` means one or more, `x?` means optional,
`|` separates alternatives, and `"x"` is a literal token.

### Lexical grammar

``` text
token        ::= keyword | identifier | intLit | floatLit | operator | delimiter
keyword      ::= "input" | "weight" | "graph" | "return" | "Tensor"
identifier   ::= [A-Za-z_][A-Za-z0-9_]*
intLit       ::= [0-9]+
floatLit     ::= [0-9]+ "." [0-9]+
operator     ::= "+" | "-" | "*" | "/" | "@" | "="
delimiter    ::= ":" | "," | "[" | "]" | "{" | "}" | "(" | ")"
comment      ::= "#" (any char except newline)*          # skipped
whitespace   ::= (" " | "\t" | "\r" | "\n")+             # skipped, not significant
```

An `identifier` that matches a `keyword` is that keyword. An axis name
is an ordinary identifier.

### Syntactic grammar

``` text
program      ::= decl* graphDecl+

decl         ::= inputDecl | weightDecl
inputDecl    ::= "input"  identifier ":" tensorType
weightDecl   ::= "weight" identifier ":" tensorType

tensorType   ::= "Tensor" "[" axis ("," axis)* "]"
axis         ::= identifier "=" dimSize
dimSize      ::= intLit | identifier          # concrete size or symbolic dimension

graphDecl    ::= "graph" identifier "{" statement* returnStmt "}"
statement    ::= identifier "=" expression    # no terminator; expression self-delimiting
returnStmt   ::= "return" identifier

expression   ::= callExpr
              |  identifier binop operand      # single binary operation
              |  identifier                    # a bare value reference
operand      ::= identifier | floatLit         # tensor value or scalar literal
binop        ::= "@" | "+" | "-" | "*" | "/"

callExpr     ::= identifier "(" arg ("," arg)* ")"
arg          ::= identifier                    # positional tensor operand
              |  identifier "=" argValue       # keyword argument
              |  "[" axis ("," axis)* "]"      # reshape target axis list
argValue     ::= intLit                        # e.g. stride=1, padding=1, size=2
              |  identifier                     # e.g. axis=kseq
              |  "[" intLit ("," intLit)* "]"   # e.g. perm=[0,1,3,2]
```

### Notes on the grammar

- The grammar is LL(1). One token of lookahead is enough: the parser
  recognizes a `statement` by `identifier "="`, and a `returnStmt` by
  `"return"`.
- The grammar does not encode operator argument well-formedness — which
  operators take which keyword arguments, and what kind each argument
  must be. That is a static-semantic rule (Sections 3.7–3.17). The
  parser accepts any `arg` shape. The semantic analyzer checks each one
  against its operator's rule.
- All other correctness properties are static semantics (Sections
  3.4–3.19), not syntax: axis names, sizes, ranks, contraction
  compatibility, broadcasting, element counts, permutations, axis
  existence, and channel and spatial arithmetic.
