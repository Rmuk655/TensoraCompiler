# Tensora: A Small, Strongly Typed Tensor Language for Semantic Tensor Types and ML Compiler Transformations

## Abstract

Tensora is a small, strongly typed tensor language. It lets a team
experiment with semantic tensor types, static reasoning, and ML
compiler transformations. The central idea is this: the language makes
high-level semantic information about tensors explicit. Static
analysis establishes this information. A typed compiler intermediate
representation (IR) keeps this information while it is useful.
Diagnostics, analyses, and transformations use it. The compiler lowers
or erases this information once it is no longer needed. Named,
symbolic tensor axes are the v1 vehicle for this idea. They are an
important feature, not the whole identity of the language. They let
the compiler tell apart two tensors that share a shape but not a
meaning. They also let static reasoning talk about dimensions
symbolically, not only as concrete integers.

A Tensora program declares the tensors that enter a computation and
the operators that transform them. The compiler decides whether the
resulting graph is semantically valid. It infers the type of every
intermediate tensor by symbolic reasoning, and records these facts in
a typed computation graph. It applies team-authored graph-level
transformations, informed by that semantic information (including a
real data-flow analysis and a shape-guided loop transformation). It
lowers valid programs through MLIR and LLVM to native CPU code.

Version 1 of the language targets forward computation for inference.
Training, automatic differentiation, optimizer updates, distributed
execution, and production model serving are explicitly out of scope.
This boundary keeps the language small. It still exposes real compiler
problems: symbolic shape reasoning, semantic-axis contracts, dataflow
graph construction, optimization legality, lowering, and native code
generation. The project bounds this scope to what one academic compiler
effort can build, test, and defend end to end. The scope is not wider than this
just to make the project look more original.

Tensora does not claim that tensors, shape types, symbolic dimensions,
named axes, computation graphs, static shape inference, SSA, operator
fusion, loop tiling, or MLIR/LLVM lowering are new ideas. Each of these
exists in prior systems. Named axes appear in PyTorch Named Tensors
and in research languages such as Dex. Symbolic and dynamic shapes
appear in TensorFlow, ONNX, and PyTorch's dynamic-shape tracing.
SSA-form IRs and loop tiling are standard compiler techniques. MLIR
and LLVM are existing infrastructure; this project builds on them, and
does not reinvent them. As an academic compiler, Tensora's contribution is this: a
coherent, complete, end-to-end compiler, in which an original
declarative source language is checked by a from-scratch semantic
analyzer. The compiler explains the semantic information it
establishes. It keeps that information in a typed IR through a real
data-flow analysis and a shape-guided structured-loop transformation.
It lowers the result to executable code through standard compiler
infrastructure. The implementation aims to be readable, testable, and
extensible: an educational compiler, an open-source testbed for ML
graph passes, and a possible starting point for later research on
layout, fusion, memory, and accelerator lowering.

--------------------------------------------------------------------------

# 1. Introduction to Tensora

## 1.1 Beginnings of the Tensora Project

Machine-learning models are usually written in a general-purpose
language, such as Python, using a library such as PyTorch, TensorFlow,
or JAX. This is productive. But the structure of the ML computation
sits inside a much larger and more general programming environment.
The programmer discovers many errors in that structure only at run
time. The tensor semantics that matter for correctness — which axis
means what, which dimensions must agree, which reshape is actually
valid — exist only in the programmer's head and in runtime values. The
compiler cannot check any of this ahead of time.

Tensora starts from a different question: **what can a compiler do
when tensor semantics are explicit, statically checked, and kept
through a typed intermediate representation?** The forward computation
of many models is a directed graph. Its nodes are tensor operators,
and its edges are tensor values:

``` text
input --> Linear --> ReLU --> Linear --> output
```

Once the program states this structure explicitly, with each axis
carrying a name in addition to a size, a compiler can reason about it
before the program runs. The compiler can reject incompatible matrix
dimensions and incompatible axis meanings. It can infer the type of
every intermediate tensor, find dead operations, cancel inverse
reshapes and transposes, and find fusible regions. It can feed a real
liveness and peak-memory analysis, use verified shapes to guide a loop
transformation, and lower the verified graph to lower-level
representations. The compiler keeps this semantic information
available to diagnostics and passes for as long as it is useful. It
erases the information only at the point where a lower representation
(MLIR, then LLVM) has no use for it.

Tensora does not aim to replace production ML frameworks. It is a
small language, built on purpose, where a student team can see and
build these compiler ideas end to end in one academic compiler project.

## 1.2 The Problem: Programming ML Computation Graphs

Consider a matrix multiplication:

``` text
Tensor[batch=B, feat=784] @ Tensor[in=784, out=256]   ->   Tensor[batch=B, out=256]
```

The operator alone determines the result. But this expression is
invalid:

``` text
Tensor[batch=B, feat=784] @ Tensor[in=512, out=256]
```

It is invalid because the contracted dimensions (784 and 512) differ.
This is not a parsing problem. Both expressions are syntactically well
formed. It is a semantic property of matrix multiplication.

The same pattern appears throughout ML computation:

-   A reshape must keep the total number of elements the same.
-   An elementwise operation on two tensors must satisfy broadcasting
    compatibility, in both size and axis meaning.
-   A linear layer's input feature dimension must match its weight.
-   A convolution's parameters determine its output spatial
    dimensions.
-   A transpose permutation must be a permutation of the tensor's
    axes.
-   A reduction axis must exist in the tensor.
-   A graph's output depends on a chain of typed intermediate values.

Tensora makes each of these properties part of the language's static
semantics.

## 1.3 Design Goals

**Simple and declarative.** A program states *what* the computation
is, not how memory is laid out or how kernels are launched.

**Statically verified.** The compiler establishes any operator
property it can prove from the program, before execution. A violation
names the rule and the values involved.

**Named and meaningful.** Every axis has a name. This lets the
compiler tell apart two tensors that share a shape but not a meaning.
It also lets the compiler report errors and explanations in the
vocabulary of the model.

**Explainable.** The compiler makes the types and constraints it
infers visible to the programmer. This lets the language teach how
tensor computation flows through a model.

**Optimizable.** The source exposes graph structure. The compiler
keeps enough verified information to run explicit, testable
graph-level transformations before lowering.

**Extensible.** Signatures and semantic rules describe each operator.
A team can add a new operator without redesigning the compiler.

**Architecture neutral.** The source encodes no target instruction
set. The required implementation lowers through MLIR and LLVM to
native CPU code. Accelerator lowering is a future extension.

## 1.4 The Compilation Environment

``` text
Tensora source
   -> Lexer / Parser
   -> Abstract Syntax Tree
   -> Semantic Analysis (named-axis symbolic shape + operator contracts)
   -> Typed Computation Graph
   -> Graph Optimization (team-authored passes)
   -> MLIR (linalg / tensor / arith / func / scf)
   -> LLVM
   -> Native CPU executable
```

The command line can show every intermediate form (Section 11). This
supports both grading demonstrations and debugging.

## 1.5 Scope and Non-Goals

Version 1 describes forward ML computation only. It does **not**
provide automatic differentiation, backpropagation, optimizer updates,
training loops, dataset loaders, checkpointing, distributed execution,
production serving, or mandatory GPU support. The team excludes these
on purpose. They keep the academic compiler *complete*, instead of
turning it into an incomplete ML framework. Section 12 lists what is
core and what is a future extension.

--------------------------------------------------------------------------

# 2. Tensora is Simple and Declarative

## 2.1 A First Program

``` tensora
input  x:  Tensor[batch=B, feat=784]
weight W1: Tensor[in=784, out=256]
weight b1: Tensor[out=256]
weight W2: Tensor[in=256, out=10]
weight b2: Tensor[out=10]

graph classifier {
    h = linear(x, W1, b1)
    h = relu(h)
    y = linear(h, W2, b2)
    return y
}
```

The program declares external tensors and one graph. The compiler
infers the full type of every intermediate value:

``` text
x : Tensor[batch=B, feat=784]
h : Tensor[batch=B, out=256]     (after linear, unchanged by relu)
y : Tensor[batch=B, out=10]
```

## 2.2 Program Structure

A Tensora program is a sequence of top-level declarations, followed by
one or more graphs.

-   `input name: Tensor[...]` declares data the program supplies to a
    graph at execution time.
-   `weight name: Tensor[...]` declares an externally initialized
    model parameter (a trained weight the program loads before
    execution).
-   `graph name { ... }` declares a named dataflow computation.

Inside a graph:

-   An **assignment** `v = expr` binds the name `v` to the value of
    `expr`. A name may be reassigned. A reassignment shadows the
    previous binding for every later reference. The compiler never
    mutates a tensor value in place, so the computation stays pure
    dataflow. (The convenient `h = relu(h)` idiom simply rebinds `h`
    to a new value.)
-   `return v` names the graph's observable output. A graph has
    exactly one `return`.

## 2.3 Tensor Values and Named Axes

The tensor is the language's single data abstraction. Its type is an
ordered list of axes. **Every axis has a name and a size**:

``` text
Tensor[name1 = size1, name2 = size2, ..., nameN = sizeN]
```

with rank `N >= 1`. An axis name is an identifier. A size is either a
**concrete** integer (`784`) or a **symbolic** dimension variable
(`B`). Every tensor in v1 holds 32-bit floating-point elements. The
surface syntax does not write the element type (Section 12 describes
a future dtype system).

``` text
Tensor[rows=32, cols=784]                      rank 2, fully concrete
Tensor[batch=B, feat=784]                      rank 2, symbolic batch
Tensor[batch=B, channel=3, height=224, width=224]   rank 4 (RGB images)
```

Axis **names** let the compiler tell apart tensors that share a shape
but not a meaning: `Tensor[batch=32, feat=32]` and
`Tensor[feat=32, batch=32]` have the same sizes, but the compiler does
not treat them as interchangeable. Axis **sizes**, when symbolic, let
one program describe a whole family of shapes (any batch size `B`).
These sizes take part in the constraints operators generate (Section
4). Axis names are ordinary identifiers the programmer chooses. They
are not reserved keywords. The names this document uses (`batch`,
`feat`, `in`, `out`, `channel`, and so on) are conventions, not
language builtins.

## 2.4 Scalar Constants

A floating-point or integer literal may appear where an operator or an
elementwise expression needs a scalar (for example, a scale factor).
Scalars are not first-class tensors in v1. They appear only as
operator arguments and in scalar-tensor elementwise expressions.

## 2.5 The Core Operator Set

Section 4 defines the core operators precisely. They are:

| Tier | Operators | Role in v1 |
|------|-----------|------------|
| Core | `+  -  *  /`, `@` (matmul), `linear`, `relu`, `reshape`, `transpose`, `sum`, `mean` | built well and carried through the entire pipeline, including loop-level optimization |
| Strong extension | `conv2d`, `maxpool`, `flatten` | fully verified; enables the small-CNN workload |
| Showcase | `softmax`, `attention` (composed from core ops) | one worked example each; verified and lowered, but not the loop-optimization focus |

This set is frozen for v1. v1 optimizes for **depth over breadth** on
purpose: it compiles a small operator set unusually far down the
stack — through a typed SSA graph, a tensor-liveness analysis, MLIR,
and a structured-loop transformation (tiling) — instead of listing a
large catalogue of operators that only become library calls. Matrix
multiplication is the reference computation carried through every
level (Section 8). Activations and layers beyond this set (`gelu`,
`layernorm`, an axis `rename`, a complete Transformer encoder layer)
are future scope (Section 12).

## 2.6 Features Deliberately Omitted

Following the tradition of stating what a language leaves out:

-   **No explicit memory management.** Allocation is a backend
    concern.
-   **No pointers or aliasing.** The language expresses computation
    over typed values, not addresses.
-   **No in-place mutation.** The core graph is pure dataflow. This
    keeps dependency analysis and optimization tractable.
-   **No training constructs.** Gradients and parameter updates need
    more semantics, and are out of scope.
-   **No explicit parallelism.** The source states the computation.
    The backend decides on parallel execution.

--------------------------------------------------------------------------

# 3. Tensora is a Computation-Graph Language

## 3.1 The Graph Model

A Tensora graph is a directed acyclic graph. Nodes are operator
applications. Edges are tensor values. If operation `B` consumes a
value that operation `A` produced, then `B` depends on `A`. The
programmer never schedules operations by hand. Data dependence implies
the order.

``` text
x --> Linear --> ReLU --> Linear --> y
        ^                   ^
        |                   |
        W1,b1               W2,b2
```

## 3.2 Dataflow Semantics

The value an operation produces is a pure function of its input values
and the operator's semantics. The core has no side effects and no
mutation. So two graphs with the same dataflow compute the same
result. A transformation is legal exactly when it keeps the value that
returns for every valid input the same. This property is what makes
the optimization phase (Section 6) possible, and lets the team check
it.

## 3.3 Why Explicit Graph Structure Matters

Making the graph a first-class language object gives the compiler
direct access to dependency information, verified types, dead
subgraphs, local rewrite patterns, fusible regions, and a clean
lowering boundary. In a general-purpose language, the compiler must
first recover this structure from control flow and library calls. In
Tensora, this structure is the program.

--------------------------------------------------------------------------

# 4. Tensora is Statically Verified and Robust

This is the language's largest phase, and its main source of compiler
work.

## 4.1 Symbol Table and Name Resolution

Every identifier a graph uses must be a declared `input`, a declared
`weight`, or a name a statement bound earlier in the same graph.
Redeclaring a top-level name is an error. Using a name before the
program binds it is an error.

The compiler implements this rule; it is not only specified.
`SymbolTable` (`include/tensora/semantic/Symbol.h`,
`src/semantic/Symbol.cpp`) is the compiler's environment. It maps
every identifier to its declared or inferred semantic information: its
kind (`Input`/`Weight`/a graph-local `Value`), its resolved tensor
type, and its declaration span. Every check in this section is exactly
an operation on this table, not a separate mechanism:

-   **Scopes.** There are two levels, matching the grammar exactly:
    one flat **global** scope (`input`/`weight` declarations, shared
    by every graph, filled once before the compiler analyzes any
    graph) and one flat **local** scope per graph (value bindings from
    that graph's statements, reset at the start of each graph through
    `enterGraph()`). A local value from one graph is never visible in
    another.
-   **Insertion.** `declareGlobal` inserts a global declaration. It
    rejects, and does not overwrite, a duplicate name — RM 3.6's
    redeclaration error. `bindLocal` inserts or updates a graph-local
    binding: the compiler accepts a rebind of an existing *local*
    value, and this rebind shadows the value (later statements in that
    graph can no longer resolve the prior binding). The compiler
    rejects a rebind of a name that is already a *global* input or
    weight, as a kind conflict — also an RM 3.6 redeclaration error.
-   **Lookup.** `resolve` is the one path every operand reference in
    the language actually calls: it checks the local scope first, then
    the global scope, and returns `nullptr` for an unresolved
    identifier — RM 3.6's "using a name before it is bound" error.
-   **Type association.** Every `Symbol` carries the type semantic
    analysis has already established for it — the verified declaration
    type for a global, or the operator contract's inferred result type
    for a local value. So name resolution and type information come
    from the same lookup, not from two passes that could disagree.
-   **Relationship to the typed graph.** The symbol table exists only
    during semantic analysis. It resolves source-level names to the
    `ValueId`s the Typed Computation Graph actually stores. The table
    is not itself part of the TCG, and no later phase — optimization,
    lowering, or anything after — consults it. By the time the TCG
    exists, the compiler has already resolved every name into it.

`tensora --emit-symbols file.tns` renders this table directly from the
same `SemanticAnalyzer` state that did the resolution. It does not
build a second, separate table. The output includes a per-graph
resolution log. This log shows what each referenced name actually
resolved to:

``` text
=== Symbol Table ===

GLOBAL
  x   input   Tensor[batch=B, feat=784]  (2:1)
  W1  weight  Tensor[in=784, out=256]    (3:1)
  b1  weight  Tensor[out=256]            (4:1)

GRAPH mlp
  h   local   Tensor[batch=B, out=256]   (10:5)
  y   local   Tensor[batch=B, out=10]    (11:5)
  RESOLVED
    x -> global::x
    W1 -> global::W1
    h -> local::h
    y -> local::y

=== End Symbol Table ===
```

Section 8's test suite exercises the table directly (`SYM-001`..
`008`). It covers declaration-order output, per-graph scope isolation,
shadowing that collapses to one entry, and both error cases (an
unresolved identifier, a rejected redeclaration) that leave no invalid
state behind.

## 4.2 Rank and Operator-Domain Checking

Each operator accepts operands of specific ranks. `conv2d` and
`maxpool` require rank-4 tensors. `@` requires rank `>= 2`. A
reduction requires its named axis to exist. The compiler reports a
violation before shape inference proceeds.

## 4.3 The Tensor Type and Symbolic Shape Inference

A tensor type is an ordered list of `(name, size)` axes. Two axes are
**name-compatible** when their names are the same identifier. Sizes
are **dimension expressions**, drawn from a deliberately restricted,
decidable grammar:

-   a concrete integer;
-   a symbolic dimension variable;
-   a product of a dimension expression with an integer constant.

So v1 only reasons about sizes that are a symbolic variable times an
integer constant, plus plain integer arithmetic. It does **not**
attempt general nonlinear constraint solving or SMT solving. If a
program needs more than this, the compiler rejects it with an explicit
"cannot statically resolve shape" diagnostic, rather than guessing.
This bound is what keeps the checker decidable.

Shape inference assigns a full type (names and sizes) to every
intermediate value. It does this by applying the operator contracts of
Section 4.5.

## 4.4 Broadcasting and Axis-Name Compatibility

A tensor-tensor elementwise operator broadcasts by NumPy rules,
extended to check axis names. The compiler aligns the two shapes at
their trailing axis. For each aligned pair of axes `(a, b)`:

-   the names must be name-compatible, **unless one of the two axes
    has concrete size `1`** (a broadcast axis) — in that case, the
    compiler exempts the pair from the name check. A name mismatch on
    a non-broadcast pair is an error, even when the sizes agree.
-   for the sizes: if they are equal, or they unify, the result size
    is that size. If one of them is the concrete integer `1`, the
    result size is the other size. Otherwise, the compiler rejects the
    operation.

A shape shorter in rank is treated as having leading axes of size `1`.
These leading axes take their names from the longer operand. A
tensor-scalar operation applies the scalar to every element, and does
not constrain names.

Axis-name checking is why the compiler rejects, at compile time, a
program that accidentally combines a `[batch, feat]` tensor with a
`[feat, batch]` tensor — even though both tensors have the same sizes.

## 4.5 Operator Contracts

Each operator has a signature. The signature states its input
requirements and its output type. The following are normative for v1.
Axis names shown here (`in`, `out`, `batch`, ...) are illustrative; a
contract constrains names only where this section states it.

**Matrix multiplication `@`.** With `A : Tensor[..., i=M, k=K]` and
`B : Tensor[..., k'=K, j=N]`, the contraction is positional: the last
axis of `A` and the second-to-last axis of `B` must have unifiable
**sizes** (their names do not need to match, since the compiler
consumes them). Any leading axes must be name-compatible and
size-compatible (Section 4.4). The result is `Tensor[..., i=M, j=N]`.
It carries `A`'s row-axis name and `B`'s column-axis name.

**`linear(x, W, b)`.** With `x : Tensor[..., k=K]`,
`W : Tensor[in=K, out=N]`, and `b : Tensor[out=N]`, the result is
`Tensor[..., out=N]`, defined as `x @ W + b`, with the bias broadcast
over the leading axes. The contraction on `K` is positional (the sizes
unify). The bias axis must be name- and size-compatible with `W`'s
output axis. The result's last-axis name comes from `W`'s output
axis. `x`'s leading axes stay, with their names unchanged. Tensora
stores the weight as `[in, out]` (so the program writes `x @ W`
directly). Note that this is the transpose of PyTorch's `nn.Linear`,
which stores `[out, in]`.

**`relu(x)`.** This computes elementwise `max(x, 0)`. The result type
equals the input type.

**`softmax(x, axis=a)`.** The result type equals the input type. `a`
names an existing axis (by name); the compiler normalizes values along
this axis.

**`reshape(x, [n1=s1, ...])`.** The programmer supplies the full list
of new named axes and their sizes. The product of the target sizes
must equal the product of the source sizes. Under the
affine-in-one-variable restriction, the compiler checks this by
matching the symbolic factor on both sides, and comparing the
concrete products.

**`flatten(x)`.** For `x : Tensor[a0=d0, a1=d1, ..., an=dn]`, this
produces `Tensor[a0=d0, flat=(d1 * ... * dn)]`. Axes `a1..an` must
have concrete sizes, so their product is a concrete integer. `a0` may
be symbolic.

**`transpose(x, perm=[...])`.** `perm` is a permutation of
`0..rank-1`. The result reorders both the axes and their names, by
`perm`.

**`sum(x, axis=a)` / `mean(x, axis=a)`.** `a` names an existing axis.
The result is `x` with that axis removed.

**`conv2d(x, W, b, stride=s, padding=p)`.** With input
`x : Tensor[batch=B, channel=Cin, height=H, width=Wd]`, weight
`W : Tensor[out_channel=Cout, in_channel=Cin, kh=kH, kw=kW]`, and bias
`b : Tensor[out_channel=Cout]`, the input-channel sizes must unify
(`x.channel` with `W.in_channel`). The output is
`Tensor[batch=B, channel=Cout, height=Hout, width=Wout]`, where

``` text
Hout = (H  + 2*p - kH) / s + 1     (integer division, must divide evenly)
Wout = (Wd + 2*p - kW) / s + 1
```

`stride` and `padding` are non-negative integer literals. Dilation is
fixed at 1, and groups is fixed at 1, in v1.

**`maxpool(x, size=k, stride=t)`.** With
`x : Tensor[batch=B, channel=C, height=H, width=Wd]`, the output is
`Tensor[batch=B, channel=C, height=Hout, width=Wout]`, using the same
formula, with kernel `k`, padding `0`, and stride `t` (default
`t = k`).

## 4.6 Representative Compile-Time Errors

Incompatible matmul sizes; an axis-name mismatch in an elementwise
operation; an invalid reshape element count; a nonexistent reduction
or transpose axis; a convolution channel mismatch or spatial
arithmetic that does not divide evenly; an undefined name; and a
redeclaration.

## 4.7 Diagnostic Design

Every diagnostic reports the source location, the rule it violates,
the required property, and the actual property. All phases share one
`Diagnostic`/`SourceSpan` facility.

``` text
ShapeError at classifier.tns:7:9

    y = x @ W

cannot multiply Tensor[batch=B, feat=784] by Tensor[in=512, out=256]

    required: contracted sizes must be equal
    found:    784 != 512
```

``` text
AxisNameError at model.tns:5:9

    z = a + b

cannot add Tensor[batch=32, feat=32] and Tensor[feat=32, batch=32]

    required: aligned axes must share a name
    found:    axis 0 is 'batch' in the left operand but 'feat' in the right
```

--------------------------------------------------------------------------

# 5. Tensora is Explainable

The semantic phase already computes the type of every value, and the
constraints it proved. So the compiler can show this information
directly:

``` text
$ tensora explain classifier.tns

graph classifier
  x : Tensor[batch=B, feat=784]     (input)
  h : Tensor[batch=B, out=256]      linear(x, W1, b1)   feat=784 unifies with W1.in
  h : Tensor[batch=B, out=256]      relu(h)             type preserved
  y : Tensor[batch=B, out=10]       linear(h, W2, b2)   size 256 unifies with W2.in
  return y : Tensor[batch=B, out=10]
```

`explain` mode needs no separate analysis engine. It reuses the facts
the checker already produced. For a rejected program, it prints the
failed contract instead. This is the honest answer to "why write this
instead of PyTorch?": Tensora verifies the graph, checks axis meaning,
and *explains* the reasoning before anything runs. It is a tool for
learning and for compiler experimentation, not a production framework
(Section 9).

--------------------------------------------------------------------------

# 6. Tensora is Optimizable

## 6.1 The Typed Computation Graph

After semantic analysis, the compiler represents the program as a
typed computation graph, in **static single assignment (SSA)** form:
each value has exactly one producing operation, so def-use
relationships are explicit. Each node carries its operator kind, its
operands, its verified output type (named axes and sizes), its
symbolic dimensions, and its source location. This graph — not the
AST, and not MLIR — is the object the team's optimizations and
analyses operate on. A convenient source name (`h = relu(h)`) becomes
a distinct SSA value internally. SSA is the standard substrate for the
data-flow analyses below; the project does not claim SSA as novel.

## 6.2 Team-Authored Graph Passes

The team implements the following passes. They run on the typed graph:

1.  shape folding (replace computed-but-constant sizes with literals);
2.  dead-operation elimination (drop nodes that do not reach
    `return`);
3.  reshape cancellation (a `reshape` immediately undone);
4.  transpose cancellation (a permutation followed by its inverse);
5.  elementwise fusion (merge a chain of elementwise ops into one
    region);
6.  linear-bias-activation fusion (recognize the `linear` then `relu`
    pattern as a single fused region for lowering).

The team also implemented a constant-folding pass (combining a chain
of same-kind scalar ops, for example, `(x * 2.0) * 3.0` -> `x * 6.0`),
then retired it. The team could not prove that this pass keeps
`return`'s observable output the same under IEEE-754 f32
non-associativity. Section 3.21 of the Reference Manual requires that
guarantee from every enabled v1 pass.

Each pass has a stated legality condition. Before/after graph tests
validate each one (Section 8).

## 6.3 Tensor-Liveness Analysis

Beyond pattern rewrites, v1 implements one genuine **data-flow
analysis**, the way a compilers class teaches it. Over the SSA
graph, tensor **liveness** computes the live interval of every value
(from its definition to its last use). Every value's byte size is
known from its verified shape (`bytes = product(dims) * 4` for f32).
So liveness yields a **peak intermediate-memory estimate**: the
maximum, over program points, of the total size of tensors that are
live at the same time. This is an *analysis that feeds a decision* —
for example, it can choose a fusion or a schedule that lowers peak
memory — not a bare rewrite. It is the clearest link between Tensora's
verified tensor semantics and classical compiler analysis.

## 6.4 Two Levels of Optimization

Tensora optimizes at two levels:

- **Level 1 — ML graph.** The team-authored passes above run over the
  SSA typed graph. They use ML-level structure (fusion, dead-op
  elimination, reshape/transpose cancellation) and the liveness
  analysis.
- **Level 2 — structured loops.** After lowering `@` to
  `linalg.matmul`, and then to `scf` loops, Tensora applies a
  team-driven loop transformation (**tiling**, Section 8.4). Tensora's
  contribution is the *decision*: it selects the transformation and
  its tile sizes from verified shape properties, and requests the
  transformation. MLIR performs the mechanical rewrite. The team does
  **not** claim MLIR's or LLVM's built-in optimizations (bufferization,
  standard scalar passes, instruction selection, register allocation)
  as its own work.

## 6.5 Measuring Optimization

The evaluation records the graph-node and intermediate-tensor counts,
and the peak-memory estimate, before and after the Level-1 passes. It
records the emitted MLIR and loop structure, before and after the
Level-2 tiling. It records the native execution time. The project
makes a performance claim only where it has a measurement to back it.

--------------------------------------------------------------------------

# 7. Tensora is Architecture Neutral and Portable

## 7.1 Separating Language From Target

The source encodes no CPU instruction set and no accelerator kernel.
The same program lowers to whatever target the backend supports.

## 7.2 Lowering to MLIR

Tensora lowers its typed graph into standard MLIR dialects:

-   A graph becomes a `func.func` that operates on `tensor` values.
-   `@` becomes `linalg.matmul` (batched, via `linalg.generic`, when
    leading axes are present).
-   `conv2d` becomes `linalg.conv_2d_nchw_fchw`.
-   Elementwise operators, `relu`, and bias addition become
    `linalg.generic` bodies over `arith` operations.
-   The compiler emits `softmax` as its standard numerically stable
    sequence (subtract the axis max, exponentiate, sum, divide),
    expressed with `linalg` and `arith`.
-   `reshape`, `flatten`, and `transpose` become `tensor` reshape and
    transpose operations.
-   `sum`/`mean` become `linalg.reduce`.

Lowering `@` (matmul) to `linalg.matmul`, and then to `scf` loops, is
the entry point for v1's structured-loop optimization (Section 8.4):
the team-authored tiling transformation applies to the loops
`for i, j, k: C[i,j] += A[i,k] * B[k,j]`.

Axis names are a source-level and typed-graph concept. The compiler
uses them for checking, diagnostics, and explanation. Lowering to MLIR
erases axis names; dimensions become positional there.

## 7.3 From MLIR to Native Code

The compiler bufferizes the tensor-level module to `memref`, converts
it to loops (`scf`), and lowers it through the standard MLIR
conversion passes to the LLVM dialect. `mlir-translate` produces LLVM
IR, and the LLVM backend produces a native CPU object. CPU execution
is the required end-to-end target.

## 7.4 Future GPU Execution

GPU lowering, through MLIR's GPU and NVVM infrastructure, is a future
extension. It must never endanger the required CPU path.

--------------------------------------------------------------------------

# 8. Compilation Model and Performance

## 8.1 The Full Pipeline

``` text
source -> AST -> typed SSA graph -> (graph passes + liveness) -> MLIR
       -> structured loops -> (tiling) -> LLVM -> native CPU
```

The frontend establishes structure. The semantic phase establishes
legality and types. The team's Level-1 passes and liveness analysis
transform and measure the verified graph. MLIR lowers the graph to
structured loops, where the team's Level-2 tiling decision applies.
MLIR and LLVM generate code. The project reports structural measures
and execution time. It draws a clear line between transformations the
team implemented and optimization the underlying infrastructure
performed.

## 8.2 Frontend and Semantic Framing

One ANTLR4 grammar (`grammar/Tensora.g4`) generates the lexer and
parser. So the frontend phase directly shows grammar-driven code
generation: lexical rules, a syntactic grammar, and generator-produced
recognizers. AST construction stays in project-owned C++
(`src/parser/AstBuilder.cpp`), not in embedded grammar actions.
Semantic analysis is organized as an **attribute grammar**: each AST
node synthesizes its tensor type from its children. For `A @ B`, the
children synthesize `A.shape` and `B.shape`; the node checks
`A.shape[-1] == B.shape[-2]`, and synthesizes the result shape. This
is a textbook semantic-analysis formulation, applied to tensors.

## 8.3 Matrix Multiplication: the Reference Computation

One computation, `C = A @ B`, runs through **every** level of the
compiler. It is the project's clearest demonstration of end-to-end
compilation:

``` text
source     C = A @ B
semantic   A: Tensor[i=M, k=K], B: Tensor[k=K, j=N]  =>  C: Tensor[i=M, j=N]
SSA graph  %C = matmul %A, %B
MLIR       linalg.matmul
loops      for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
Level-2    tiled loops (Section 8.4)
LLVM       LLVM IR
machine    native CPU
```

## 8.4 Structured-Loop Tiling (Level-2 optimization)

Once matmul is lowered to `scf` loops, Tensora applies **loop
tiling**:

``` text
for i: for j: for k: C[i,j] += A[i,k]*B[k,j]
    ->
for ii: for jj: for kk:                     # tile loops
    for i in tile(ii): for j in tile(jj): for k in tile(kk):
        C[i,j] += A[i,k]*B[k,j]
```

The team's contribution is explicit, and separate from the
infrastructure: Tensora's pass **decides whether to tile, and picks
tile sizes from the verified shapes** (the concrete or symbolic `M`,
`N`, `K`), then requests the transformation. MLIR's transform
infrastructure performs the mechanical rewrite. The project benchmarks
the tiled and untiled versions, to report the effect. This connects
Tensora, as an academic compiler, directly to the loop-transformation and polyhedral
material it is meant to teach, without claiming to reimplement MLIR's tiling mechanics.

## 8.5 Performance Evaluation

The project reports: Level-1 graph and intermediate-tensor counts, and
the peak-memory estimate; Level-2 loop structure and tile sizes; and
wall-clock execution time for tiled versus untiled matmul-heavy
workloads on CPU, with the machine, compiler version, and Tensora
commit recorded.

--------------------------------------------------------------------------

# 9. Tensora Compared

## 9.1 General-Purpose Languages

C++, Java, OCaml, and Python express far more than Tensora does.
Tensora restricts what it can express on purpose, so ML graph
semantics stay explicit and statically checkable.

## 9.2 PyTorch, TensorFlow, JAX

These are production frameworks, with mature graph capture, shape
reasoning, and compilation (`torch.compile`/FX/Inductor, XLA). PyTorch
also provides named tensors; Tensora's named axes take direct
inspiration from this feature. Tensora does not compete with or
replace these systems.

## 9.3 MLIR, StableHLO, and Tensor DSLs

MLIR is compiler infrastructure that Tensora *uses*. It is not a
source language Tensora competes with. Progressive lowering that keeps
high-level information is MLIR's own design; Tensora does not claim it
as an invention. StableHLO already specifies operator-level shape
constraints. Shape-typed and array languages (Futhark, Dex, Remora,
typed tensor libraries) already provide static shape and size
checking. Tensor DSLs and loop-optimizing systems — Tensor
Comprehensions, Halide, TACO, PolyMage — already provide
high-performance tensor code generation, tiling, and polyhedral
scheduling. Tensora claims none of this as new. Its aim is to
*connect* verified ML semantics to these downstream analyses and
transformations, in a single small, teachable compiler. It does not
aim to compete on performance with any of these systems.

## 9.4 What Tensora Does Not Claim to Invent

Tensors; shape types; symbolic dimensions; named axes; computation
graphs; static shape inference; operator fusion; MLIR or LLVM
lowering. The project does not claim any of these as novel.

## 9.5 The Academic Compiler's Contribution

Tensora's contribution is the **coherent path from verified ML
semantics to downstream compiler analysis and optimization**. The team
built this path end to end, and made it real on a small operator set:
an original declarative graph language with named axes; a semantic
analyzer (framed as an attribute grammar) that performs symbolic shape
reasoning, axis-name checking, and operator-contract checking, with
precise diagnostics; a typed **SSA** graph; a
tensor-**liveness/peak-memory analysis** that a transformation
consumes; team-authored graph passes, distinct from the
infrastructure's own; a team-driven **structured-loop tiling** decision
on matmul; native code generation through MLIR and LLVM; and a curated
corpus of real ML shape/graph failures. In one sentence: *Tensora is a
statically verified ML computation language whose typed SSA graph
keeps tensor semantics for graph-, memory-, and loop-level
optimization, before lowering through MLIR and LLVM.* Judge it on the
quality and completeness of that path, not on a novelty claim — tensor
DSLs and loop optimizers are well-established prior art (Section 9.3).

--------------------------------------------------------------------------

# 10. Applications and Non-Trivial Programs

## 10.1 Multilayer Perceptron (valid)

``` tensora
input  x:  Tensor[batch=B, feat=784]
weight W1: Tensor[in=784, out=256]
weight b1: Tensor[out=256]
weight W2: Tensor[in=256, out=10]
weight b2: Tensor[out=10]

graph mlp {
    h = linear(x, W1, b1)   # Tensor[batch=B, out=256]
    h = relu(h)             # Tensor[batch=B, out=256]
    y = linear(h, W2, b2)   # Tensor[batch=B, out=10]
    return y
}
```

## 10.2 Convolutional Classifier (valid)

The compiler infers and checks every axis below. With `padding=1,
kernel=3, stride=1`, each convolution keeps the spatial size the same;
each `maxpool(size=2)` halves it. After two conv/pool stages, the
`[batch=B, channel=64, height=56, width=56]` feature map flattens to
`[batch=B, flat=64*56*56] = [batch=B, flat=200704]`.

``` tensora
input  image: Tensor[batch=B, channel=3, height=224, width=224]
weight Wc1:   Tensor[out_channel=32, in_channel=3, kh=3, kw=3]
weight bc1:   Tensor[out_channel=32]
weight Wc2:   Tensor[out_channel=64, in_channel=32, kh=3, kw=3]
weight bc2:   Tensor[out_channel=64]
weight Wfc:   Tensor[in=200704, out=10]
weight bfc:   Tensor[out=10]

graph cnn {
    x = conv2d(image, Wc1, bc1, stride=1, padding=1)  # [batch=B, channel=32, height=224, width=224]
    x = relu(x)
    x = maxpool(x, size=2)                             # [batch=B, channel=32, height=112, width=112]
    x = conv2d(x, Wc2, bc2, stride=1, padding=1)       # [batch=B, channel=64, height=112, width=112]
    x = relu(x)
    x = maxpool(x, size=2)                             # [batch=B, channel=64, height=56, width=56]
    f = flatten(x)                                     # [batch=B, flat=200704]
    y = linear(f, Wfc, bfc)                            # [batch=B, out=10]
    return y
}
```

## 10.3 Transformer Feed-Forward Block (valid)

A position-wise feed-forward block is the largest compute in a
Transformer layer. Here it is expressed without attention. (v1 uses
`relu`; `gelu` is future scope, Section 12.)

``` tensora
input  x:  Tensor[batch=B, seq=S, model=768]
weight W1: Tensor[in=768, out=3072]
weight b1: Tensor[out=3072]
weight W2: Tensor[in=3072, out=768]
weight b2: Tensor[out=768]

graph transformer_ffn {
    h = linear(x, W1, b1)   # [batch=B, seq=S, out=3072]
    h = relu(h)             # [batch=B, seq=S, out=3072]
    y = linear(h, W2, b2)   # [batch=B, seq=S, out=768]
    return y
}
```

## 10.4 Multi-Head Scaled Dot-Product Attention (advanced, valid)

Q, K, and V come already projected and split into heads. Their layout
is `[batch=B, head=12, *seq, dim=64]`, with `12 * 64 = 768`
(consistent with the model width above). The query and key sequence
axes have separate names (`qseq`, `kseq`). This is more correct (they
may differ in cross-attention), and it is also what makes
`softmax(scores, axis=kseq)` unambiguous. This example uses only core
operators: batched `@`, a rank-4 `transpose`, a scalar multiply
(`1/sqrt(64) = 0.125`), and `softmax` over the key axis.

``` tensora
input Q: Tensor[batch=B, head=12, qseq=S, dim=64]
input K: Tensor[batch=B, head=12, kseq=S, dim=64]
input V: Tensor[batch=B, head=12, kseq=S, dim=64]

graph attention {
    kt      = transpose(K, perm=[0, 1, 3, 2])   # [batch=B, head=12, dim=64, kseq=S]
    scores  = Q @ kt                             # [batch=B, head=12, qseq=S, kseq=S]
    scores  = scores * 0.125                     # scale by 1/sqrt(64)
    weights = softmax(scores, axis=kseq)         # normalize over keys
    out     = weights @ V                        # [batch=B, head=12, qseq=S, dim=64]
    return out
}
```

Grouped-query and multi-query attention (fewer key/value heads than
query heads) need head-grouping contracts, and are a future extension.
A complete Transformer encoder layer (which also needs `layernorm`,
`gelu`, and an axis `rename`) is future scope (Section 12); v1 stops
at the attention block above as its Transformer showcase.

## 10.5 Invalid: Incompatible Matmul

``` tensora
input  x: Tensor[batch=B, feat=784]
weight W: Tensor[in=512, out=256]

graph bad_matmul {
    y = x @ W               # error: contracted sizes 784 != 512
    return y
}
```

## 10.6 Invalid: Reshape Changes Element Count

``` tensora
input x: Tensor[batch=B, feat=768]

graph bad_reshape {
    y = reshape(x, [batch=B, head=10, dim=64])   # error: 10*64 = 640 != 768
    return y
}
```

## 10.7 Invalid: Convolution Channel Mismatch

``` tensora
input  image: Tensor[batch=B, channel=3, height=224, width=224]
weight Wc:    Tensor[out_channel=32, in_channel=16, kh=3, kw=3]  # expects 16 input channels
weight bc:    Tensor[out_channel=32]

graph bad_conv {
    x = conv2d(image, Wc, bc, stride=1, padding=1)  # error: channel 3 != in_channel 16
    return x
}
```

## 10.8 Invalid: Axis-Name Mismatch (a bug only names catch)

``` tensora
input a: Tensor[batch=32, feat=64]
input b: Tensor[feat=64, batch=32]   # same sizes, transposed meaning

graph bad_axes {
    z = a + b            # error: axis 0 is 'batch' on the left but 'feat' on the right
    return z
}
```

--------------------------------------------------------------------------

# 11. The Tensora Language Environment

The compiler exposes each phase from the command line:

``` text
tensora --emit-tokens file.tns     # lexer output
tensora --emit-ast    file.tns     # parser output
tensora --check       file.tns     # semantic analysis (errors or OK)
tensora explain       file.tns     # inferred types and constraints
tensora --emit-graph  file.tns     # typed computation graph
tensora --emit-mlir   file.tns     # generated MLIR
tensora --emit-llvm   file.tns     # generated LLVM IR
tensora file.tns -o program        # native executable
```

These commands support both classroom demonstrations of the academic compiler and
day-to-day compiler debugging.

--------------------------------------------------------------------------

# 12. Core Language (v1) versus Future Extensions

**Core, frozen for v1.** f32 tensors with named axes whose sizes are
concrete or symbolic; `input`, `weight`, scalar literals; `graph`,
assignment with rebinding, a single `return`; the tiered operator set
of Section 2.5, with the contracts of Section 4.5; bounded symbolic
reasoning (equality/unification, affine-in-one-variable, concrete
integer arithmetic), together with axis-name checking; semantic
checking with diagnostics (framed as an attribute grammar); `explain`
mode; a typed **SSA** graph; the graph passes of Section 6.2; a
**tensor-liveness / peak-memory analysis** that a transformation
consumes; lowering to MLIR and a team-driven structured-loop
transformation (**tiling** on matmul, Section 8.4); LLVM/native CPU
execution; a real ML failure corpus for testing.

**Future extensions, explicitly not v1.** Training, autograd, optimizer
updates; general nonlinear or SMT-backed shape solving; grouped-query
and multi-query attention contracts; further layers and activations
(`gelu`, `layernorm`, an axis `rename`, a complete Transformer
encoder/decoder layer, `sigmoid`, `tanh`, batch normalization, dropout,
average pooling, dilated and grouped convolution); GPU and accelerator
lowering; a custom Tensora MLIR dialect; and dynamic control flow.

One future tier stands out as the most significant: the **integer-
tensor + `gather`
tier**. It is what separates v1 from a full language model. It adds an
integer element type (a dtype system), an integer-indexed `gather`
(token and positional **embedding lookup**), and `sin`/`cos` (for
rotary positional encodings). With this tier, the embedding front end
of an LLM (and recommender embedding tables) becomes expressible;
everything downstream of the embedding is already covered by v1. The
team plans this tier as open-source future scope. It is not part of
the graded v1 language. See the roadmap for details.

--------------------------------------------------------------------------

# 13. Conclusions

Tensora is a small declarative language for forward ML computation.
Its static semantics, named axes, and explicit graph structure feed
directly into team-authored optimization, and into MLIR/LLVM lowering.
As an academic compiler, the required result is a complete pipeline: from source,
through a symbolic shape-and-contract checker, an explain facility,
graph-level optimizations, and MLIR/LLVM, to native CPU execution — all
exercised by a corpus of valid and invalid ML programs. Tensora claims
no new ideas in tensor typing or ML compilation. Its value is a
coherent, complete, readable, and extensible compiler. A five-person
team can build, test, explain, and defend it. It leaves a clean path
toward later work on layout, fusion, memory, and accelerator lowering.
