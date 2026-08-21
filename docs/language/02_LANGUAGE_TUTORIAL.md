# 2. Tensora Language Tutorial

## 2.1 First Program

``` tensora
input  x: Tensor[batch=B, feat=784]
weight W: Tensor[in=784, out=256]

graph projection {
    y = x @ W
    return y
}
```

`B` is symbolic. The compiler infers this type:

``` text
y : Tensor[batch=B, out=256]
```

## 2.2 Understanding Matrix Multiplication

For this input:

``` text
Tensor[i=M, k=K] @ Tensor[k=K, j=N]
```

the result is:

``` text
Tensor[i=M, j=N]
```

The contraction works by position. The last axis of the left operand
and the second-to-last axis of the right operand must have equal (or
unifiable) **sizes**. Their names do not need to match, because the
contracted axis does not appear in the result. Leading axes, if any,
must match in name and in size. Section 2.8 gives the broadcasting
rules for this check.

This program is invalid:

``` text
Tensor[batch=B, feat=784] @ Tensor[in=512, out=256]
```

The compiler rejects it, because `784 != 512`.

## 2.3 Linear Layers

``` tensora
input  x: Tensor[batch=B, feat=784]
weight W: Tensor[in=784, out=256]
weight b: Tensor[out=256]

graph layer {
    h = linear(x, W, b)
    return h
}
```

The compiler infers `Tensor[batch=B, out=256]`. `linear` requires the
bias axis to match `W`'s output axis in name and size. The bias
argument is not optional in the core language.

## 2.4 Building an MLP

``` tensora
input  x:  Tensor[batch=B, feat=784]
weight W1: Tensor[in=784, out=256]
weight b1: Tensor[out=256]
weight W2: Tensor[in=256, out=10]
weight b2: Tensor[out=10]

graph mlp {
    h = linear(x, W1, b1)
    h = relu(h)
    y = linear(h, W2, b2)
    return y
}
```

Type flow:

``` text
Tensor[batch=B, feat=784]
   |
 Linear
   |
Tensor[batch=B, out=256]
   |
 ReLU
   |
Tensor[batch=B, out=256]
   |
 Linear
   |
Tensor[batch=B, out=10]
```

## 2.5 Explain Mode

``` text
tensora explain mlp.tns
```

This command shows the inferred type of every value. It also shows the
operator contract that produced each type (see Section 5 of the
whitepaper).

## 2.6 Reshape

``` tensora
input x: Tensor[batch=B, feat=768]

graph split {
    y = reshape(x, [batch=B, head=12, dim=64])
    return y
}
```

This program is valid, because `12 * 64 = 768`. `reshape` requires the
full list of new named axes and their sizes. The product of the target
sizes must equal the product of the source sizes.

``` tensora
reshape(x, [batch=B, head=10, dim=64])   # invalid: 10*64 = 640 != 768
```

## 2.7 Transpose

The compiler can permute `Tensor[batch=B, seq=S, head=H, dim=D]` into
`Tensor[batch=B, head=H, seq=S, dim=D]`:

``` tensora
y = transpose(x, perm=[0, 2, 1, 3])
```

`perm` must name each axis position exactly once. Under the
permutation, each size and each axis name moves together with its new
position.

## 2.8 Elementwise Operations and Broadcasting

Elementwise arithmetic (`+ - * /`) broadcasts by NumPy rules, plus one
more check: the axis name. Aligned axes must have equal, unifiable, or
size-1 dimensions. Their names must also match, unless the compiler is
broadcasting one axis from size 1. Two tensors can have the same
*sizes* but different *axis names* — the compiler treats them as
incompatible:

``` tensora
input a: Tensor[batch=32, feat=64]
input b: Tensor[feat=64, batch=32]

z = a + b   # error: axis 0 is 'batch' on the left but 'feat' on the right
```

This is the main reason Tensora's checker catches bugs that a
size-only shape checker would miss.

## 2.9 CNN Example

`conv2d` takes an explicit weight and bias, the same way `linear` does.
There is no implicit-weight form.

``` tensora
input  image: Tensor[batch=B, channel=3, height=224, width=224]
weight Wc:    Tensor[out_channel=32, in_channel=3, kh=3, kw=3]
weight bc:    Tensor[out_channel=32]

graph features {
    x = conv2d(image, Wc, bc, stride=1, padding=1)  # [batch=B, channel=32, height=224, width=224]
    x = relu(x)
    x = maxpool(x, size=2)                           # [batch=B, channel=32, height=112, width=112]
    return x
}
```

The compiler derives the output spatial dimensions from the
convolution and pooling formulas (see whitepaper Section 4.5).

## 2.10 Transformer Feed-Forward Example

``` tensora
input  x:  Tensor[batch=B, seq=S, model=768]
weight W1: Tensor[in=768, out=3072]
weight b1: Tensor[out=3072]
weight W2: Tensor[in=3072, out=768]
weight b2: Tensor[out=768]

graph transformer_mlp {
    h = linear(x, W1, b1)
    h = relu(h)
    y = linear(h, W2, b2)
    return y
}
```

v1 uses `relu`. Real Transformers use `gelu`. That activation, and
others such as `sigmoid` and `tanh`, are future scope. They are not
part of the frozen v1 set.

## 2.11 Multi-Head Attention Example

Attention is part of the frozen v1 language (see whitepaper Section
10.4). The compiler builds it entirely from core operators. Q, K, and V
already come split into heads. The query and key sequence axes have
separate names (`qseq`, `kseq`), so `softmax(..., axis=kseq)` has only
one meaning:

``` tensora
input Q: Tensor[batch=B, head=12, qseq=S, dim=64]
input K: Tensor[batch=B, head=12, kseq=S, dim=64]
input V: Tensor[batch=B, head=12, kseq=S, dim=64]

graph attention {
    kt      = transpose(K, perm=[0, 1, 3, 2])
    scores  = Q @ kt
    scores  = scores * 0.125
    weights = softmax(scores, axis=kseq)
    out     = weights @ V
    return out
}
```

## 2.11c Tracing Matmul Through the Whole Compiler

`C = A @ B` is the one computation worth understanding in full depth.
Tensora carries it through every level of the compiler. Tracing this
one computation is the clearest way to see how the phases connect:

``` text
source     C = A @ B
semantic   A: Tensor[i=M, k=K], B: Tensor[k=K, j=N]  =>  C: Tensor[i=M, j=N]
SSA graph  %C = matmul %A, %B
MLIR       linalg.matmul
loops      for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
tiled      for ii: for jj: for kk: for i,j,k in tiles: C[i,j] += ...
LLVM       LLVM IR  ->  native CPU
```

The tiling step (see whitepaper Section 8.4) is a team-authored
design choice: the compiler picks tile sizes from the verified shapes
`M`, `N`, and `K`. This one example touches semantic analysis, SSA,
lowering, and loop optimization. These are the core topics of the
course.

## 2.12 Compilation

Every flag below is real. You can run each one (`build/tensora`), and
each flag maps to one compiler phase:

``` text
tensora --emit-tokens          file.tns   # lexer: token stream
tensora --emit-parse-tree      file.tns   # parser: raw ANTLR parse tree
tensora --emit-ast             file.tns   # parser: the AST
tensora --emit-ast-dot         file.tns   # parser: the AST as Graphviz DOT
tensora --check                file.tns   # semantic analysis: OK or diagnostics
tensora --emit-symbols         file.tns   # semantic analysis: the symbol table (RM 3.6)
tensora --emit-graph           file.tns   # semantic analysis: the Typed Computation Graph
tensora --emit-graph-dot       file.tns   # semantic analysis: the graph as Graphviz DOT
tensora --explain              file.tns   # like --emit-graph, annotated per RM section
tensora --emit-optimized-graph file.tns   # optimization: the graph after the Level-1 passes
tensora --emit-mlir            file.tns   # lowering: MLIR (Level-2 tiling included)
tensora --emit-llvm            file.tns   # backend: translated LLVM IR
tensora --run                  file.tns   # full pipeline: JIT-execute
tensora file.tns -o file.o                # full pipeline: ahead-of-time native object file
```

Each flag maps to one compiler phase, and each phase has its own
output form. The lexer produces tokens. The parser produces the parse
tree and the AST. Semantic analysis produces the Typed Computation
Graph. Optimization produces the optimized graph. Lowering produces
MLIR. The backend produces LLVM IR. So the CLI is a map of the whole
pipeline, not just a debugging tool added on at the end.

The flags `--emit-tokens`, `--emit-parse-tree`, `--emit-ast`,
`--emit-ast-dot`, `--check`, `--emit-symbols`, `--emit-graph`,
`--emit-graph-dot`, `--explain`, and `--emit-optimized-graph` work on
any well-formed program, including one with symbolic dimensions.
`--emit-mlir`, `--emit-llvm`, `--run`, and `-o` need every input,
weight, and the graph's output to have a **concrete** (non-symbolic)
size. Lowering a symbolic dimension past type-checking is a known
limit (see `docs/testing/TEST_MATRIX.md` LOW-005). This limit is not a
silent bug: the driver rejects a symbolic shape with a clear error,
instead of compiling it wrong. `--run` uses the same synthetic value,
1, for every declared input and weight. Tensora source states shapes,
not values — the language has no tensor-literal syntax. This choice is
still easy to check by hand: for a matmul, every output element equals
the size of the contracted dimension.

The `examples/` directory has runnable, concrete-shaped versions of
the programs in this tutorial: `mlp_concrete.tns`, `matmul_large.tns`
(256x256x256, large enough to trigger the Level-2 tiling pass — check
with `--emit-mlir` for `scf.for` loops), `cnn.tns`, and `attention.tns`
(single-head). The compiler can lower batched matmul, so the rank-4,
multi-head version from Section 2.11 can also run; the project has
simply not added that example file yet.

## 2.13 Learning Progression

The tutorial builds up its examples in this order on purpose:

``` text
matmul (traced through the whole compiler, including tiling)
 -> linear
 -> MLP
 -> CNN
 -> Transformer feed-forward block
 -> multi-head attention (showcase)
```

Every stage uses only the frozen v1 operator set (see whitepaper
Section 2.5). Nothing in this progression needs a future extension. A
full Transformer encoder layer needs `layernorm`, `gelu`, and
`rename`. That layer is future scope.

## 2.14 Why Tensora: For First-Time ML Engineers

The most common first model anyone builds is an image or digit
classifier. Here it is in Tensora — the whole inference definition,
with no framework scaffolding around it:

``` tensora
input  x:  Tensor[batch=B, feat=784]     # a flattened 28x28 image
weight W1: Tensor[in=784, out=128]   weight b1: Tensor[out=128]
weight W2: Tensor[in=128, out=10]    weight b2: Tensor[out=10]

graph classifier {
    h = linear(x, W1, b1)
    h = relu(h)
    y = linear(h, W2, b2)
    return y                              # logits over 10 classes
}
```

Here are three concrete, honest benefits for someone learning ML:

1. **No framework boilerplate.** The code above is the entire
   inference program: no `class`, no `nn.Module`, no device or dtype
   management, no session or tracing. This is a fair comparison to a
   PyTorch script for *defining and checking an inference graph* —
   Tensora adds no ceremony around the math. It is not a fair
   comparison for the pure model *math*, which takes a similar number
   of lines in either system. Tensora also does not do training.

2. **The compiler catches shape and axis bugs at compile time, not at
   run time.** In PyTorch, a wrong dimension often runs for a while
   before it throws a deep, confusing runtime error. Sometimes it
   broadcasts silently into a subtle bug instead. Tensora rejects the
   error before anything runs, and it points at the exact line and
   values:

   ``` tensora
   weight W2: Tensor[in=256, out=10]    # oops: should be 128
   ...
   y = linear(h, W2, b2)
   # ShapeError: contracted sizes must be equal; found 128 != 256
   ```

   Tensora also names its axes. This lets it catch a class of bugs
   that a size-only checker, and PyTorch, both miss: combining a
   `[batch, feat]` tensor with a `[feat, batch]` tensor. Tensora
   rejects this combination even when the sizes line up (see Section
   2.8).

3. **You can see the model's shapes and the reasoning behind them.**
   `explain` prints the inferred type of every value, and the
   contract that produced it. A learner can watch shapes flow through
   the network:

   ``` text
   tensora explain classifier.tns
   ```

4. **The compiler applies optimizations you would otherwise hand-tune.**
   In PyTorch, getting fusion and redundant-op elimination means
   learning the compilation pipeline (`torch.compile`, fusion
   backends). In Tensora, these run as ordinary compiler passes on the
   graph. You do not configure anything extra.

Tensora is a learning and verification tool. It is not a replacement
for PyTorch: it targets forward inference only, so training stays in
the frameworks. Its value is this: the model is a small, declarative,
statically checked, and explainable program.
