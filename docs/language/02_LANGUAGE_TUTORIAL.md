# 2. Tensora Language Tutorial

## 2.0 Your First 5 Minutes

Tensora is a small declarative language for describing forward ML
computations using statically checked tensors with named axes. Here is
a complete Tensora program:

``` tensora
input  x: Tensor[batch=B, feat=784]
weight W: Tensor[in=784, out=256]

graph projection {
    y = x @ W
    return y
}
```

Save it as `projection.tns` and run it through the compiler's type
checker, no execution needed:

``` bash
tensora --explain projection.tns
```

which prints every value's inferred type:

``` text
y : Tensor[batch=B, out=256]
```

`B` is symbolic — a batch size the program never fixes. The compiler
still proves `x`'s feature axis (`784`) matches `W`'s input axis
(`784`) and infers the rest. Now break it: change `weight W` above to
`Tensor[in=512, out=256]`, save it, and run `tensora --check`. Section
2.2 walks through exactly what error you should see, and why. That
predict → break → fix loop, repeated on progressively bigger programs,
is the rest of this tutorial.

**Where this is going.** The same four constructs above
(`input`/`weight`/`graph`/`@`) are enough to eventually check and run a
real self-attention block taken from a genuine pretrained ViT-Base
Transformer and verify Tensora's output against PyTorch's to
`1.86e-07`. See the [README's real-world
demos](../../README.md#real-world-inference-demos) for that payoff
(one-time setup: `make setup-tensora-demos` — nothing below needs it yet), and
Section 2.14 for the full, honest case for why Tensora exists
alongside PyTorch rather than instead of it.

Section 2.12 covers the full CLI (`--check`, `--explain`, `--run`,
`--emit-graph`/`--emit-optimized-graph`, `--emit-mlir`) and every
runnable example in this tutorial's build-up order. Section 2.13 forks
into an ML-focused path and a compiler-internals path once the basics
below are done.

## 2.1 First Program, in Detail

Section 2.0's `projection` program above is this tutorial's running
example. `input` and `weight` both declare tensors with named,
sized axes; the only difference is that `--run` expects `weight`
values to come from trained data (Section 2.12 Step 4), while `input`
values are supplied per call. `graph` is the unit the compiler
type-checks and compiles. `y = x @ W` is matrix multiplication; Section
2.2 covers exactly which axes it contracts and why the result's type
comes out `Tensor[batch=B, out=256]`.

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

> **Try it — break it (1 minute).** Copy Section 2.0's program to
> `bad_matmul.tns` and change `weight W` to
> `Tensor[in=512, out=256]`. Run `tensora --check bad_matmul.tns` and
> read the diagnostic; it should point at exactly the mismatched sizes
> above. Then fix it back to `784` and re-run to see it pass. This
> predict-break-fix loop is worth repeating for any change you're
> unsure about, throughout the rest of this tutorial.

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

Chain two linear layers with a `relu` between them and you have a full
multilayer perceptron. The runnable version is
[`examples/mlp.tns`](../../examples/mlp.tns) — open it rather than
retyping it here; it is the same 784 → 256 → 10 shape as Section 2.3,
just with a second layer appended. Try:

``` text
tensora --explain examples/mlp.tns
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
tensora --explain examples/mlp.tns
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

Both cases are exercised, accept and reject, in the test suite:
[`SEM-RS-001`](../testing/TEST_MATRIX.md#semantic--reshape--flatten-rm-312-313)
and [`SEM-RS-002`](../testing/TEST_MATRIX.md#semantic--reshape--flatten-rm-312-313).

## 2.7 Transpose

The compiler can permute `Tensor[batch=B, seq=S, head=H, dim=D]` into
`Tensor[batch=B, head=H, seq=S, dim=D]`:

``` tensora
y = transpose(x, perm=[0, 2, 1, 3])
```

`perm` must name each axis position exactly once. Under the
permutation, each size and each axis name moves together with its new
position — see
[`SEM-TR-001`](../testing/TEST_MATRIX.md#semantic--transpose--reductions--softmax-rm-314-311-315)
for this exact rank-4 case, and the two rows after it for the reject
cases (repeated or out-of-range `perm` index).

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
size-only shape checker would miss. **Further reading:** `SEM-EW-003`
in the [test matrix](../testing/TEST_MATRIX.md#semantic--elementwise--broadcasting--axis-names-rm-38),
whitepaper [Section 10.8](00_WHITEPAPER.md#108-invalid-axis-name-mismatch-a-bug-only-names-catch)
for this exact failure mode in more depth.

> **Try it — break it (1 minute).** Save the invalid `a + b` program
> above to `bad_broadcast.tns` and run `tensora --check
> bad_broadcast.tns`. Predict the error first: will Tensora accept it
> because the *sizes* (`32`, `64`) line up, or reject it because the
> *names* don't? Then rename `b`'s axes to `Tensor[batch=32, feat=64]`
> (matching `a`) and re-run — now it passes. A size-only checker (and
> plain PyTorch) would have accepted the original, buggy version.

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
convolution and pooling formulas (see whitepaper Section 4.5). That
covers one conv/pool stage; for the full two-stage classifier with a
`flatten` and a final `linear` head, run
[`examples/cnn.tns`](../../examples/cnn.tns) — its header comment gives
a hand-computed reference value that
[`tests/codegen/test_cnn_e2e.cpp`](../../tests/codegen/test_cnn_e2e.cpp)
checks the compiler against.

## 2.10 Transformer Feed-Forward Example

A Transformer's feed-forward block is exactly the MLP shape from
Section 2.4 at Transformer width: two `linear` layers with a `relu`
between them, applied per sequence position. Run
[`examples/transformer_ffn.tns`](../../examples/transformer_ffn.tns)
directly — it is the whitepaper's own
[Section 10.3](00_WHITEPAPER.md#103-transformer-feed-forward-block-valid)
example, so there is no separate copy to keep in sync here. v1 uses
`relu`; real Transformers use `gelu`. That activation, and others such
as `sigmoid` and `tanh`, are future scope (not part of the frozen v1
set).

## 2.11 Multi-Head Attention Example

Attention is part of the frozen v1 language. This is the same example
as the whitepaper's Transformer showcase
([Section 10.4](00_WHITEPAPER.md#104-multi-head-scaled-dot-product-attention-advanced-valid)),
repeated here in the tutorial's build-up order rather than the
whitepaper's fully-annotated form. The compiler builds it entirely from
core operators. Q, K, and V already come split into heads. The query
and key sequence axes have separate names (`qseq`, `kseq`), so
`softmax(..., axis=kseq)` has only one meaning:

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

This exact rank-4 program type-checks end to end
([`ATT-001`](../testing/TEST_MATRIX.md#semantic--attention--multi-head-whitepaper-104)),
and it now runs, too — three runnable siblings exist:
[`examples/attention.tns`](../../examples/attention.tns) (rank-2,
single-head, matches the numerical reference in
[`TEST_MATRIX.md` row E2E-004](../testing/TEST_MATRIX.md#end-to-end-rm-322-whitepaper-10)),
[`examples/attn_text_classifier.tns`](../../examples/attn_text_classifier.tns)
(a real PyTorch-trained, batched, single-head self-attention
classifier — run it with `make demo-attn-text-classifier`), and
[`examples/attn_multihead_retrieval.tns`](../../examples/attn_multihead_retrieval.tns)
(rank-4, 4-head, real PyTorch-trained weights, but a deliberately
different task from the single-head classifier: Associative Recall
(AR), a standard synthetic benchmark for probing self-attention's
retrieval ability — given 6 shuffled `key:value` slots and a query
naming one key, retrieve the value bound to it *in that specific
sequence* — run it with `make demo-attn-retrieval`; see the example's
own header for the research citation). The multi-head
example's header comment works through how it gets a `head` axis onto
Q/K/V and back off the output projection using only batched
matmul/transpose/sum — no reshape — since general reshape is not
lowered (Section 1.7 of
[`docs/ProjectMgmt/01_INTRODUCTION.md`](../ProjectMgmt/01_INTRODUCTION.md)),
and how it reads out "the hidden state at the query position" (no
slicing op either) via a one-hot selector weight and `sum`.

## 2.11c Tracing Matmul Through the Whole Compiler

`C = A @ B` is the one computation worth understanding in full depth:
Tensora carries it through every level of the compiler, from the typed
SSA graph down through MLIR, structured loops, tiling, and LLVM. This
is the whitepaper's own trace
([Section 8.3](00_WHITEPAPER.md#83-matrix-multiplication-the-reference-computation)),
so there is no separate copy to keep in sync here — read it there for
the full pipeline diagram and the tiling design rationale (whitepaper
Section 8.4).

## 2.12 Running Programs Yourself

Every example above is a real, checkable `.tns` file, not a snippet
invented for this tutorial. The fastest way to learn the rest of the
language is to run them yourself, in the order below — simplest first,
ending with the full, real-data programs.

**Step 1 — check and explain a program.** These two flags accept any
well-formed program, symbolic dimensions included, and they are where
you start when learning what a program means:

``` bash
tensora --check   examples/mlp.tns   # OK, or a diagnostic pointing at the exact line/values
tensora --explain examples/mlp.tns   # every value's inferred type, and which RM rule produced it
```

**Step 2 — run a program.** `--run` and `-o` need every input, weight,
and output size to be **concrete** (not symbolic) — a known limit
([`TEST_MATRIX.md` LOW-005](../testing/TEST_MATRIX.md#lowering-rm-322)),
reported with a clear compiler error rather than a miscompile. That is
why the table below has both a symbolic-shaped teaching version of a
program (`examples/mlp.tns`) and a concrete-shaped runnable twin
(`examples/mlp_concrete.tns`) where the distinction matters:

``` bash
tensora --run examples/mlp_concrete.tns   # JIT-executes with synthetic all-ones inputs
```

All-ones inputs are still easy to check by hand: for a matmul, every
output element equals the size of the contracted dimension (see the
comment at the top of each `examples/*.tns` file for its own worked
number).

**Step 3 — run every example, in order.** This table is the whole
`examples/` catalog, in the same build-up order as this tutorial:

| File | Demonstrates | Try it |
|---|---|---|
| [`mlp.tns`](../../examples/mlp.tns) | `linear` + `relu`, symbolic batch (Section 2.4) | `tensora --explain examples/mlp.tns` |
| [`mlp_concrete.tns`](../../examples/mlp_concrete.tns) | the same MLP, concrete shapes | `tensora --run examples/mlp_concrete.tns` |
| [`matmul_large.tns`](../../examples/matmul_large.tns) | a 256x256x256 matmul, large enough to trigger Level-2 tiling | `tensora --emit-mlir examples/matmul_large.tns` (look for `scf.for`) |
| [`cnn.tns`](../../examples/cnn.tns) | `conv2d` -> `relu` -> `flatten` -> `linear` -> `softmax` (Section 2.9) | `tensora --run examples/cnn.tns` |
| [`cnn_grid_detector.tns`](../../examples/cnn_grid_detector.tns) | fixed-grid object detection: per-cell `conv2d`/`relu`/`maxpool` backbone + 1x1 `conv2d` classification head, real PyTorch-trained weights, draws boxes | `make demo-cnn-grid-detector` |
| [`cnn_grid_detector_real.tns`](../../examples/cnn_grid_detector_real.tns) | same task, real Pascal-VOC-2012 photographs at finer (16x16) grid resolution instead of drawn shapes | `make demo-cnn-grid-detector-real` |
| [`transformer_ffn.tns`](../../examples/transformer_ffn.tns) / [`transformer_ffn_concrete.tns`](../../examples/transformer_ffn_concrete.tns) | a Transformer feed-forward block at real width (Section 2.10) | `tensora --run examples/transformer_ffn_concrete.tns` |
| [`attention.tns`](../../examples/attention.tns) | single-head scaled dot-product attention, rank-2 (Section 2.11) | `tensora --run examples/attention.tns` |
| [`attn_text_classifier.tns`](../../examples/attn_text_classifier.tns) | single-head attention, real PyTorch-trained weights | `make demo-attn-text-classifier` |
| [`attn_multihead_retrieval.tns`](../../examples/attn_multihead_retrieval.tns) | 4-head attention, rank-4, real PyTorch-trained weights, Associative Recall (Section 2.11) | `make demo-attn-retrieval` |
| [`vit_attention_block.tns`](../../examples/vit_attention_block.tns) | one self-attention block, 12-head/768-dim, real weights from a genuine pretrained ViT-Base checkpoint (not just a small model trained for the demo); reused, at every encoder layer and inside a live model, by the two demos to its right | `make demo-vit-attention-one-layer` / `make demo-vit-attention-all-layers` / `make demo-vit-attention-substitution` |
| [`vit_attention_weights.tns`](../../examples/vit_attention_weights.tns) | the same block, stopped right after `softmax` — returns the attention matrix itself, for a per-head attention-heatmap visualization over a real photograph | `make demo-vit-attention-viz` |
| [`optimizations/*.tns`](../../examples/optimizations/) | one file per Level-1 pass (positive case) — `dead_code.tns`, `shape_fold.tns`, `reshape_cancel.tns`, `transpose_cancel.tns`, `elementwise_fusion.tns`, `linear_bias_activation.tns` | `tensora --emit-graph examples/optimizations/dead_code.tns` vs `--emit-optimized-graph` |

**Step 4 — real trained weights, not synthetic ones.** `--run`'s
default all-ones input is for hand-checking the compiler, not for
seeing a trained model work. To run a program on genuine data, pass
`--data-dir DIR`: the compiler loads one float32 `.npy` file per
declared input and weight (`DIR/<name>.npy`), checks its shape and
dtype against what the program declares, and rejects a mismatch with a
clear diagnostic instead of silently misreading memory. `--output
OUT.npy` writes the result tensor the same way. Every attention/MLP/CNN
example in the table above that says "real PyTorch-trained weights"
works exactly this way — a `tools/models/prepare_*.py` script trains a
small model once in PyTorch, exports its weights and a batch of
held-out inputs as `.npy` files under `demos/artifacts/`, and the demo
script (e.g. `demos/attn_retrieval_demo.sh`) runs `tensora --run ...
--data-dir ...` and compares every output row against the saved
PyTorch reference. See
[`docs/development/BUILD.md`](../development/BUILD.md) for the full
`--data-dir`/`--output` reference, and the [README's real-world
demos](../../README.md#real-world-inference-demos) section for how to
prepare each one. `make setup-tensora-demos` prepares every one of them in one
command (Section 2.0).

**Give it knobs.** A static demo is interesting once; one you can
change is worth returning to. `prepare_vit_attention_demo.py` accepts
a different photo and a different encoder layer without editing the
script — `make prep-vit-attention IMAGE=my-cat.jpg LAYER=8` (`IMAGE`
is a local file path or URL; `LAYER` is `0`-`11`) — then re-run `make
demo-vit-attention-viz` and compare the new figures against the
original dog/layer-11 ones. The attention-microscope figure similarly
takes a `"row,col"` argument for one specific patch instead of its
default face/body/background trio (README's ViT demos section).
Predicting what changes before running it, then checking, is the same
loop as the "break it" exercises above, applied to a real model
instead of a hand-written program.

**Step 5 — the whole compiler, phase by phase.** If you also want to
see *how* the compiler gets from source to native code — not just that
it does — the six `demos/0N_*.sh` scripts (`make example`) walk the
same pipeline one flag per compiler phase (tokens, parse tree/AST,
type-checking, optimization, MLIR/LLVM, and the full JIT/ahead-of-time
run). That level of detail is the [README's phase-demo
table](../../README.md#tensora-phase-demos), not repeated here.

## 2.13 Learning Progression

The tutorial builds up its examples in this order on purpose:

``` text
matmul (traced through the whole compiler, including tiling)
 -> linear
 -> MLP
 -> CNN
 -> Transformer feed-forward block
 -> single-head attention
 -> multi-head attention (real trained weights)
```

Every stage uses only the frozen v1 operator set (further reading:
whitepaper Section 2.5). Nothing in this progression needs a future
extension. A full Transformer encoder layer needs `layernorm`, `gelu`,
and `rename`. That layer is future scope.

**Pick a path.** From here, the material forks into two independent
directions — follow either, or both, in any order:

``` text
                  Section 2.1-2.12 (basics)
                          |
              +-----------+-----------+
              |                       |
        ML EXPLORER            COMPILER EXPLORER
              |                       |
   real CNN (2.9, 2.14)      matmul trace (2.11c)
              |                       |
   real attention (2.11)     --emit-mlir / tiling (2.12 Step 3)
              |                       |
   real pretrained ViT       optimization passes (2.12 Step 3, 2.13's own
   (README's ViT demos)      note above)
              |                       |
   attention microscope /    phase-by-phase demos
   asymmetry / diagnostics   (2.12 Step 5)
   (README's ViT demos)
```

The ML-explorer path answers "what does this computation do and why
does it work?"; the compiler-explorer path answers "how does Tensora
get from this source text to native code?". Section 2.14 (next) is the
start of the ML-explorer path's case for why the language exists at
all.

## 2.14 Why Tensora: For First-Time ML Engineers

Tensora is a learning and verification tool. It is not a replacement
for PyTorch: it targets forward inference only, so training stays in
the frameworks you already use. What it offers instead is a model
that is small, declarative, statically checked, and explainable — a
place to see exactly what an inference graph is doing and why the
compiler accepted or rejected it.

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

This snippet is illustrative (no MNIST weights attached to it), but the
same idea runs for real: `make demo-cnn-image-classifier` classifies
genuine STL-10 images (10 real classes, not illustrative ones) through
[`examples/cnn_stl10.tns`](../../examples/cnn_stl10.tns) with
PyTorch-trained weights, and checks the result against PyTorch — see
Section 2.12's Step 4 for how `--data-dir` gets those trained weights
into the program above's `linear`-based cousin. A second CNN demo goes
one step further than whole-image classification: `make
demo-cnn-grid-detector` predicts a class *per cell* of a fixed 4x4 grid
over a 32x32 image containing 1-3 shapes
([`examples/cnn_grid_detector.tns`](../../examples/cnn_grid_detector.tns)),
then draws a box at each detected cell — "what AND where", not just
"what", using the same `conv2d`/`relu`/`maxpool`/`softmax` operators
already introduced above (a 1x1 `conv2d` as a per-cell linear head, no
new operators needed).

Here are four concrete, honest benefits for someone learning ML:

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
   `--explain` prints the inferred type of every value, and the
   contract that produced it. A learner can watch shapes flow through
   the network:

   ``` text
   tensora --explain classifier.tns
   ```

4. **The optimizer's decisions are visible, not a black box.**
   PyTorch itself also fuses ops and tiles matmuls (`torch.compile`,
   Inductor) — that is not something Tensora does that PyTorch
   cannot. The difference is inspectability: `--emit-graph` and
   `--emit-optimized-graph` show the exact graph before and after each
   Level-1 pass (dead-code elimination, reshape/transpose
   cancellation, elementwise fusion), and `--emit-mlir` shows whether
   a matmul was tiled and with what tile sizes. A learner can see
   which rewrite fired and why, rather than reading a JIT trace or
   inductor log.

Tensora's value for a first-time ML engineer is not that it is faster
or more capable than PyTorch — it is not, and does not try to be. It
is that the whole inference graph, and every check and rewrite the
compiler performs on it, fits in view at once.

For a model of this same shape with genuine PyTorch-trained weights
instead of illustrative ones, run
[`examples/mlp_placement.tns`](../../examples/mlp_placement.tns) via
`make demo-mlp-placement` — it is where this tutorial's ideas meet real
data.
