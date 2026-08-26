# Future Language Constructs and PyTorch Interop: full detail

> This is the detailed design analysis behind §4/§8/§9 of
> `docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md` on `main`.
> That document keeps only a short, high-level pointer to each
> direction; this file is where the actual design debate, the option
> tables, and the sketches live. It stays on `v0.3-research` because it
> is detail, not a broad direction — `main`'s roadmap is meant to be
> readable without it.
>
> Written before the V0.2 spikes (§2 of the roadmap on `main`). Where a
> section below assumes a "ragged axis kind" win is the flagship
> research question (most references to "§4.F"), read that as
> superseded: the V0.2 spikes found ordinary ragged attention is
> already covered by `torch.nested`, and the real, narrower research
> question is verified relationships among per-instance metadata
> *contents* -- see `research/ragged_spike/PROPOSAL.md` in this
> directory tree for the corrected framing. The engineering-design
> content below (operator gaps, the PyTorch interop options, the ABI
> design) is otherwise still the current thinking.

## 8. Future Language Constructs (post-v0.1)

> Everything in this section is out of v0.1 scope. It does not change
> the frozen whitepaper, reference manual, architecture design, or test
> plan. All of those are v0.1-only. Every construct below is
> inference-only (forward computation, no autograd, no training). This
> matches the v0.1 boundary. Each construct can become open-source
> engineering work, or a measured research contribution, once v0.1
> exists. This section does not choose between those two paths.

### 8.0 Prior art and the gap (honest positioning)

These systems are close neighbors. Cite them. Do not reinvent them.

| System | Kind | Static verification | Named axes | Semantics in IR |
|---|---|---|---|---|
| Tensor Comprehensions (arXiv:1802.04730) | source DSL, Einstein-style index notation | none | no | n/a (polyhedral JIT -> CUDA) |
| MLIR Toy tutorial | teaching DSL (rank <= 2, f64) | shape inference only | no | toy dialect |
| MLIR `tensor` dialect | IR (not a language) | structural/shape only | explicitly none | it is the IR |
| TensorFlow MLIR | production bridge | shape inference | no (some attrs) | TF/HLO dialects |
| **Tensora v0.1** | declarative graph language | strong named-axis contracts | yes | TCG, then erased at MLIR |

None of these systems fill one gap: a source tensor language whose
named-axis contracts are both statically verified and kept in the IR
and the optimizer. Tensor Comprehensions has expressive index notation,
but no verification. Toy and the `tensor` dialect have no names and no
contracts. TensorFlow lowers semantics away into HLO. The constructs
below target that one gap. Tensora does not claim to invent index
notation, polyhedral compilation, autotuning, custom dialects, or
fusion. It claims only to combine named-axis verification with keeping
those names through the pipeline.

### 8.1 Verified index-notation operators (spine construct)

Bring in Tensor-Comprehensions-style index notation. Tie every index to
one declared, verified named axis. Sketch:

``` tensora
op matmul(a: Tensor[batch=B, i=M, k=K],
          b: Tensor[batch=B, k=K, j=N]) -> Tensor[batch=B, i=M, j=N] {
    out(batch, i, j) = sum(k) a(batch, i, k) * b(batch, k, j)
}
```

Static rules make this different from Tensor Comprehensions: every
index must match a declared axis name. Each index resolves to exactly
one axis size, using the v0.1 bounded solver. Free indices (on the left
side) must equal the output axes. Summation indices must be bound on
the right side, and must not appear in the output. This stays
forward/inference-only: `sum(k) ...` is a contraction, not a gradient.
It lowers to `linalg.generic`, using the derived iteration space.

- Leaning: research. Verification-over-notation is the new part here.
  It also has clear open-source value.
- Depends on: nothing beyond v0.1. It is the foundation for 8.2 and 8.4.

### 8.2 User-defined operators

Let users define new named operators. Build them from 8.1's index
notation, plus an explicit contract (preconditions over named axes).
Check the contract the same way as a built-in operator. Lower it
through the same path:

``` tensora
op split_heads(x: Tensor[batch=B, seq=S, model=D])
    -> Tensor[batch=B, head=H, seq=S, dim=Dh]
    where { D == H * Dh; } {
    out(batch, h, seq, dh) = x(batch, seq, h * Dh + dh)
}
```

Tensor Comprehensions allows custom operators, but checks nothing.
Tensora's version type-checks the contract and the index algebra. This
is how the frozen built-in operator set becomes extensible, without
touching the compiler itself.

- Leaning: open-source (it adds extensibility). It also has research
  value (contract inference).
- Depends on: 8.1.

### 8.3 A `tensora` MLIR dialect

Add a custom dialect. It carries named axes and proven contract facts
as attributes on high-level ops. IR-level passes can then read these
facts before the ops lower to `linalg`/`tensor`. The `tensor` dialect
cannot carry this information. That is why v0.1 erases names at the
MLIR boundary (ARCH-005). This construct is what turns the claim
"semantics survive compilation" into something you can test, not just
something you say.

- Leaning: research. It is the base for the paper-shaped question in
  Section 4.A.
- Depends on: v0.1 lowering. It enables 8.4, 8.5, and 8.6.
- Cost note: this is real engineering work (TableGen ops and types,
  verifiers, a conversion pass). Only build it once a later pass
  actually needs these preserved facts.

### 8.4 Contract-driven specialization and autotuning

Use proven contract facts — divisibility, head counts, concrete sizes,
`hidden == heads * head_dim` — as priors. Use them to generate
specialized code variants, and to prune an autotuning search. Tensor
Comprehensions autotunes without such priors. Tensora's verified facts
would narrow the search space before the search even starts.

- Leaning: research. This is measurable: does using contract priors
  cut autotuning cost, or improve the chosen schedule?
- Depends on: 8.3. The facts must survive into the IR before a pass can
  use them.

### 8.5 Layout / axis-semantics optimization

Use what an axis *means*, not just its size, to cut layout
conversions. Remove redundant transposes. Choose a physical layout per
axis role. Recognize when a `transpose` is a no-op for its consumer.
This extends the v0.1 transpose-cancellation pass with axis-aware
reasoning.

- Leaning: research (paper-shaped question 4.C), plus open-source
  performance value.
- Depends on: 8.3. The names must be available in the IR.

### 8.6 Polyhedral loop optimization

Apply Tensor-Comprehensions-style polyhedral scheduling — tiling,
fusion, loop transformation — to the verified, named-axis IR. This is
the heaviest, most research-intensive construct here. It would likely
use MLIR's own affine/polyhedral tools, rather than a new scheduler
built from scratch.

One open problem with full polyhedral scheduling is cost: computing the
schedule can be slow, because it solves over general polyhedra.
**Sub-polyhedral scheduling** is a known way to make this step fast, by
restricting the constraints to a simpler form (two variables per
inequality, UTVPI) that has near-linear-time solvers, at the cost of
some expressive power. See Upadrasta and Cohen, "Sub-polyhedral
scheduling using (unit-)two-variable-per-inequality polyhedra" (POPL
2013), and their earlier "Potential and Challenges of
Two-Variable-Per-Inequality Sub-Polyhedral Compilation" (IMPACT 2011).
If this construct is ever built, this is the technique to evaluate for
keeping the scheduling step tractable.

- Leaning: research, long horizon.
- Depends on: 8.1 (index notation gives clean iteration spaces) and
  8.3.

### 8.7 Verifier and real-world ML failure corpus as a product

Package the semantic verifier together with a curated corpus of real
ML tensor bugs, taken from cited public issues and model code. Offer
this as standalone open-source value. None of the neighboring systems
in §8.0 verify named-axis contracts. So a tool that says "this catches
N real shape/axis/contract bugs before you run" stands on its own.

- Leaning: open-source, for adoption. It also supports paper-shaped
  question 4.B.
- Depends on: v0.1 only. This is the lowest-risk construct in this
  list. It is a good first deliverable after v0.1.

### 8.8 Suggested ordering

``` text
v0.1 (frozen)
  -> 8.7 verifier + failure corpus        (open-source, low risk, no new language)
  -> 8.1 verified index notation          (foundation)
  -> 8.2 user-defined operators           (extensibility, builds on 8.1)
  -> 8.3 tensora dialect                 (research substrate)
  -> 8.4 / 8.5 contract-driven passes     (measured research on top of 8.3)
  -> 8.6 polyhedral                        (long-horizon research)
  -> 8.11 ragged/variable-length axes     (separate research thread, parallel
                                            to 8.4-8.6; see §4.F, §9.6)
```

Pick the branch — open-source or research — after v0.1 exists, and
after the faculty conversation in Section 4. This section supports
either path.

### 8.9a Transformer-layer completeness tier — open-source future

V0.1 was rescoped to a compiler MVP. It favors depth over breadth: a
small operator set, an SSA graph, a tensor-liveness analysis, a
structured-loop tiling transformation, and an ANTLR frontend. As part
of that rescope, three constructs moved out of v0.1 into this future
tier. A complete Transformer encoder or decoder layer needs all three:

- **`gelu`** — the feed-forward activation real Transformers use.
- **`layernorm`** — normalization over a named axis. As one operator,
  the surface language needs neither `sqrt` nor keepdim reductions.
- **`rename`** — a type-level axis relabel. This is needed for two
  reasons: self-attention from one input produces two axes that both
  come from `seq`, and a `linear` output (auto-named `out`) must be
  relabeled to match a residual add.

With these three operators, a full Transformer encoder or decoder layer
becomes expressible: layernorm, then multi-head self-attention, then a
residual add, then layernorm again, then a `gelu` feed-forward block,
then a second residual add. No type-system change is needed. These
three are the natural first open-source additions after v0.1, ahead of
the heavier integer-tensor tier below.

V0.1's Transformer showcase is `examples/attn_text_classifier.tns`. It
is a real self-attention classifier, with PyTorch-trained weights,
running end to end through Tensora. See
`tools/models/prepare_text_attn_demo.py`. A full pretrained
BERT/DistilBERT extractive-QA model needs this tier plus §8.9: `gelu`,
`layernorm`, and `rename` for the encoder layers, and the integer
element type plus `gather` (below) for the token-embedding input path.
Until both are built, no pretrained QA model can be reproduced
faithfully — every real one uses LayerNorm, GELU, and integer-indexed
embeddings.

### 8.9 Integer-tensor + gather tier (the LLM input path) — open-source future

> Note: this tier is what remains between v0.1 (plus §8.9a) and a full
> language model. It stays future scope because it reopens the type
> system.

V0.1 verifies any feed-forward network, and any Transformer layer or
body. It does not verify a full LLM. The reason: token and positional
embedding lookup needs integer-indexed tensors, and v0.1's type system
is f32-only. This tier adds, as one unit:

- **An integer element type** — a minimal dtype system: f32, plus
  `i32`/`i64` index tensors. This is the load-bearing change here.
  Everything else in this tier depends on it.
- **`gather(table, indices)`** — embedding lookup. Given
  `table: Tensor[vocab=V, model=D]` and integer
  `indices: Tensor[batch=B, seq=S]`, it produces
  `Tensor[batch=B, seq=S, model=D]`. This covers token embeddings,
  learned positional embeddings, and recommender embedding tables.
- **`sin` / `cos`** — elementwise operators. These enable rotary
  positional encodings (RoPE).
- **`avgpool` / global average pool** — completes ResNet-class CNNs.
  (Inference-time BatchNorm already folds into `mul` and `add`.)

With this tier, the LLM front end — embedding, then transformer body,
then LM head — becomes expressible end to end. So do recommender
models. Everything downstream of the embedding step is already covered
by v0.1 plus `layernorm`, `gelu`, and `rename`.

Why this stays future work, not v0.1: adding an integer element type
reopens the type system. It adds mixed int/float operator contracts. It
makes index-range a semantic concern. It adds dtype to the surface
syntax and to lowering. This is a bigger, higher-risk change than the
three v0.1 additions in §8.9a. It is also not needed for v0.1's own MVP
claim: "verifies any feed-forward net and any Transformer layer." It is
the natural first open-source milestone after v0.1.

- Leaning: open-source, for adoption — "verify and run real
  embedding-based models." It also strengthens paper-shaped question
  4.B, for failure detection on real LLM and recommender code.
- Depends on: v0.1, plus the integer element type this tier itself
  introduces. It does not depend on 8.1–8.6.

### 8.10 Documentation versioning (V0.1 vs open-source)

The language reference and tutorial run on two separate tracks. This
keeps the V0.1 scope from blurring into future scope:

- **V0.1, normative and graded.**
  `docs/language/03_LANGUAGE_REFERENCE_MANUAL.md` and
  `docs/language/02_LANGUAGE_TUTORIAL.md` contain only V0.1 scope. They
  are the artifacts defended in the viva. They stay small, and cover
  only V0.1 scope. Nothing from Section 8 appears in them.
- **Beyond V0.1 (open source), non-normative until built.** Every
  future construct lives only here, in Section 8, as a sketch. It is
  not added to the V0.1 reference or tutorial. When the open-source
  version actually implements a construct (v0.2+), that construct
  moves out of Section 8 and into a separate, version-tagged manual and
  tutorial — for example `docs/opensource/REFERENCE.md` and
  `TUTORIAL.md`, created at that time. Those may start as a copy of the
  V0.1 docs, then extend from there.

Rule of thumb: a reference manual documents what is built and frozen.
Documenting unbuilt features in the normative manual would mislead a
reader, and a grader. So the V0.1 manual and tutorial carry no
beyond-V0.1 material. The open-source manual is a future artifact. This
roadmap seeds it, and implementation promotes it.

### 8.11 Ragged / variable-length axes

This is the language-construct side of the paper-shaped question in
§4.F. Add a ragged axis kind. This is different from the
constant/symbolic/expression `Dimension` kinds RM 3.x already defines.
A ragged axis carries one length per batch element — sketch syntax,
not frozen, for example `seq: Ragged[batch=B]` — instead of one shared
size for the whole batch. Static verification would check every use of
a ragged axis one of two ways: either it reduces over that axis's true
per-element length, not a padded one, or it is an explicit, checked
pad or mask operation. This turns "did you accidentally compute over
padding" into a static, checkable fact, instead of a silent
correctness or efficiency bug.

This is deliberately kept separate from §8.9's integer-tensor tier,
even though both "reopen the type system." §8.9 adds a new element
dtype (`i32`/`i64`) on top of an unchanged shape model. This construct
adds a new *kind* of shape. Most of v0.1's semantic layer and codegen
assumes a `Dimension` is one number per axis. Every pass, contract, and
lowering rule built on that assumption needs an explicit decision about
what a ragged axis means to it. This is the largest single type-system
change anywhere in this roadmap — larger than §8.9. Do not start it
without the feasibility spike §4.F calls for. In particular, check
what MLIR lowering strategy is even viable, since MLIR has no
first-class ragged or jagged tensor type to lower to.

- Leaning: research. This is what makes §4.F answerable at all.
- Depends on: nothing beyond v0.1 to sketch. It needs a real
  feasibility spike — at minimum, a lowering strategy — before any
  engineering time is committed.
- Payoff if it works: a concrete, measured semantic-optimization win —
  attention over real variable-length sequences, §4.F — that a
  shape-erasing compiler cannot claim by construction. See §9.6.

## 9. PyTorch Interoperation and the Merge Point (post-v0.1)

> Everything in this section is out of v0.1 scope. It records a design
> analysis, not a commitment. The goal: let Tensora drop into existing
> PyTorch code, contribute Tensora's semantic and graph-level
> optimizations, and reuse the ecosystem's mature low-level CPU/GPU
> kernels instead of rebuilding them. Tensora deliberately does not
> invest below the IR level.

### 9.1 Correcting the premise: LLVM is too low to be the merge point

One intuitive plan says: let Tensora lower to LLVM, and merge with a
parallel PyTorch LLVM path, so Tensora can use PyTorch's low-level
optimization. This assumption does not hold. PyTorch's real performance
lives below anything like portable LLVM IR:

- **Eager mode** sends each op to a precompiled kernel library —
  cuBLAS/cuDNN on GPU, MKL/oneDNN on CPU, or ATen. These ship as `.so`
  kernels. They are not linkable LLVM bitcode.
- **`torch.compile`** captures an FX graph with TorchDynamo. TorchInductor
  then generates Triton code for GPU, or C++/OpenMP code for CPU,
  compiled per shape at runtime. The CPU path does reach LLVM through
  the C++ compiler. But it is an ephemeral artifact, not a stable `.bc`
  file to link against.

So there is no lasting "parallel PyTorch LLVM-IR path" to merge with.
By the time either side reaches LLVM IR, the decisions that matter —
fusion, layout, kernel choice — are already made. LLVM is too low a
level to be the useful merge point.

### 9.2 The real merge point is MLIR `linalg`, not LLVM

The wider ecosystem already agrees on this point. torch-mlir lowers
PyTorch models — through Dynamo/FX or TorchScript — into MLIR: the
`torch` dialect, then `linalg`, `tosa`, or `stablehlo`. Tensora already
emits `linalg`, `tensor`, `arith`, and `scf` (see architecture ARCH-005
and the codegen lowerer). So the two systems can meet in one MLIR
module, at the `linalg` level. They can share MLIR passes — fusion,
bufferization, tiling — across that boundary, then lower once to LLVM
and the same backend kernels PyTorch's own compiler paths already use.

This is where Tensora's advantage adds to PyTorch, instead of competing
with it. Tensora's semantic verification and graph optimization run
above `linalg`. The low-level performance comes for free, because
Tensora does not redo that work at the LLVM level. Record this decision
— merge at MLIR `linalg`, not at LLVM — as an `ARCH-XXX` when this work
starts. It also depends directly on §8.3 (a `tensora` dialect), if
named-axis facts need to survive to that shared level.

### 9.3 Integration options, ranked by realism

Option 1 below splits into two ordered phases: 1a, then 1b. 1a is the
smaller engineering step, and it is the right prerequisite for 1b, not
an alternative to it. §9.3a records the architecture debate behind this
split, and the ABI it depends on. The rest of the table below is
unchanged from the original analysis.

| # | Option | Cross-boundary optimization | Effort | Precedent |
|---|---|---|---|---|
| 1a | **Explicit custom op + in-memory tensor ABI** — user marks one Tensora region (e.g. `tensora_attention(x, Wq, Wk, Wv)`) via `torch.library.custom_op()`; PyTorch tensors passed by pointer/shape/stride, no `.npy` | at Tensora's own boundary | small | `torch.library` custom ops (stable extension API) |
| 1b | **`torch.compile` backend** — automatic subgraph extraction via Dynamo/`torch.export`, claim supported subgraphs, fall back to eager | at Tensora's own boundary | moderate (needs 1a's ABI first) | ONNX Runtime EPs, torch-tensorrt, Hidet |
| 2 | **MLIR `linalg` convergence** (torch-mlir + Tensora linalg) | yes, across the boundary | hard (research) | IREE, torch-mlir |
| 3 | **Single-binary FFI** (AOT `.o` + C ABI, no `torch.library` registration) — subsumed by 1a for a PyTorch host; still relevant for a non-PyTorch host | none (fixed wall) | low | Rust-in-C, Julia `ccall`, C-extensions |
| 4 | **Literal LLVM-IR merge + LTO** | in principle, in practice little | low mechanically | clang+rustc cross-language LTO |

- **Option 1a is the right first step, ahead of 1b.** It needs no FX
  graph traversal, no ATen decomposition matching, and no Dynamo
  guard/recompile handling. It needs only `torch.library.custom_op()`,
  a stable API PyTorch has committed to keeping stable. The user picks
  which region Tensora optimizes by hand — for example, a
  `linear`+`relu`+`softmax` block, using only operators Tensora already
  has. Everything else in the model stays ordinary PyTorch. This is
  also easy to benchmark right away: run the same model three ways —
  eager, `torch.compile`, and PyTorch-with-Tensora-region. That is a
  real adoption/paper-shaped experiment (§4.A, §4.E).
- **Option 1b builds on 1a.** Once 1a's ABI and region-compilation
  model are proven, replace the explicit `tensora_xxx(...)` calls with
  automatic subgraph extraction, through Dynamo/`torch.export`. This is
  the real "drop-in" end state. 1a is how you reach it without first
  building Dynamo-level machinery.
- **Option 2 is the long-term, principled research path.** This stays
  unchanged. It is the only option with real cross-boundary
  optimization, and it has an active ecosystem to build on. It runs in
  parallel with 1a/1b, not in competition. 1a/1b are the pragmatic,
  adoption-focused track. Option 2 is the research track.
- **Option 3 is narrowed, not dropped.** For a PyTorch host, 1a already
  covers it: 1a is an in-memory FFI call, plus a `torch.library`
  registration, at little extra cost. That registration buys
  `torch.compile`/`torch.export` visibility that Option 3 cannot get.
  Option 3 is still the right answer only when embedding Tensora in a
  non-PyTorch host.
- **Option 4 is the literal reading of the original idea.** It stays
  the weakest choice for this goal (see §9.1). Nothing here changes
  that.

### 9.3a The tensor ABI, boundary granularity, and ownership

This section records the design debate behind the 1a/1b split above,
and the constraints that follow from it.

**Build a runtime ABI that does not depend on any one frontend. Do not
build another `.npy` loader.** `.npy` is the right interop method for
the standalone CLI (ARCH-016). It is a file-based frontend, for
offline, disk-backed data. It is the wrong method for a live PyTorch
call. A `torch.Tensor` already owns contiguous CPU memory. Round-
tripping it through a file would add disk I/O and a copy, for no
reason. The right primitive is a small, stable, in-memory tensor view:

```c
struct TensorView {
    void* data;
    int64_t* sizes;
    int64_t* strides;
    int64_t rank;
    DType dtype;
};
```

This struct describes memory that already exists. It does not own or
load that memory. Both `.npy` today, and a future PyTorch bridge,
should be two frontends onto the same underlying runtime tensor ABI.
They should not be two separate execution paths. This points at a real
gap in the current architecture: ARCH-016 made `.npy` *the* runtime
data-I/O boundary for `--run`, without separating "how tensors enter
Tensora" from "how `.npy` specifically loads them." Close that gap with
its own `ARCH-XXX`, before writing any bridge code.

**Ownership and location.** The bridge is Tensora's own interface into
its runtime. It is not PyTorch's interface. It belongs in the Tensora
project, in a layer that does not pollute `src/codegen/`:

```text
src/
├── codegen/          # compiles Tensora programs -> native code (unchanged)
├── runtime/           # NEW: how an already-compiled kernel receives buffers
integration/
└── pytorch/           # NEW: teaches PyTorch to call src/runtime/'s ABI
```

`src/codegen/` and `integration/pytorch/` do different jobs, and must
stay separate. Codegen compiles programs. The bridge only moves
already-live memory to an already-compiled kernel. This also means the
PyTorch integration is a client of the Tensora runtime, not a fork of
it — the compiler itself stays framework-independent. Building this
pulls in `libtorch`, or the PyTorch C++ extension toolchain, as a new
pinned third-party build dependency. That needs its own `INTEG-XXX`,
the same treatment LLVM/MLIR got in INTEG-003, once this work starts.
No `LANG-XXX` is needed here — the language and grammar do not change.

**Boundary granularity is a hard requirement, not a nice-to-have.**
Crossing the boundary once per operator — one call for `matmul`, then
one for `relu`, then one for `softmax` — lets dispatch and validation
overhead dominate, and erases any benefit. PyTorch already fuses at
that same granularity anyway, through `torch.compile`/Inductor. The
boundary must instead sit at fusion-worthy semantic regions: one call,
for example `tensora_attention_block(x, Wq, Wk, Wv)`, that runs Q/K/V
projection, then scale, then softmax, then V, then output, as one
verified, optimized Tensora graph. This is not a small implementation
detail. It is exactly where §9.4's real advantage — statically verified
fusion — has room to beat `torch.compile`/Inductor's own fusion. So
region-level compilation and the core advantage argument are one
design decision, not two.

**Overhead, stated plainly.** With region-level boundaries, the cost of
each crossing is: dispatch, plus argument validation, plus one native
call. This is small next to the fused computation inside the region.
Three points from this analysis should not be glossed over:

- **"Zero-copy" is conditional, not guaranteed.** A `TensorView` with
  arbitrary strides needs Tensora's MLIR lowering to accept
  non-contiguous input. Today's lowering assumes concrete, and likely
  contiguous, shapes. Until this is checked, the honest claim is:
  zero-copy for contiguous tensors, a fallback copy otherwise. That is
  still far cheaper than a `.npy` round-trip. It is not a universal
  guarantee.
- **Autograd safety.** Tensora is inference-only by charter (see
  [README.md](../../README.md)). A custom op placed inside an
  otherwise-trainable PyTorch model must either register as
  non-differentiable, or fail loudly under `requires_grad`. Otherwise
  it risks silently breaking backpropagation in the rest of the model.
  This is a real correctness risk, not a hypothetical one.
- **GPU support depends on which choice a region uses — see §9.3b.**
  When Tensora compiles a region itself (§9.3b's choice 2), GPU support
  is out of scope for this first bridge: keeping device memory,
  streams, and synchronization correct is a much harder problem than
  the CPU case, and this matches the project's current CPU-only build
  (no GPU backend exists yet). When Tensora instead checks a region and
  calls an existing GPU kernel for it (§9.3b's choices 1 and 3), GPU
  already works today, because Tensora runs none of its own code on
  the GPU.

### 9.3b Dispatch to an existing kernel, instead of compiling one, when a fast one already exists — and why the ragged win still needs care

This section corrects a gap in §9.3a above. §9.3a assumes Tensora
always compiles the wrapped region itself, through its own MLIR → LLVM
→ native path, and PyTorch only calls the result. That is the right
design when no existing, tuned implementation of the region exists.
It is the wrong design when one already does — and for a well-known
op like attention, one already does.

A fast GPU attention kernel — FlashAttention, cuDNN's fused attention,
or a Triton kernel from `torch.compile` — is not a generic result of
lowering math to a GPU. Each one is the product of years of dedicated
engineering: tiling the work to fit fast on-chip memory, never writing
the full attention matrix to slow memory, fusing every step into one
kernel. If Tensora grew its own MLIR GPU-dialect path and lowered
attention through it, the result would be correct, but not
competitive. Building a competitive one from scratch is its own
multi-year research problem — the same one those projects already
solved. This would also work against §9.1/§9.2's own point: reuse
mature kernels, do not rebuild them.

**A plain dispatch to the ordinary dense op is not enough by itself.**
`torch.nn.functional.scaled_dot_product_attention` is fast, but it only
runs on a fixed, padded, dense tensor. Even with a padding mask, it
still allocates the full padded tensor, and still does the full padded
amount of work. Calling it gets PyTorch's kernel speed, but gets that
speed on the wasteful, padded version of the problem — it throws away
exactly the memory and compute win §4.F and §8.11 are about. A correct
design must not stop at "dispatch to the first tuned op that matches."

**So this plan ranks three choices for how a region's math actually
runs, in this order:**

1. **Best: an existing, hand-tuned kernel that is itself ragged-aware.**
   Real kernels already exist that never pad at all. FlashAttention's
   `flash_attn_varlen_func` takes one flattened tensor of all real
   tokens across a batch, plus a small array of sequence lengths — no
   padding, and still a hand-tuned CUDA kernel. PyTorch's own
   `NestedTensor` type, and vLLM's PagedAttention kernel, solve the
   same problem the same way, for real LLM serving. When a kernel like
   this matches the region, Tensora checks the contract, packs the data
   the way that kernel expects, and calls it. This is the only choice
   that gets both the kernel speed and the memory/compute win at once.
2. **Fallback: compile the region with Tensora's own backend**, as
   described in §9.3a. Use this when no existing kernel matches the
   exact ragged shape needed. This still avoids the padding waste. It
   does not match a hand-tuned kernel's speed. This choice stays
   CPU-only for now (but see the TACO/Tiramisu/DISTAL note below for
   how it could later reach a generated GPU kernel).
3. **Last resort: an existing, tuned, but padded op** — for example
   `scaled_dot_product_attention` itself. This gets PyTorch's kernel
   speed, but gives up the padding-memory win. Use this only when
   raggedness does not apply, or is not worth the extra work.

For the ragged-attention case study (§4.F, §8.11), choice 1 is the
target, not choice 3. A design that only reaches choice 3 would keep
PyTorch's kernel speed, but would silently give up the entire reason
this case study exists.

**Which stage does the dispatch, and is there a data-transfer cost?**
Choice 1 and choice 3 are not compiler-lowering steps. An existing
kernel like `flash_attn_varlen_func` is a Python/C++ callable, not an
IR sink Tensora lowers into. So the dispatch happens in the runtime
bridge (`integration/pytorch/`, §9.3a), not in MLIR lowering: Tensora's
semantic layer verifies the region, then the bridge calls the existing
function. The surrounding PyTorch code and Tensora's bridge are both
just callers of the same kernel — that is why they interoperate with no
special handshake. This also means choice 1 has no CPU/GPU transfer
cost: if PyTorch already placed the tensors on the GPU, and the kernel
runs on the GPU, the data never leaves the GPU, because Tensora runs no
compute on it — it only checks, and builds a small sequence-length
array. The GPU/CPU transfer risk applies only to choice 2, where
Tensora's own compiled kernel runs on the CPU today.

**Why Tensora at all, if PyTorch already has `NestedTensor`?** For a
single op that `NestedTensor` already covers, Tensora adds
verification, not capability. PyTorch's ragged support is dynamic, and
checked at runtime if at all. Tensora checks the ragged contract
statically, across a whole region, before running — it catches a wrong
reduction over padded length, or a mismatched batch axis, as a compile
error, not a runtime surprise. So the ragged advantage is weakest
exactly where PyTorch already has a kernel, and strongest for a custom
fused region that no `NestedTensor` path covers. This plan should be
honest about that: "Tensora can run ragged attention" is not the
claim, because PyTorch can too. The claim is "Tensora verifies the
ragged contract and picks the right kernel," and, for regions with no
library kernel, "Tensora compiles a ragged kernel where PyTorch would
fall back to padding." This is the same honest-USP point as §9.4.

**On TACO, Tiramisu, and DISTAL — they upgrade choice 2, not choice 1.**
These are real systems that improve computation at the algorithm and
schedule level: TACO for sparse tensor algebra, Tiramisu for polyhedral
schedule transforms, DISTAL for distributed mapping. Their benefit is
above the kernel, at the IR/schedule level. That means they could feed
Tensora's own compile path (choice 2): use their algorithm, express the
result in MLIR, and lower it through MLIR's own GPU path (the LLVM
NVPTX backend) to a generated GPU kernel. That would make choice 2 both
stronger and GPU-capable, for regions where no library kernel exists.
What they cannot do is route their output into cuDNN or FlashAttention.
Those are fixed, precompiled library kernels, reached by dispatch, not
"from LLVM IR to kernel" — the same point as §9.1. So their algorithm
plus PyTorch's exact hand-tuned kernel cannot be stacked for one op.
For a given region you get either the library kernel (choice 1), or
your own IR-generated kernel improved by these systems (an upgraded
choice 2) — not both fused together. Wiring any of them in is its own
large project, out of scope for V0.2, but it is the natural way to make
choice 2 competitive later.

### 9.4 Honest USP consequence

If Tensora does not touch kernels, every advantage must come from the
graph or semantic level: static shape specialization, verified
fusions, algebraic rewrites, and schedules chosen before lowering.
`torch.compile`/Inductor and XLA already do a lot of graph-level
fusion. So the real, defensible advantage is narrower: statically
verified, semantically guaranteed optimization — correctness-driven,
provable from named-axis contracts — feeding the same LLVM/kernel
backends. It is not a faster matmul. This ties directly to
paper-shaped question 4.A, and to constructs §8.3–§8.5.

### 9.5 Operator-gap dependency

Options 1a, 1b, and 2 all cover only the subgraphs Tensora can already
express. The operator gap to a full pretrained model is already listed
in §8.9a (`gelu`, `layernorm`, `rename`) and §8.9 (integer element
type, `gather`, `sin`/`cos`, `avgpool`). Interop does not remove this
gap. It lets Tensora handle the subgraphs it does support, while
PyTorch handles the rest. That graceful fallback is exactly why 1a/1b
are worth building before the gap is closed.

Option 1a narrows this gap further. The user picks the region by hand —
for example, one `linear`+`relu`+`softmax` block, using only operators
Tensora already has (see §9.3a). So the operator gap only needs to
close for *that* region, not for the whole surrounding model. This is
a real advantage over converting an entire pretrained model, where the
§8.9a/§8.9 gap covers a full Transformer or embedding path. A useful
first integration demo needs zero new operators, as long as the chosen
region is a linear/matmul+activation block, or a precomputed-float
attention subgraph — not, for example, a whole BERT model end to end.

### 9.6 Priority ranking: interop, the operator gap, and the ragged-tensor case study

Three questions deserve a direct answer, instead of staying implicit
across separate sections.

**Is closing the full-BERT operator gap (§8.9a/§8.9) the top
priority?** No. Closing operator gaps is necessary infrastructure. It
is not a goal by itself. Reproducing more of what other frameworks
already do does not, on its own, create adoption or research interest.
§8.9a is different: it is small, and it is a shared enabler for two
real priorities — a real attention-block integration demo for §9.3/1a,
and a real transformer layer for the ragged-attention case study
(§4.F/§8.11). Treat §8.9a as a cheap prerequisite for those, not as a
goal on its own. §8.9, the integer-tensor/`gather` tier, is a bigger
lift. It is urgent only if a specific goal needs the embedding/LLM
input path. Do not build it "to get closer to BERT" as an end in
itself — another BERT reproduction has little value by itself.

**Is PyTorch interop too important to leave buried in a "post-v0.1
design analysis" section?** Yes. Section 2 now corrects this. Section
2's own goal — usability over feature count — is exactly the reason to
pull Option 1a (§9.3) forward. It is a small, concrete engineering
task: a `torch.library` custom op, plus the `TensorView` ABI in §9.3a.
It is also what makes every other investment in this roadmap usable
outside Tensora's own command line. Sequence it as an early v0.2+/v0.3
deliverable. Do not defer it until every operator gap is closed.

**Should a ragged-tensor, CoRa-style win (§4.F/§8.11) be the flagship
semantic-optimization case study?** Yes, with the feasibility caveat
already stated in both sections. It is the one candidate in this whole
roadmap that ties Tensora's specific advantage — verified semantic
facts that survive to the optimizer — to a measured, prior-art-backed
performance win, based on CoRa's own results on ragged and attention
workloads. This is stronger than a general claim that "semantic
information might help," which any shape-aware compiler could also
make. It gives people a concrete reason to look into semantic-aware
optimization, more sharply than 4.A–4.E do on their own. The honest
tradeoff: this is also the single largest type-system change in this
document (§8.11). Run its feasibility spike before picking it as the
Section 5 paper question. Do not assume it will work.

One more honest point, carried over from §4.F: showing that Tensora
uses less memory than padded PyTorch, on the same development machine,
is a compiler-side result. It is not yet an edge-device result. To
claim an edge-device advantage, measure the same compiled model on a
real constrained device, and report peak RAM, model size on flash,
latency, and energy use. See [13_V0.2_PLAN.md](13_V0.2_PLAN.md) for
this plan.

**Net sequencing this implies:** §8.9a first (it is cheap, and shared
by both later items). Then §9.3/1a (interop, for usability), and, in
parallel once §8.9a exists, a §8.11 feasibility spike (research). §8.9,
and the rest of §8's heavier constructs, follow only if a specific goal
needs them. They are not built by default.

### 9.7 Be critical: language vs a PyTorch pass

A hard question worth facing directly: **is a new language justified for
this goal at all, or should the semantics be added to PyTorch instead?**

Honest answer: for the real-world optimization goal, a new surface
language is **not** justified. Requiring users to rewrite a model in
Tensora syntax is an adoption barrier with little payoff. Two separate
things must not be conflated:

- **Tensora the language** justifies itself as a teaching and research
  compiler (v0.1). That is its charter, and it is done.
- **The transferable asset** is the semantic analysis — verified
  named-axis and ragged contracts — and the graph-level orchestration
  it enables. This does not need a new surface syntax. It can live as a
  pass over PyTorch's own graph.

**Resolving the apparent contradiction.** This roadmap says both "extra
semantics enable optimization" and "you cannot combine semantic
optimization with a tuned kernel." The second is too strong. You cannot
inject a pass inside cuDNN or FlashAttention. But you can, at the graph
level, choose the ragged kernel over the padded one, fuse the trailing
ops, and skip padding work. That decision is the semantic win, and it
composes with the hand-tuned kernel at the dispatch boundary. Semantics
decide what runs; tuned kernels run it fast.

**The creative insertion point.** Deliver the semantics as a PyTorch
graph pass (`torch.fx`, or a `torch.compile` backend), not a language.
The pass: (a) recovers and verifies the semantic contract on the
captured graph as metadata, not a rewrite; (b) rewrites the graph to
exploit it — ragged kernel selection, fusion, padding elimination;
(c) hands the graph back to Inductor and library kernels for low-level
codegen.

**The critical caveat.** `torch.compile`/Inductor already captures
graphs, fuses, and can call FlashAttention. The only thing a semantic
pass adds is verified per-sequence ragged contracts that Inductor does
not carry, and the ragged rewrites they unlock. Whether that beats a
well-configured `torch.compile` + `NestedTensor` + FlashAttention path
is an open experiment, not a claim. The defensible niche is narrow:
custom fused regions where PyTorch falls back to padding. This is the
same experiment as §4.F, and it must be run before any adoption claim.
