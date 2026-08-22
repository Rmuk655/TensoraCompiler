# Tensora Open-Source and Research Roadmap

This is not a required final-report chapter unless the team chooses to
include it. It separates semester commitments from ambitious extensions.

## 1. Course Release: v0.1

Must be complete, tested, documented, and reproducible.

-   language spec;
-   frontend;
-   typed graph;
-   semantic checker;
-   CPU MLIR/LLVM path;
-   custom optimization;
-   examples;
-   test corpus;
-   explain mode.

## 2. Open-Source Release: v0.2

Prioritize usability over feature count:

-   installation script/container;
-   concise quick start;
-   stable diagnostics;
-   graph visualization;
-   more operators;
-   benchmark harness;
-   real-world failure corpus;
-   contributor documentation.

## 3. ML-Systems Extension: v0.3

Possible directions:

-   GPU lowering;
-   layout propagation;
-   memory-cost estimation;
-   richer fusion;
-   attention graph recognition;
-   static-shape specialization;
-   interoperability with an external model/graph representation.

## 4. Paper-Shaped Questions

A paper requires a question, baseline, method, and evaluation.

### A. Semantic Information for Optimization

Does retaining high-level verified tensor information enable
transformations or specialization that are harder after generic
lowering?

### B. Failure Detection

How many real tensor-program failures in a curated corpus can Tensora
detect statically, and what classes remain dynamic?

### C. Layout Optimization

Can semantic graph information reduce transpose/layout-conversion
overhead on selected ML workloads?

### D. Memory-Aware Graph Optimization

Can static shape information predict expensive intermediates and guide
transformations?

### E. Small Inference Workloads

Can compiler transformations reduce overhead in repeated small inference
graphs, including autoregressive-style workloads?

Do not attempt all five. Select one with faculty after v0.1 exists.

## 5. Experimental Discipline

For any research extension record:

-   baseline;
-   hardware;
-   software versions;
-   workload;
-   compiler flags;
-   warmup;
-   repetitions;
-   variance;
-   generated IR;
-   correctness check.

## 6. Relationship to Existing Ecosystems

Treat PyTorch, JAX, TensorFlow, XLA, MLIR, LLVM, TVM, Triton, and tensor
languages as related systems to learn from and compare against, not as
strawmen.

## 7. Open-Source-to-Paper Path

This section ties Section 2 (open-source release), Section 4 (paper
questions), and Section 5 (experimental discipline) into one explicit,
ordered path, so "release it as open source" and "write a paper" are
not two separate, disconnected tracks:

``` text
v0.1 (course, graded, frozen)
   |
v0.2 open-source release (Section 2)
   -> real users, real issues, real usage patterns -- not guessed ones
   |
mine v0.2 feedback for evidence
   -> which paper-shaped question (Section 4) does real usage actually
      support? e.g. a spike of shape/axis bugs in reported issues
      strengthens question 4.B; a request for layout control strengthens
      4.C
   |
pick ONE question with faculty, informed by that evidence
   -> do not pick blind before v0.2 exists; real adoption data is the
      whole point of releasing first
   |
define baseline + methodology (Section 5) for that one question
   |
prototype the pass/analysis needed (often 8.1/8.3, since most questions
depend on semantic facts surviving into the IR)
   |
run the experiment, compare against baseline
   |
write the paper; upstream reusable infrastructure where appropriate
```

The open-source release is not just a distribution step -- it is the
mechanism that turns Section 4's five candidate questions into one
evidence-backed choice, instead of a guess made before any real user
exists. Skipping v0.2 and picking a paper question directly from v0.1
is possible but loses this evidence; only do that if the timeline does
not allow for a real release-and-feedback cycle before the faculty
conversation.

## 8. Future Language Constructs (post-v0.1)

> Everything in this section is **out of v0.1 scope**. It does not affect
> the frozen whitepaper, reference manual, architecture design, or test
> plan (all of which are v0.1-only). All constructs are **inference-only**
> (forward computation, no autograd/training), consistent with the v0.1
> boundary. Framing is deliberately left open: each construct is written so
> it can be pursued as open-source engineering or as a measured research
> contribution once v0.1 exists.

### 8.0 Prior art and the gap (honest positioning)

These systems are close neighbors and must be cited, not reinvented:

| System | Kind | Static verification | Named axes | Semantics in IR |
|---|---|---|---|---|
| Tensor Comprehensions (arXiv:1802.04730) | source DSL, Einstein-style index notation | none | no | n/a (polyhedral JIT -> CUDA) |
| MLIR Toy tutorial | teaching DSL (rank <= 2, f64) | shape inference only | no | toy dialect |
| MLIR `tensor` dialect | IR (not a language) | structural/shape only | explicitly none | it is the IR |
| TensorFlow MLIR | production bridge | shape inference | no (some attrs) | TF/HLO dialects |
| **Tensora v0.1** | declarative graph language | strong named-axis contracts | yes | TCG, then erased at MLIR |

The gap none of them fill: a source tensor language whose **named-axis
semantic contracts are statically verified _and_ preserved into the IR /
optimizer**. Tensor Comprehensions has the expressive index notation but
zero verification; Toy and the `tensor` dialect have no names and no
contracts; TensorFlow lowers semantics away to HLO. The constructs below
target that gap. Tensora does **not** claim to invent index notation,
polyhedral compilation, autotuning, custom dialects, or fusion — only to
combine named-axis verification with their preservation.

### 8.1 Verified index-notation operators (spine construct)

Bring Tensor-Comprehensions-style index notation, but tie every index to a
declared, verified named axis. Sketch:

``` tensora
op matmul(a: Tensor[batch=B, i=M, k=K],
          b: Tensor[batch=B, k=K, j=N]) -> Tensor[batch=B, i=M, j=N] {
    out(batch, i, j) = sum(k) a(batch, i, k) * b(batch, k, j)
}
```

Static rules (the differentiator over TC): every index must correspond to
a declared axis name; each index resolves to exactly one axis size (via
the v0.1 bounded solver); free indices (those on the left) must equal the
output axes; summation indices must be bound on the right and absent from
the output. It is forward/inference-only: `sum(k) ...` is a contraction,
not a gradient. Lowers to `linalg.generic` with the derived iteration
space.

- Leaning: research (verification-over-notation is the novel bit) with
  clear open-source utility.
- Depends on: nothing beyond v0.1; it is the foundation for 8.2 and 8.4.

### 8.2 User-defined operators

Let users define new named operators from 8.1's index notation plus an
explicit contract (preconditions over named axes), checked like the
built-ins and lowered through the same path:

``` tensora
op split_heads(x: Tensor[batch=B, seq=S, model=D])
    -> Tensor[batch=B, head=H, seq=S, dim=Dh]
    where { D == H * Dh; } {
    out(batch, h, seq, dh) = x(batch, seq, h * Dh + dh)
}
```

Tensor Comprehensions allows custom operators but verifies nothing;
Tensora's version type-checks the contract and the index algebra. This is
how the frozen built-in operator set becomes *extensible without touching
the compiler*.

- Leaning: open-source (extensibility) with research value (contract
  inference).
- Depends on: 8.1.

### 8.3 A `tensora` MLIR dialect

A custom dialect that carries named axes and proven contract facts as
attributes on high-level ops, so IR-level passes can consume them before
the ops are lowered to `linalg`/`tensor`. The `tensor` dialect explicitly
cannot represent this; v0.1 therefore erases names at the MLIR boundary
(ARCH-005). This construct is what makes the "semantics survive
compilation" claim empirically testable rather than rhetorical.

- Leaning: research (it is the substrate for the paper-shaped question in
  Section 4.A).
- Depends on: v0.1 lowering; enables 8.4, 8.5, 8.6.
- Cost note: real work (TableGen ops/types, verifiers, a conversion pass);
  only justified once a downstream pass needs preserved patterns.

### 8.4 Contract-driven specialization and autotuning

Use *proven* contract facts (divisibility, head counts, concrete sizes,
`hidden == heads * head_dim`) as priors to generate specialized variants
and to **prune** an autotuning search. Tensor Comprehensions autotunes
without such priors; Tensora's verified facts constrain the space before
search begins.

- Leaning: research (measurable: does using contract priors reduce
  autotuning cost or improve chosen schedules?).
- Depends on: 8.3 (facts must survive into the IR to be usable there).

### 8.5 Layout / axis-semantics optimization

Use axis *meaning* (not just size) to minimize layout conversions —
eliminate redundant transposes, choose a physical layout per axis role,
and recognize when a `transpose` is semantically a no-op for the
consumer. Extends the v0.1 transpose-cancellation pass with axis-aware
reasoning.

- Leaning: research (paper-shaped question 4.C) plus open-source perf.
- Depends on: 8.3 (names available in the IR).

### 8.6 Polyhedral loop optimization

Tensor-Comprehensions-style polyhedral scheduling (tiling, fusion, loop
transformation) applied to the verified, named-axis IR. Heaviest and most
research-intensive; likely leans on MLIR's affine/polyhedral facilities
rather than a from-scratch scheduler.

- Leaning: research (long horizon).
- Depends on: 8.1 (index notation gives clean iteration spaces) and 8.3.

### 8.7 Verifier and real-world ML failure corpus as a product

Package the semantic verifier plus a curated corpus of *real* ML tensor
bugs (from cited public issues / model code) as standalone open-source
value. None of the neighboring systems verify named-axis contracts, so a
"catch these N real shape/axis/contract bugs before you run" tool is
defensible on its own.

- Leaning: open-source (adoption); supports paper-shaped question 4.B.
- Depends on: v0.1 only. Lowest-risk of the seven; good first post-v0.1
  deliverable.

### 8.8 Suggested ordering

``` text
v0.1 (frozen)
  -> 8.7 verifier + failure corpus        (open-source, low risk, no new language)
  -> 8.1 verified index notation          (foundation)
  -> 8.2 user-defined operators           (extensibility, builds on 8.1)
  -> 8.3 tensora dialect                 (research substrate)
  -> 8.4 / 8.5 contract-driven passes     (measured research on top of 8.3)
  -> 8.6 polyhedral                        (long-horizon research)
```

Pick the branch (open-source vs research) after v0.1 exists and after the
faculty conversation in Section 4; this section is written to support
either.

### 8.9a Transformer-layer completeness tier — open-source future

V0.1 was rescoped to a **compiler-course MVP** that optimizes for depth
over breadth (small operator set; SSA graph; a tensor-liveness analysis; a
structured-loop tiling transformation; ANTLR frontend). As part of
that rescope, three constructs that a *complete* Transformer encoder/
decoder layer needs were moved **out of v0.1** into this future tier:

- **`gelu`** — the feed-forward activation real Transformers use.
- **`layernorm`** — normalization over a named axis (one operator, so the
  surface language needs neither `sqrt` nor keepdim reductions).
- **`rename`** — a type-level axis relabel, needed because self-attention
  from a single input produces two axes that both originate from `seq`,
  and because a `linear` output (auto-named `out`) must be relabeled to
  match a residual add.

With these three, a complete Transformer encoder/decoder layer (layernorm
-> multi-head self-attention -> residual -> layernorm -> gelu FFN ->
residual) becomes expressible entirely from operators — no type-system
change required. They are the natural **first** open-source additions after
v0.1, ahead of the heavier integer-tensor tier below. (V0.1's Transformer
showcase is `examples/attn_text_classifier.tns` — a real self-attention
**classifier** with PyTorch-trained weights, running end to end through
Tensora; see `tools/models/prepare_text_attn_demo.py`. A full pretrained
**BERT/DistilBERT extractive-QA** model is specifically what needs this
tier plus §8.9: `gelu` + `layernorm` + `rename` for the encoder layers,
and the integer element type + `gather` below for the token-embedding
input path. Until then, no pretrained QA model can be reproduced
faithfully, since every one uses LayerNorm, GELU, and integer-indexed
embeddings.)

### 8.9 Integer-tensor + gather tier (the LLM input path) — open-source future

> Note: this tier is what remains between v0.1 (plus §8.9a) and a full
> language model, and is kept as future scope because it reopens the type
> system.

v0.1 verifies any feed-forward network and any Transformer layer/body, but
not a full LLM, because the token/positional **embedding lookup** needs
integer-indexed tensors that v0.1's f32-only type system does not have.
This tier adds, as a coherent unit:

- **An integer element type** — a minimal dtype system (f32 plus `i32`/`i64`
  index tensors). This is the load-bearing change; everything else here
  depends on it.
- **`gather(table, indices)`** — embedding lookup. With
  `table: Tensor[vocab=V, model=D]` and integer
  `indices: Tensor[batch=B, seq=S]`, produces
  `Tensor[batch=B, seq=S, model=D]`. Covers token embeddings, learned
  positional embeddings, and recommender embedding tables.
- **`sin` / `cos`** — elementwise, enabling rotary positional encodings
  (RoPE).
- **`avgpool` / global average pool** — completes ResNet-class CNNs
  (inference-time BatchNorm already folds into `mul`+`add`).

With this tier, the LLM front end (embedding -> transformer body -> LM
head) and recommender models become expressible end to end; everything
downstream of the embedding is already covered by v0.1 plus `layernorm`,
`gelu`, and `rename`.

Why it is future, not v0.1: introducing an integer element type reopens
the type system (mixed int/float operator contracts, index-range as a
semantic concern, dtype in the surface syntax and in lowering) — a
materially larger and higher-risk change than the three v0.1 additions,
and unnecessary for the v0.1 MVP claim ("verifies any feed-forward net and
any Transformer layer"). It is the natural first open-source milestone
after v0.1.

- Leaning: open-source (adoption: "verify and run real embedding-based
  models"); also strengthens paper-shaped question 4.B (failure detection
  on real LLM/recommender code).
- Depends on: v0.1 (+ the integer element type, which this tier
  introduces). Independent of 8.1–8.6.

### 8.10 Documentation versioning (V0.1 course vs open-source)

The language reference and tutorial are handled on **two tracks**, so the
course scope never blurs:

- **V0.1 (course), normative and graded.** `docs/language/03_LANGUAGE_
  REFERENCE_MANUAL.md` and `docs/language/02_LANGUAGE_TUTORIAL.md` contain
  **only** V0.1 scope and are the artifacts defended in the viva. They stay
  small and course-focused. Nothing from Section 8 appears in them.
- **Beyond V0.1 (open source), non-normative until built.** Every future
  construct lives **only here in Section 8** as a sketch. It is *not* added
  to the V0.1 reference or tutorial. When the open-source version actually
  **implements** a construct (v0.2+), that construct graduates from a
  Section-8 sketch into a **separate, version-tagged** manual and tutorial —
  e.g. `docs/opensource/REFERENCE.md` / `TUTORIAL.md` created at that time —
  which may start as a copy of the V0.1 docs and then extend.

Rule of thumb: a reference manual documents what is **built and frozen**.
Documenting unbuilt features in the normative manual would mislead a reader
(and a grader), so the V0.1 manual/tutorial carry no beyond-V0.1 material,
and the open-source manual is a *future* artifact seeded by this roadmap
and promoted on implementation.
