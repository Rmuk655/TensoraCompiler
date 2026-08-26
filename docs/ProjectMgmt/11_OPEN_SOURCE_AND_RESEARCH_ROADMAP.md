# Tensora Open-Source and Research Roadmap

> The runnable V0.3 research spikes, the falsifiable thesis documents
> (ragged tensors, extent contracts, polyhedral optimization, PyTorch and
> StableHLO interop), and the full V0.2 plan/closeout/future-research
> documents (`13_V0.2_PLAN.md`, `14_V0.2_CLOSEOUT.md`,
> `15_FUTURE_RESEARCH_DIRECTIONS.md`) all live on the `v0.3-research`
> branch. `main` holds the frozen v0.1 compiler and this roadmap, which
> carries forward only the broad, load-bearing conclusions.

## 1. Release V0.1

V0.1 is done. It is complete, tested, documented, and reproducible.

Done:

-   language spec;
-   frontend;
-   typed graph;
-   semantic checker;
-   CPU MLIR/LLVM path;
-   custom optimization;
-   examples;
-   test corpus;
-   explain mode.

## 2. V0.2: closed (spikes, not a build)

V0.2 was deliberately scoped small: not a build, but a set of runnable
spikes to quick-check whether any future direction gives a real-world
benefit worth open-sourcing, before committing engineering time. It is
closed. The full spike code, numbers, and the original plan document
now live on the `v0.3-research` branch
(`docs/ProjectMgmt/13_V0.2_PLAN.md`, `14_V0.2_CLOSEOUT.md`,
`research/ragged_spike/`). What follows is the conclusion, carried
forward because it decides what v0.3+ should attempt.

**Conclusions (high level; full detail and numbers on the branch):**

-   No easy "faster than PyTorch" win exists. Core PyTorch already
    covers ordinary ragged attention, and single-node scheduling
    (`torch.compile`/Inductor) has little headroom left to beat.
-   Plain polyhedral loop optimization (tiling, vectorization) is
    reachable in Tensora today, but it uses no Tensora-specific
    semantics — it shipped as settled v0.1 engineering (the
    `--polyhedral` flag, ARCH-018), not a research result.
-   The one surviving, narrow research direction is **verified
    relationships among the contents of per-instance metadata** (for
    example, that a `lengths` array is the true extent of the tensor it
    describes, or that `offsets` is a checked prefix-sum of `lengths`).
    No system checked so far verifies this; it still needs a decisive
    experiment before it is a claim, not a hope.
-   Static semantic verification (named-axis/shape contracts) is a real,
    undominated capability, but it is a checker, not a speedup claim.

See §3 below for how this shapes the future-versions plan, and
`docs/ProjectMgmt/15_FUTURE_RESEARCH_DIRECTIONS.md` on the
`v0.3-research` branch for the full research framing and numbers.

### 2a. Is v0.1 ready to open source?

**Yes — as an academic, teaching compiler, right now.** It is complete,
tested, documented, and reproducible (§1). That is a real, defensible
release on its own, independent of anything below.

**Open-sourcing it further, for real-world *adoption* rather than as a
teaching artifact, needs two things neither of which exists yet:**
PyTorch interop (so it can be used without rewriting a model in a new
language), and a measured, real performance or verification win over
what PyTorch already does on its own. Part 1 below is how to find out
whether the second one exists; Part 1b is how to build the first one
regardless of the answer to the first.

## 3. Future Versions (v0.3+): Two Parts

Everything after v0.1 splits into two parts that answer different
questions and run on different timelines. Do not conflate them.

**Part 1 — research: does Tensora have a real edge, and can it reach
users at all?**

-   **1a. Semantic-knowledge advantage (research).** Does the
    ML-specific semantic knowledge Tensora's type system can keep give
    it something PyTorch, `torch.compile`, JAX/XLA/StableHLO, and other
    commercially available compilers cannot already reach on their own?
    §2 already rules out the easy versions of this claim; what remains
    is narrow and still unproven. The candidate questions below (A-F)
    are this part's detail, with F corrected to match that finding.
-   **1b. PyTorch-native embedding (research and engineering).**
    Tensora cannot compete with PyTorch's low-level kernels (§9.1/§9.4),
    so any real value must be delivered *above* that layer, embedded as
    a first-class citizen inside PyTorch, not as a competing standalone
    runtime. §8 below is this part's detail. Worth doing even if 1a
    finds nothing — it is the adoption prerequisite either way.

**The goal is the idea, not the language.** If 1a finds a real win, the
right outcome is to get it into users' hands with the least friction —
as a PyTorch graph pass, or, if general enough, a contribution to
PyTorch core or an MLIR/StableHLO RFC (§9.7) — not necessarily to keep
Tensora alive as a permanent, separate language. Tensora-the-language's
job is already done by v0.1 (§2a).

**Part 2 — eventual open source (engineering, for adoption).** Close
the remaining operator gaps (§8.9a, §8.9), and ship the smaller v0.2
items that never depended on the spike results (a container image, a
general benchmark harness, the real-world failure corpus in §8.7). §7
below is this part's detail.

### Part 1a candidate questions (paper-shaped)

A paper needs a question, a baseline, a method, and an evaluation.

### A. Semantic Information for Optimization

Does keeping verified tensor information help the compiler? Does it
enable transformations that are hard after generic lowering?

### B. Failure Detection

How many real tensor-program failures can Tensora catch by static
checks alone? Use a curated corpus of real failures. Which failure
classes stay dynamic (cannot be caught statically)?

### C. Layout Optimization

Can semantic graph information cut transpose and layout-conversion
overhead? Test this on selected ML workloads.

### D. Memory-Aware Graph Optimization

Can static shape information predict which intermediate tensors are
expensive? Can it guide transformations to avoid that cost?

### E. Small Inference Workloads

Can compiler transformations cut overhead in repeated small inference
graphs? This includes autoregressive-style workloads.

### F. Ragged / Variable-Length Axis Optimization

Ordinary ragged attention is not the opportunity: core PyTorch
(`torch.nested`) already covers it (§2). The corrected, narrower
question: can Tensora verify relationships among the *contents* of
per-instance metadata (a length array, a prefix-sum offsets array, a
transfer function between an input and output extent) that no checked
system verifies today, and does keeping that verified fact change a
downstream legality or scheduling decision? See
`research/ragged_spike/PROPOSAL.md` on the `v0.3-research` branch for
the full scope, the prior-art bound, and the still-open decisive
experiment.

Do not attempt all six questions. Pick one with faculty after v0.1
exists.

## 4. Experimental Discipline

For any research extension, record:

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

## 5. Relationship to Existing Ecosystems

Treat PyTorch, JAX, TensorFlow, XLA, MLIR, LLVM, TVM, Triton, and other
tensor languages as related systems. Learn from them and compare
against them. Do not treat them as strawmen.

## 6. Open-Source-to-Paper Path

This section links three things into one path: Section 2 (V0.2's
conclusions), Section 3 (candidate questions), and Section 4
(experimental discipline). "Release it as open source" and "write a
paper" are not two separate tracks. They are one path:

``` text
v0.1 (frozen, open-sourced academically now -- §2a)
   |
real users, real issues, real usage patterns -- not guessed ones
   |
mine that feedback for evidence
   -> which candidate question (Section 3) does real usage actually
      support?
   |
pick ONE question with faculty, informed by that evidence
   -> do not pick blind; real adoption data is the whole point of
      releasing first
   |
define baseline + methodology (Section 4) for that one question
   |
prototype the pass/analysis needed (the branch's §8.1/§8.3 constructs,
since most questions depend on semantic facts surviving into the IR)
   |
run the experiment, compare against baseline
   |
write the paper; upstream reusable infrastructure where appropriate
```

Releasing v0.1 is not just a way to share the code. It is how Section
3's candidate questions turn into one evidence-backed choice, instead of
a guess made before any real user exists. You can skip waiting for that
feedback and pick a paper question straight from v0.1's own analysis.
But that loses this evidence. Only do that if there is no time for a
real release-and-feedback cycle before the faculty conversation.

## 7. Future Language Constructs and Operator Gaps (Part 2 detail)

> Out of v0.1 scope; does not change the frozen whitepaper, reference
> manual, or test plan. Full design (sketches, prior-art table, ordering
> rationale) lives on the `v0.3-research` branch,
> `docs/ProjectMgmt/16_FUTURE_CONSTRUCTS_AND_INTEROP_DETAIL.md`.

No neighboring system (Tensor Comprehensions, MLIR's Toy tutorial, the
plain `tensor` dialect, TensorFlow-MLIR) combines statically verified
named-axis contracts with keeping those names live in the IR and
optimizer. That is the gap the constructs below target.

-   **Verified index-notation operators** — tie every index in an
    operator body to a declared, checked axis name (the foundation for
    user-defined operators below).
-   **User-defined operators** — let users add named operators, checked
    the same way built-ins are.
-   **A `tensora` MLIR dialect** — carry named axes and proven facts as
    IR attributes, instead of erasing them at the MLIR boundary. This is
    the substrate Part 1a's semantic-advantage question needs to be
    testable at all.
-   **Contract-driven specialization, layout optimization, polyhedral
    scheduling** — use proven facts (divisibility, concrete sizes, axis
    roles) to prune autotuning search, cut layout conversions, and guide
    loop transforms. Long-horizon research; see the sub-polyhedral
    (UTVPI) scheduling citation on the branch for keeping the search
    tractable.
-   **Verifier + real-world failure corpus** — package the semantic
    checker with a curated corpus of real shape/axis bugs. Lowest risk,
    open-source, adoption-facing; a good first Part 2 deliverable.
-   **Operator-gap tiers (the concrete Part 2 work):**
    -   §8.9a: `gelu`/`layernorm`/`rename` — completes a real Transformer
        layer (v0.1's showcase, `examples/attn_text_classifier.tns`,
        already proves a self-attention classifier end to end without
        these).
    -   §8.9: integer element type + `gather` (+ `sin`/`cos`, `avgpool`)
        — the LLM/embedding input path; a bigger, type-system-reopening
        lift, only worth it once a specific goal needs it.
-   **Ragged/variable-length axes** — a new *kind* of dimension (one
    length per batch element, not a shared size), needed if Part 1a's
    semantic-advantage question is tested via a real ragged-shaped
    language construct. This is the largest single type-system change in
    this roadmap. It is not, on its own, the flagship research claim —
    §2's V0.2 conclusions narrowed that to verified metadata-content
    relationships, which may or may not need this construct to test. Run
    a feasibility spike (including a real MLIR lowering strategy — MLIR
    has no built-in ragged tensor type) before committing engineering
    time.
-   **Documentation versioning** — the v0.1 reference/tutorial stay
    frozen and normative; any built future construct gets its own
    version-tagged manual, not an edit to the v0.1 ones.

## 8. PyTorch Interoperation (Part 1b detail)

> Out of v0.1 scope; records a design analysis, not a commitment. Full
> design (the ABI, the ranked option table, the dispatch-choice
> ranking) lives on the `v0.3-research` branch,
> `docs/ProjectMgmt/16_FUTURE_CONSTRUCTS_AND_INTEROP_DETAIL.md`.

**The merge point is MLIR `linalg`, not LLVM.** By the time either
Tensora or PyTorch reaches LLVM IR, the decisions that matter — fusion,
layout, kernel choice — are already made; PyTorch's real performance
lives below that, in precompiled kernels (cuBLAS/cuDNN/ATen) or
`torch.compile`'s per-shape Triton/C++ codegen. Tensora already emits
`linalg`/`tensor`/`arith`/`scf`, the same level torch-mlir lowers PyTorch
into. That is where the two systems can actually meet.

**The pragmatic path is a `torch.library` custom op first, then
automatic subgraph extraction.** A user marks one region by hand (e.g.
`tensora_attention(x, Wq, Wk, Wv)`) through PyTorch's stable custom-op
API — no FX graph traversal or Dynamo machinery needed. Once that ABI
and region-compilation model works, the natural next step is automatic
subgraph extraction via `torch.compile`/`torch.export`. Full MLIR
`linalg` convergence (option 2 on the branch's table) is the long-term,
principled path, running in parallel, not competing.

**When a fast, existing kernel already covers a region (e.g.
FlashAttention for attention), dispatch to it rather than compiling a
competitive one from scratch** — building a competitive GPU kernel is
its own multi-year research problem others already solved. Where a
region needs raggedness a dense kernel can't give for free, prefer a
kernel that is itself ragged-aware (e.g. `flash_attn_varlen_func`) over
a padded one, so the verified extent contract does not go to waste.

**Honest USP.** If Tensora does not touch kernels, every advantage has
to come from the graph/semantic level: static shape specialization,
verified fusions, schedules chosen before lowering — not a faster
matmul. `torch.compile`/Inductor already does a lot of graph-level
fusion, so the defensible niche is narrow: verified, semantically
guaranteed optimization feeding the same backends everyone already uses.

**Is a new language even justified for this goal?** No. For real-world
adoption, requiring a rewrite into Tensora syntax is a pure adoption
cost with no offsetting benefit. The transferable asset is the
verified-semantics analysis itself, which does not need a new surface
language — it can run as a pass over PyTorch's own captured graph
(`torch.fx`, or a `torch.compile` backend): recover and verify the
contract as metadata, rewrite the graph to exploit it, then hand back to
Inductor and library kernels for the actual codegen. This is the same
point §3 makes: the goal is the idea, not the language.
