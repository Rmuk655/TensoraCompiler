# Capability matrix (V0.2 P0)

The point of the spikes was to find the *missing row*: a capability that
no existing system covers well, that Tensora could. This matrix is that
search, filled honestly.

Legend: **T** = tested here (this repo), **P** = from the system's paper/
docs, **X** = unsupported, tested here, **x** = unsupported, per docs/
paper, **?** = unknown/not checked, **n/a** = not that kind of system.

Columns: Tensora v0.1 · PyTorch jagged · CoRa · TACO · Tiramisu · DISTAL
· vLLM · llama.cpp.

| capability | Tensora | PT jagged | CoRa | TACO | Tiramisu | DISTAL | vLLM | llama.cpp |
|---|---|---|---|---|---|---|---|---|
| ragged representation (1 dim) | x | **T** yes | P yes | ? | ? | ? | P yes | P yes |
| **multiple independent ragged dims** | x | **X no** | ? (likely no) | ? | ? | ? | x | x |
| dynamic ragged extent | x | T yes | P yes | ? | P (param) | ? | P yes | P yes |
| reduction over ragged axis | x | **T yes** | P yes | P yes | P yes | ? | P | P |
| positional index on ragged axis | x | **X no** | ? | ? | ? | ? | ? | ? |
| KV-cache decode | x | **X no** | ? | n/a | n/a | n/a | **P yes** | P yes |
| GQA / MQA | x | **X no** | ? | n/a | n/a | n/a | P yes | P yes |
| ragged fusion (region) | x | **T yes** | P yes | ? | P yes | ? | P | P |
| CPU codegen | P (LLVM) | **X no** | P yes | P yes | P yes | P | x | **P yes** |
| GPU codegen | x | **T yes** | P yes | ? | P yes | P | P yes | P (some) |
| **static semantic verification** | **T yes** | x | x | x | x | x | x | x |
| dependence-aware scheduling | x | n/a | P yes | P yes | **P yes** | P yes | n/a | n/a |
| distributed placement | x | P (manual) | ? | x | P yes | **P yes** | P (serving) | x |
| automatic schedule search | x | n/a (Inductor) | ? | P (some) | P yes | ? | n/a | n/a |

## What the matrix shows

Two cells stand out.

- **Static semantic verification** is Tensora's only clear "yes" today,
  and **no other system has it.** PyTorch/CoRa/TACO/Tiramisu/DISTAL all
  optimize from shapes/formats/schedules, not from a *verified* semantic
  contract. This is Tensora's real asset (v0.1 already does axis-name
  and contract checking; `demos/03`).
- **Multiple independent ragged dimensions** is a row where nearly
  everyone is weak: PyTorch jagged cannot represent it (tested);
  serving systems (vLLM/llama.cpp) are built around the single ragged
  sequence axis; the compilers (CoRa/TACO/Tiramisu/DISTAL) would need
  checking, but none advertises two independent ragged axes as a
  first-class representation. This is the candidate missing row
  (`pytorch_nested/ragged2d_spike.py` confirms the PyTorch cell).

Everything else is already covered by *someone*: ordinary ragged
attention (PyTorch jagged), ragged codegen (CoRa), CPU/edge + KV-cache
(llama.cpp, vLLM), scheduling (Tiramisu), distribution (DISTAL). So the
honest openings are the two cells above -- not "ragged in general."

## The routing thesis this suggests

No single backend fills every cell. That points at a specific role for
Tensora that does not compete with any of them:

> Tensora = one semantic front-end. Verify the contract (independence,
> reduction algebra, axis equivalence, ragged extents), then **route** to
> the backend that fits: TACO for format/algebra, Tiramisu for
> polyhedral scheduling, DISTAL for distribution, vLLM/llama.cpp for
> serving. The user writes once; Tensora picks the backend.

The verified contract is what makes routing safe (it proves the
transformation the backend will do is legal). The two-ragged-dim
representation is what none of the backends take as input yet -- so it
is both the missing row and the thing Tensora's IR would add.

## Update (P0 round 2): cells filled from docs, and a correction

Filled from authoritative docs/papers (web), not runs:

- **TACO** has a real CUDA backend (`-cuda`, `codegen_cuda.cpp`,
  GPUBlock/Warp/Thread). It generates **its own CUDA**, not cuBLAS/cuDNN.
  Its per-level formats (dense/compressed) can encode nested/hierarchical
  structure, so "multiple ragged dims" is arguably representable in
  TACO's model. Inputs: expression + level-formats + schedule -- **no
  verified-contract input.** (github.com/tensor-compiler/taco)
- **DISTAL** is **dense-only**, distributed (multi-CPU/GPU), built on a
  TACO fork. **No ragged.** Inputs: format + scheduling languages -- no
  verified-contract input. (arxiv 2203.08069)
- **Tiramisu** targets Nvidia GPU + MPI-distributed + FPGA, generates
  **Halide IR**, and takes its **own scheduling DSL, not MLIR**. Has a
  dense+sparse DL variant. No verified-contract input.
  (tiramisu-compiler.org)
- **TensorFlow RaggedTensor** already supports **multiple ragged
  dimensions** (docs), so multi-ragged is not unique to a new system.

**Correction to the earlier reading.** A real ISL experiment
(`tiramisu/islpy_ragged.py`) shows the polyhedral model **derives**
independence and reduction from the iteration domain itself (dependence
distances `(0,0,k)` for matmul). So "Tensora supplies independence /
reduction facts" is **not** a gap -- polyhedral compilers already get
those. And "multiple ragged dims" is representable in TF and arguably in
TACO's level formats. So neither is the clean unique cell I first
claimed.

**The one cell that survives, scoped.** *In the systems checked so far*
(TACO/Tiramisu/DISTAL/SpDISTAL/CoRa/PyTorch) none was found to take, as a
*verified, typed, propagated* input, the **data-dependent extent** carried
today as side metadata: `lengths`, `cu_seqlens`, `padding_mask`,
`position_ids`, `row_splits`. (This is a scoped observation, not a proof
of universal absence -- the compiler literature is large; a wider search
is part of the proposal's falsification step.) Classical polyhedral
analysis needs affine bounds; a ragged extent `L[b]` is a runtime array,
not affine, so it cannot be derived -- it must be *given*. Real code
carries it by hand and propagates it manually (wav2vec2:
`L_out = floor((L_in+2P-K)/S)+1` through convs; LLMs:
`position_ids = (mask.cumsum(-1)-1).clamp(min=0)`; attention: building
`cu_seqlens` from `seq_lens`).

Precise terminology: the extent *values* are runtime data; the *relation*
`L' = f(L)` is static. So the contribution is **compile-time verification
of runtime-dependent extent contracts**, not "static verification of
dynamic shapes." Shown concretely in `extent/extent_propagation.py`
(verify, catch a wrong transfer, quantify waste). See [PROPOSAL.md](PROPOSAL.md).

## Update (P0 round 4): prior-art bound + two corrections

A reviewer round added shape/type systems the earlier matrix missed.
These **narrow** the novelty; they do not kill it, but the claim must be
tightened.

| system | what it already does | so Tensora cannot claim |
|---|---|---|
| **Futhark** | compile-time size-dependent types over whole arrays (`[n][m] x [m][p] -> [n][p]`) | "typed extent relationships checked at compile time" (not novel) |
| **JAX / StableHLO** | symbolic dims + constraints; but docs separate this from *data-dependent dynamism* they do not represent | "symbolic shapes are new" |
| **MLIR ValueBounds + affine** | a runtime SSA value can be an affine loop bound (`affine.for %i=0 to %N`); islpy confirms `[N]->{[i]:0<=i<N}` | **"the backend cannot obtain L[b]"** (false) |
| **TensorFlow RaggedTensor** | multiple ragged dims | "multiple ragged dims is unique" |
| **2026 staged shape-dependent types** (arXiv 2604.23807) | compile-time tensor shape checking | "compile-time shape checking is new" |

**Correction 1 (the big one).** "The backend cannot obtain the extent" is
wrong -- MLIR consumes a runtime bound directly. So Tensora's contribution
is **not** a new loop representation. It is the high-level *provenance +
verification + propagation* of a per-instance extent through an ML graph;
the backend already accepts the result.

**Correction 2 (scope).** The `L'=f(L)` model does not cover all ML.
`extent/operator_taxonomy.py` classifies ops: A (unchanged), B (`f(L_in)`),
C (`f(L_in, metadata)`) -- covered; **D (data-dependent extent: nonzero,
boolean_mask, filter, MoE routing, dynamic decode)** -- NOT covered, the
extent is a runtime value of the data. Honest scope: a large sub-language.

**Correction 3 (two contracts).** `extent/contracts.py` shows the
relational contract `L'=f(L)` (static) does not imply representation
consistency (`L[b]` matches the buffer). A wrong length is silently
wrong. The system needs a runtime boundary check + static propagation --
"static verification" alone overclaims.

Net: the surviving, tightened claim is in [PROPOSAL.md](PROPOSAL.md), and
the decisive test is the A-vs-B MLIR lowering experiment (does the
verified contract change generated IR vs opaque `L[b]`?), which needs an
MLIR build not present here.

Still unrun (needs heavy installs / a cluster): head-to-head performance
vs vLLM/DISTAL/Tiramisu. The capability cells above are from docs; a
performance bake-off is future work.

## Update (P0 round 3): the ragged x distributed cell

Prompted by "what if a ragged tensor needs TACO codegen AND DISTAL
distribution?" No single system does ragged x distributed:

| | single-node | distributed |
|---|---|---|
| **dense** | TACO | DISTAL |
| **sparse** | TACO | SpDISTAL (SC 2022) |
| **ragged** | CoRa | **-- open --** |

SpDISTAL covers sparse+distributed but sparse != ragged; CoRa covers
ragged but single-node (it does within-node load balancing via thread
remapping). You cannot pipe TACO -> DISTAL: DISTAL and SpDISTAL are
*forks* that build distribution into TACO's core, not schedulers you bolt
on -- local codegen and distribution are co-designed.

The concrete gap is **load balancing by extent**
(`distal/ragged_balance.py`, measured). A dense distributor partitions by
index count; ragged attention work is ~L[b]^2, so count != work:

| case | by-count imbalance | by-work imbalance |
|---|---|---|
| B=256 mild | 1.04x | 1.00x |
| B=32 extreme | 1.71x | 1.02x |
| B=32 heavy-tail (5% at 512) | 4.59x | 4.26x |

- Work-aware partition needs the **verified per-sequence extent** --
  Tensora's one unique fact -- and fixes moderate skew (1.7-2x -> ~1.0x).
- Honest ceiling: a heavy tail (one sequence > per-worker mean) cannot be
  balanced by batch partition at all (4.6x -> 4.3x); that needs
  **intra-sequence splitting** (sequence parallelism). So ragged x
  distributed is two-level: work-aware batch distribution + intra-sequence
  split for the tail. This is a concrete, currently-open capability that
  uses exactly the verified-extent fact.
