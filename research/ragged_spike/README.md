# V0.2 spikes: is there a real gap for Tensora?

Go/no-go measurements for the V0.2 direction, with real numbers, before
building any tool. Each spike probes the *hypothesis* behind a system
(the systems themselves mostly need source builds / a cluster) to see if
it is cheaply pluggable into Tensora. Run on an NVIDIA RTX 500 Ada Laptop
GPU (4.0 GB) + 22 CPU cores, torch 2.13 + CUDA, WSL2.

**The P0 deliverable is [MATRIX.md](MATRIX.md)** -- the capability matrix
that locates the one surviving gap. The 1-page falsifiable framing is
[PROPOSAL.md](PROPOSAL.md); the runnable core demo is
[extent/extent_propagation.py](extent/extent_propagation.py). This README
is the detailed results log; [V0.3_PITCH.md](V0.3_PITCH.md) is the pitch.

**The surviving thesis (one line):** compile-time verification of
runtime-dependent extent contracts -- make `L[b]` a typed fact, verify
`L'=f(L)` per operator, propagate it. Not "ragged", not independence/
reduction (polyhedral derives those), not multi-ragged (TF/TACO have it).

**Positioning ([POSITIONING.md](POSITIONING.md)):** Tensora is an
*embedded ML kernel compiler* (the Futhark model) -- a small `.tns`
region compiled and called from a PyTorch app, not a PyTorch replacement.
Two separate goals: the *scientific* one (verified extents) and the
*system* one (embeddable kernel compiler). Futhark's `flatmap` (Aug 2026)
is the sharpest prior-art bound and is still not performant -- the space
is hard and open.

## Layout (organized by the system whose hypothesis each spike probes)

| dir | system / hypothesis | spikes |
|---|---|---|
| `pytorch_nested/` | PyTorch jagged: what does it cover? | `spike.py`, `fused_spike.py`, `coverage_spike.py`, `hetero_sweep.py`, `hetero_random.py`, `ragged2d_spike.py` |
| `distal/` | DISTAL: semantic fact → communication; ragged×distributed | `dist_spike.py` (measured), `shard_analysis.py` (model), `ragged_balance.py` (load balance) |
| `tiramisu/` | Tiramisu/polyhedral (real ISL via islpy) | `matmul_schedules.c`, `islpy_ragged.py`, `SETUP.md` |
| `taco/` | TACO: structure/format → less work | `taco_hypothesis.py` |
| `extent/` | **the surviving thesis**: verified extent propagation | `extent_propagation.py`, `operator_taxonomy.py` (scope: A/B/C vs D), `contracts.py` (two contracts + cu_seqlens), `affine_boundary.py` (ragged domain not affine) |
| `jax_stablehlo/` | JAX/StableHLO boundary (per-instance extent) | `jax_boundary.py` |
| `inductor/` | Inductor: is there single-node headroom? | `sched_spike.py`, `gemm_spike.py` |

## How to run (examples)

```bash
source .venv/bin/activate
python research/ragged_spike/pytorch_nested/spike.py        # ragged vs padded
python research/ragged_spike/pytorch_nested/ragged2d_spike.py  # two-ragged-dim gap
python research/ragged_spike/distal/dist_spike.py --procs 4  # measured all-reduce
```

## The four variants

- `eager_padded` — naive attention; builds the full [B,H,L,L] scores on
  padded data. Many hand-written attention blocks do this.
- `flash_padded` — `scaled_dot_product_attention` on padded data.
  PyTorch's own fused kernel; no full score matrix.
- `ragged_nested` — `scaled_dot_product_attention` on a jagged
  `NestedTensor`. **PyTorch's own ragged path.**
- `ragged_loop` — one SDPA call per sequence, no padding at all.

All four are checked for equal output on real (non-pad) positions
(max error ~1.5e-3, normal for fp16).

## Results

One config (batch=16, maxlen=2048, heads=8, head_dim=64, fp16). The
skewed batch had 5.86x more padded token slots than real tokens.

| variant | peak memory | latency |
|---|---|---|
| eager_padded | 3192 MB | 221.3 ms |
| flash_padded | 1188 MB | 27.5 ms |
| ragged_nested | 85 MB | 3.3 ms |
| ragged_loop | 53 MB | 0.8 ms |

Memory budget sweep (fail = peak exceeds the 4.0 GB VRAM), maxlen=2048:

| batch | eager_padded | flash_padded | ragged_nested | ragged_loop |
|---|---|---|---|---|
| 8 | 1603 MB | 595 MB | 44 MB | 23 MB |
| 16 | 3193 MB | 1177 MB | 63 MB | 31 MB |
| 32 | 6368 MB (over) | 2336 MB | 86 MB | 41 MB |
| 64 | — | 4667 MB (over) | 171 MB | 75 MB |

`eager_padded` exceeds 4 GB at batch 32. `flash_padded` exceeds it at
batch 64. Both ragged variants stay far under budget and never fail in
the sweep.

## Fused-region result (`fused_spike.py`)

The plain-attention result above left one hope: maybe PyTorch's ragged
path breaks on a whole custom fused region, and Tensora wins there. It
does not. Region tested: attention → RMSNorm → Linear(512,2048) → GELU →
Linear(2048,512) → residual, batch=16, maxlen=2048, 5.86x padding waste.

| variant | peak memory | latency |
|---|---|---|
| fused_padded | 1172 MB | 39.0 ms |
| fused_nested (PyTorch's ragged path) | 105 MB | 10.1 ms |
| fused_ragged_loop | 81 MB | 3.4 ms |

**PyTorch's `NestedTensor` carried the whole fused region** — ~11x less
memory than padded, correct output. It did not break.

An op-by-op probe of jagged `NestedTensor` shows how wide its support
now is:

| op | supported? |
|---|---|
| gelu, linear (elementwise / feature-dim) | yes |
| mean over feature dim (normalization) | yes |
| mean / sum / max over the ragged seq dim (pooling) | yes |
| softmax over the seq dim | yes |
| cumsum over the seq dim | no (`NotImplementedError`) |
| index a position, e.g. `x[:, 0]` | no (slice on jagged dim) |

Ragged pooling over the true length — the case we thought would be a
silent padding bug — is handled correctly by `NestedTensor` too. Only
exotic ops (cumsum over sequence, positional indexing) break.

## Coverage spike (`coverage_spike.py`) — where PyTorch's ragged path breaks

The critique was right that one fused region is not "mainstream
inference." This probes the untested cases. Empirical PyTorch column:

| case | GPU (torch.nested) | CPU (torch.nested) |
|---|---|---|
| causal self-attention | OK | BREAK (no viable backend) |
| cross-attention (Q len != KV len) | OK | BREAK (no viable backend) |
| **KV-cache decode step** | **BREAK** (Q,K,V must be ragged on same dim) | BREAK |
| **grouped/multi-query (GQA/MQA)** | **BREAK** (not implemented for jagged) | BREAK |
| two ragged dimensions | BREAK | BREAK |
| autograd (backward) | OK | BREAK (no viable backend) |
| torch.compile(fullgraph) | OK | BREAK |

This changes the earlier conclusion. `torch.nested` covers dense-batch
self/cross attention on GPU (plus autograd and compile), but **breaks on
the inference-serving path — KV-cache decode and GQA/MQA — and has no
viable CPU backend at all.** KV-cache and GQA are not exotic: they are
how essentially every modern LLM serves tokens. So "PyTorch captures
ragged" is false for the cases that matter most for inference.

Honest caveat: these exact gaps are already targeted by specialized
systems (vLLM / PagedAttention for KV-cache, llama.cpp for CPU/edge). So
the gap is in *core PyTorch*, not in the ecosystem. The open research
question is whether a semantic compiler adds value over those point
solutions — not whether core PyTorch has the gap (it does).

**Correctness (the reframed question).** Mean-pooling over the sequence:
the naive padded result (divides by max length) differs from the
true-length result by up to 0.647 on the test batch. The padded answer
is silently WRONG unless the ragged extent is known. The semantic fact
(true length per row) is exactly what makes it correct. This is a
runnable demonstration that making the extent explicit changes the
*result*, not just speed.

## Two-ragged-dim spike (`ragged2d_spike.py`) — the candidate missing row

`torch.jagged` supports one ragged dim. Workloads with two independent
ragged axes (documents→sentences→tokens; ragged×ragged pairwise ops)
cannot be represented, so both axes get padded, and the waste is
quadratic in the skew.

- Building a tensor with two independent ragged dims: **BREAK**
  (`Cannot represent given tensor list as a nested tensor`).
- Ragged×ragged pairwise `C[b,i,j]=sum_d A[b,i,d]B[b,j,d]`, B=16,
  varying `L_A`,`L_B`: padded 0.50 GMAC vs true 0.16 GMAC = **3.1×
  wasted work**; the `[B,jA,jB]` output has two ragged dims, so no
  core-PyTorch ragged type can hold it — padded is the only vectorized
  option.

This is a **representation gap, not a missing kernel** — the strongest
candidate for a real V0.3 target (see [MATRIX.md](MATRIX.md)).

## Randomized heterogeneity (`hetero_random.py`) — rigorous version

Many random batches per config, several batch sizes and max lengths,
three skew levels, mean ± std. Answers "is a rigorous sweep hard?": no,
except multi-GPU (one GPU here).

| batch | maxlen | skew | work elim % | ragged speedup |
|---|---|---|---|---|
| 8 | 1024 | 0.20 | 16 ± 4 | 3.6 ± 0.5 |
| 8 | 1024 | 0.95 | 57 ± 11 | 6.0 ± 1.2 |
| 32 | 1024 | 0.20 | 19 ± 1 | 4.4 ± 0.6 |
| 32 | 1024 | 0.95 | 63 ± 5 | 7.7 ± 1.3 |

At maxlen=1024 the ragged speedup rises with skew (3.6→6.0, 4.4→7.7×),
tracking work-eliminated with tight confidence intervals. (At maxlen=256
launch overhead muddies it.)

## Conclusion (honest go/no-go)

**The ragged opportunity is real and large.** On a skewed batch, ragged
attention used about **14x less memory** and ran about **8x faster**
than PyTorch's own good fused kernel on padded data. On a 4 GB GPU,
padded attention runs out of memory at a batch where ragged still fits
easily.

**But PyTorch already captures this for plain attention.**
`ragged_nested` is not Tensora — it is core `torch.nested` plus SDPA. A
user gets the 85 MB / 3.3 ms result today, with a few lines and no
Tensora. So **for a single attention op, Tensora adds nothing** over
what PyTorch already ships. This confirms the risk raised in the
roadmap (§9.3b, §9.7): the value is not "Tensora can do ragged
attention."

**For the one region tested, the custom-fused-region hope did not
hold.** `NestedTensor` handled that specific region — attention,
RMSNorm, FFN, and ragged pooling — with ~10x memory savings and no
Tensora. This does **not** mean "PyTorch captures ragged in general."
It means: *for this mainstream Transformer inference region*, jagged
`NestedTensor` already captures the principal padding-avoidance benefit.
The coverage spike below then tested those untested cases and found real
breaks: KV-cache decode, GQA/MQA, and all CPU nested attention. So the
scoped claim, not the broad one, is what the evidence supports.

## Verdict

- **Ragged is a real, large win** (~10-14x memory, ~8x speed on skewed
  batches; padded OOMs on a 4 GB GPU where ragged fits).
- **Core `torch.nested` delivers that win for dense-batch self/cross
  attention on GPU** (plus autograd and compile). A Tensora pass adds
  nothing there.
- **But core PyTorch does NOT cover the inference-serving path.** The
  coverage spike found KV-cache decode and GQA/MQA break on GPU nested,
  and CPU nested attention has no viable backend at all. These are the
  cases that matter most for real LLM inference.
- **Those gaps are real in core PyTorch, but already filled by
  specialized systems** (vLLM / PagedAttention for KV-cache, llama.cpp
  for CPU/edge). So the open question is whether a semantic compiler adds
  value over those point solutions — not whether the core-PyTorch gap
  exists (it does).
- **Verification is a distinct, runnable value.** The correctness spike
  showed a padded mean-pool is silently wrong (by 0.647) unless the
  ragged extent is known. This is a property PyTorch does not check. But
  see the caveat: once a user is inside `NestedTensor`, the abstraction
  already enforces it, so the verification thesis needs real workloads
  where the extent is manual and easy to get wrong (hole 5 in the future-
  directions doc).

## Scheduling-headroom result (`sched_spike.py`)

The future research directions (semantic axis facts feeding
TACO/Tiramisu/DISTAL/Inductor) rest on one claim: explicit semantic
facts let a compiler beat what it infers from shapes alone. This spike
tests the single-node version of that claim: how much headroom does
PyTorch's own compiler (Inductor, via `torch.compile`) leave?

Method: measure a device-copy bandwidth ceiling, then report each
kernel's achieved bandwidth as a percent of it. A pointwise op cannot
beat moving its bytes, so "near the ceiling" means "near optimal."

| kernel | variant | latency | GB/s | % of ceiling |
|---|---|---|---|---|
| pointwise chain | eager | 18.9 ms | 28 | 26% |
| pointwise chain | **compile** | 4.9 ms | 110 | **100%** |
| rmsnorm (reduction) | eager | 16.3 ms | 33 | 30% |
| rmsnorm (reduction) | **compile** | 5.5 ms | 98 | **89%** |
| norm+pointwise fused | eager | 30.4 ms | 18 | 16% |
| norm+pointwise fused | **compile** | 5.0 ms | 108 | **98%** |

(ceiling = 110 GB/s device-to-device copy on this GPU.)

`torch.compile` reaches 89-100% of the copy-bandwidth ceiling. That
ceiling bounds these bandwidth-bound pointwise/reduction/fused kernels;
it is **not** a proof of global optimality (compute-bound kernels,
occupancy, arithmetic intensity, and algorithm selection are not
captured by it). So the honest claim is narrower: **there is little
obvious single-node headroom for a semantic scheduler on the tested
bandwidth-bound kernels.** The 4-6x eager→compile speedup is Inductor's
win, free to any user, not Tensora's.

## Heterogeneity sweep (`hetero_sweep.py`)

Varies only the length distribution, at a **fixed** maxlen of 128 (an
earlier version also grew maxlen across cases, which changed absolute
work and was not a clean causal test). The theoretical
**work-eliminated** column is the clean signal; wall-time is noisy at
these small sizes, so it is not a curve.

| case | pad ratio | attention work eliminated |
|---|---|---|
| A uniform [128,128,128,128] | 1.00x | 0% |
| B mild [128,120,112,104] | 1.10x | 17% |
| C moderate [128,96,64,32] | 1.60x | 53% |
| D strong [128,64,32,8] | 2.21x | 67% |
| E extreme [128,32,16,8] | 2.78x | 73% |

At fixed maxlen, the eliminated work rises monotonically with skew
(0% → 73%). So ragged only helps in the heterogeneous regime. A fuller
study would sample many batches from length distributions, not five
hand-picked cases.

## Sharding analysis (`shard_analysis.py`, analytical)

The distributed direction (DISTAL) needs a cluster to run, but its key
point shows on any machine. For `Y = X @ W`, `X[batch, seq, hidden]`,
splitting across P devices:

- shard **batch** (an independent axis): 0 bytes of communication.
- shard **hidden** (the reduction axis): an all-reduce of the output,
  hundreds of MB per step (537 MB at P=2, ~1 GB at P=32 for a 32x2048x4096
  workload).

Both do the same local compute. The semantic facts
(`batch=independent`, `hidden=reduction`) decide the communication cost
up front; a shape-only IR must rediscover that batch carries no
cross-device dependence.

## Distributed spike (`dist_spike.py`, MEASURED, not modeled)

`shard_analysis.py` above only modeled communication. This measures it,
with real `torch.distributed` (gloo) collectives across processes on the
22 CPU cores -- co-located, but a real all-reduce. Workload
`X[8,256,2048] @ W[2048,2048]`, P=4 ranks (5 threads each):

| sharding | axis kind | per-step | communication |
|---|---|---|---|
| shard batch | independent | 35.9 ms | none |
| shard hidden | reduction | 46.8 ms | all-reduce (17 MB) |
| (all-reduce only) | | 11.1 ms | |

Sharding the reduction axis costs a measured **~11 ms/step more** (the
real all-reduce), which sharding the independent axis avoids entirely.
Correctness verified (max err 9.2e-5). The semantic fact predicts this
before running. This corrects the earlier "analytical only, needs a
cluster" claim: the hypothesis IS testable here.

Honest caveats: gloo over loopback is far faster than a real network, so
the ~11 ms is a **lower bound** on a real cluster (the real gap is
larger). It is CPU multi-process, not multi-node, and it does not test
DISTAL the system -- only the hypothesis DISTAL is built on. The
heterogeneous-GPU angle is not reachable: this torch build has no XPU
backend for the non-NVIDIA GPU, and only one CUDA device.

Scaling (all-reduce cost vs process count): 8.4 ms at P=2, 11.1 ms at
P=4, 23.9 ms at P=8. The reduction-axis communication penalty grows with
scale; the independent-axis sharding stays at zero communication
throughout. So the semantic fact matters more, not less, as you add
devices.

The general lesson (from a reviewer challenge): distinguish "cannot run
system X" from "cannot test the hypothesis behind X." Most hypotheses
have a real local proxy. Distributed did.

Applying that lesson to every "cannot test" claim, and following through:

- **Distributed / DISTAL hypothesis:** MEASURED (`dist_spike.py`).
- **TACO hypothesis:** TESTED (`taco_hypothesis.py`) with `torch.sparse`.
  (The PyPI `pytaco` is an unrelated package; real TACO needs a C++
  source build, so this is the hypothesis, not TACO's codegen.)
- **Tiramisu hypothesis:** SIMULATED (`matmul_schedules.c`) with native
  C loops -- schedule choice gives a 30x swing. (Not Tiramisu itself.)
- **"Inductor near-optimal":** now shown for compute-bound GEMM too
  (`gemm_spike.py`), not just bandwidth-bound. cuBLAS is the ceiling.
- **Heterogeneous GPU** (the non-NVIDIA GPU): genuinely blocked -- this
  torch build has no XPU/ROCm backend for it, verified. A real hard
  limit, not a lazy dismissal.

Only the heterogeneous-GPU case remains untestable here; every other
dismissed hypothesis was tested with a real local proxy.

## Testing the other systems' hypotheses (not the systems)

Same lesson applied to TACO, Tiramisu, and the compute-bound Inductor
question: test the hypothesis with a real local proxy.

**TACO's hypothesis** (`taco_hypothesis.py`) — structure/format exposed
to the compiler makes work track nonzeros, not dense size. Real
`torch.sparse` CSR @ dense vs dense @ dense, `M[4096,4096] @ X[4096,512]`:

| density | dense | sparse | speedup | dense MB | sparse MB |
|---|---|---|---|---|---|
| 0.50 | 5.9 ms | 32.9 ms | 0.2x | 67 | 67 |
| 0.10 | 5.2 ms | 6.2 ms | 0.8x | 67 | 13 |
| 0.02 | 5.3 ms | 1.5 ms | 3.6x | 67 | 3 |
| 0.005 | 5.4 ms | 0.6 ms | 8.6x | 67 | 1 |

Confirmed with an honest crossover: below ~10% density the format-aware
kernel wins (8.6x faster, 67x less memory at 0.5%); above it, dense
wins. Structure is optimization information. (Proxy only: this is
`torch.sparse`, not TACO's codegen, and sparsity != raggedness.)

**Tiramisu's hypothesis** (`matmul_schedules.c`, native C via gcc -O2) —
the same computation under different loop schedules has very different
speed. `N=1024` matmul:

| schedule | GFLOP/s |
|---|---|
| naive ijk | 0.83 |
| interchange ikj | 25.5 |
| tiled ikj | 16.0 |

Schedule choice alone gives a **30x** swing. The reorder is legal only
because k is a reduction axis — the exact semantic fact a compiler needs.
This is Tiramisu's premise, shown with real native loops (not Tiramisu).

**Polyhedral, with the real ISL library** (`tiramisu/islpy_ragged.py`,
`islpy` — the engine Tiramisu/PolyDL/Pluto are built on; see `SETUP.md`):

- ISL represents the **ragged** iteration domain exactly: 16,747 true
  points vs 73,728 padded box (4.4x). A padded tensor shape cannot
  express this; a polyhedral set can.
- ISL **derives** parallelism: matmul dependence distances are `(0,0,k)`
  — i,j parallel, only k (reduction) carries a dependence.

Important correction from this: the polyhedral model already **derives**
independence/reduction, so those are **not** what Tensora uniquely adds.
The gap is the **data-dependent extent** (`L[b]`), which is non-affine
and cannot be derived — it must be given as a verified fact. See
[MATRIX.md](MATRIX.md) (P0 round 2) and [V0.3_PITCH.md](V0.3_PITCH.md).

**Compute-bound Inductor challenge** (`gemm_spike.py`) — the earlier
scheduling spike only tested bandwidth-bound kernels. On a compute-bound
square GEMM (fp16, tensor cores), `torch.compile` does **not** beat plain
`torch.matmul`:

| M=N=K | eager | compile |
|---|---|---|
| 2048 | 27.7 TFLOP/s | 27.1 |
| 4096 | 22.1 | 21.8 |
| 8192 | 20.7 | 18.3 |

Both call cuBLAS; ~20-28 TFLOP/s is a high fraction of this laptop GPU's
fp16 roofline. So cuBLAS is the practical ceiling on the compute-bound
kernel too -- no headroom for a semantic scheduler there either. This
closes the earlier gap: "near-optimal" now holds on both bandwidth-bound
and compute-bound kernels (practical ceiling, not a theoretical proof).

## Overall synthesis across all spikes

- **Ragged, GPU dense-batch attention (spike.py, fused_spike.py):** real
  win, already delivered by `torch.nested`. No niche for Tensora.
- **Ragged, inference-serving path (coverage_spike.py):** core PyTorch
  BREAKS on KV-cache decode, GQA/MQA, and all CPU nested attention.
  These are the cases real LLM inference uses. Real gap in core PyTorch,
  but already filled by vLLM / llama.cpp.
- **Heterogeneity (hetero_sweep.py):** the ragged gain exists only when
  the batch is uneven (0% work eliminated when uniform, ~73% when
  extreme, at fixed maxlen). Narrow regime.
- **Scheduling (sched_spike.py, gemm_spike.py):** Inductor reaches
  89-100% of the copy-bandwidth ceiling on bandwidth-bound kernels, and
  does not beat cuBLAS on a compute-bound GEMM. Little single-node
  headroom on either (practical ceiling, not a theoretical proof).
- **Structure/schedule as information (taco_hypothesis.py,
  matmul_schedules.c):** confirmed in principle -- sparse format gives
  up to 8.6x when sparse enough; loop-schedule choice gives a 30x swing
  on the same matmul. These validate the general thesis (structure and
  schedule are optimization information), independent of ragged.
- **Sharding, MEASURED (dist_spike.py):** with real gloo collectives on
  the 22 cores, sharding the reduction axis cost ~11 ms/step of real
  all-reduce that sharding the independent axis avoided. The semantic
  fact predicts it. This is a lower bound vs a real network, CPU-only,
  and does not test DISTAL the system -- but the hypothesis is measured,
  not modeled.
- **Therefore: core PyTorch already covers GPU dense-batch ragged and is
  near the bandwidth limit single-node. The remaining gaps (KV-cache,
  GQA, CPU/edge, distributed) are real but already occupied by
  specialized systems. No open-source "faster than PyTorch" quick win;
  the open questions are all research, versus specialized baselines.**

What survives, honestly:

1. **Distributed (DISTAL direction).** The one un-dominated direction:
   independence / reduction-algebra / axis-equivalence facts feeding a
   distributed mapper. The hypothesis is now MEASURED locally
   (`dist_spike.py`): a semantic fact predicts a real all-reduce cost.
   A full result still needs a real cluster and a comparison against
   DISTAL, but it is no longer untestable. Research collaboration, not
   an open-source quick win.
2. **Verification.** Static named-axis / contract checking that PyTorch
   does not do. Not undermined by any spike here (the spikes disprove a
   *speed* niche, not a *correctness* one). This is the most credible
   open-source value, but it is a linter/checker, not a speedup.
3. **The inference-serving gaps in core PyTorch** (KV-cache decode,
   GQA/MQA, CPU/edge) -- real and confirmed by the coverage spike, but
   already served by vLLM / llama.cpp. Any pitch must be "versus those
   specialized systems," not "versus core PyTorch."

**Implication for open-sourcing.** No "faster than PyTorch" quick win is
supported: core PyTorch covers GPU dense-batch ragged, and the gaps it
does have are already filled by specialized systems. The honest
outcomes are: (a) open-source v0.1 as an academic teaching compiler;
and/or (b) pursue the distributed / verification / inference-serving
directions as research with the professor, measured against the
specialized baselines. Do not open-source on a "faster than PyTorch"
claim.
