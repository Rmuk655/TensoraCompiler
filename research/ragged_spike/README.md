# Ragged-attention measurement spike

Go/no-go measurement for the V0.2 direction. It asks one question with
real numbers, before any tool is built: **is there a real memory and
speed gap between padded and ragged attention on a skewed batch, and
does PyTorch already capture it?**

Uses core PyTorch only (no `flash_attn`, no `vllm`), so it measures what
a stock user actually has. Run on an NVIDIA RTX 500 Ada Laptop GPU
(4.0 GB), torch 2.13 + CUDA, WSL2.

## How to run

```bash
source .venv/bin/activate
python research/ragged_spike/spike.py            # one config, four variants
python research/ragged_spike/spike.py --sweep    # grow batch to the memory budget
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

**The custom-fused-region hope did not hold.** `NestedTensor` handles
the whole mainstream transformer inference path — attention,
normalization, FFN, and ragged pooling — with ~10x memory savings and
no Tensora. So a "Tensora ragged pass that beats PyTorch on mainstream
inference" is not supported by the evidence. The only gaps are exotic
ops (`cumsum` over sequence, positional indexing on the jagged dim),
which are not where mainstream inference memory pain lives.

## Verdict

- **Ragged is a real, large win** (~10-14x memory, ~8x speed on skewed
  batches; padded OOMs on a 4 GB GPU where ragged fits).
- **PyTorch already delivers that win** for the mainstream path, through
  core `torch.nested`. A Tensora pass adds nothing there.
- **So the pragmatic "beat PyTorch on ragged inference perf" thesis for
  V0.2 is not supported.** Pursue it only for the narrow set of ops
  `NestedTensor` does not cover, or on targets where its kernels are
  weak (for example CPU/edge, or generated GPU kernels via polyhedral
  scheduling) — which is research, not a quick adoption win. That
  research framing is the V0.3 pitch
  (`research/ragged_spike/V0.3_PITCH.md`).
- **The un-dominated real-world value is verification** (catching
  wrong-axis / contract bugs PyTorch does not check), not speed.

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

`torch.compile` reaches 89-100% of the bandwidth ceiling. **Inductor is
already at the single-node hardware limit** for pointwise, reduction,
and fused kernels. A semantic scheduler could not beat it there; it
could only match it. The 4-6x eager→compile speedup is Inductor's win,
free to any user, not Tensora's.

## Heterogeneity sweep (`hetero_sweep.py`)

Turns the ragged result into a causal curve: the gain tracks how uneven
the batch is. The **compute-eliminated** column is the clean signal;
wall-times here are sub-millisecond and noisy (launch overhead dominates
at these tiny sizes, so the uniform case shows a misleading speedup).

| case | pad ratio | attention work eliminated |
|---|---|---|
| A uniform [128,128,128,128] | 1.00x | 0% |
| B one short [100,100,100,20] | 1.25x | 24% |
| C graded [128,64,32,8] | 2.21x | 67% |
| D very skewed [512,64,16,4] | 3.44x | 75% |

A uniform batch eliminates no work; a very skewed batch eliminates ~75%.
So ragged only helps in the heterogeneous regime -- the honest "when
does it help" boundary.

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
cross-device dependence. This is analytical only -- a real speedup claim
needs a real cluster and backend, which is future research.

## Overall synthesis across all spikes

- **Ragged (spike.py, fused_spike.py):** real win, already delivered by
  `torch.nested`. No single-node niche for Tensora.
- **Heterogeneity (hetero_sweep.py):** the ragged gain exists only when
  the batch is uneven (0% work eliminated when uniform, ~75% when very
  skewed). Narrow regime.
- **Scheduling (sched_spike.py):** Inductor is already at the bandwidth
  limit single-node. No headroom for a semantic scheduler on one GPU.
- **Sharding (shard_analysis.py):** the one place a semantic fact clearly
  changes the answer (communication-free vs hundreds of MB) -- but it is
  distributed, so it needs a cluster and is research, not a laptop win.
- **Therefore, on a single node, there is no runtime perf/memory win
  available to open source that PyTorch does not already deliver.**

What survives, honestly:

1. **Distributed (DISTAL direction).** The one un-dominated direction:
   independence / reduction-algebra / axis-equivalence facts feeding a
   distributed mapper. Not testable on one laptop GPU (needs a cluster).
   This is a research collaboration, not an open-source quick win.
2. **Verification.** Static named-axis / contract checking that PyTorch
   does not do. Not undermined by any spike here (the spikes disprove a
   *speed* niche, not a *correctness* one). This is the most credible
   open-source value, but it is a linter/checker, not a speedup.
3. **Ops PyTorch's ragged path lacks, and weak-kernel targets** (CPU /
   edge / generated kernels) -- research, per `V0.3_PITCH.md`.

**Implication for open-sourcing.** The perf-benefit case for open
sourcing is not supported by the evidence. The honest outcomes are:
(a) open-source v0.1 as an academic teaching compiler; and/or (b) pursue
the distributed and verification directions as research with the
professor. Do not open-source on a "faster than PyTorch" claim.
