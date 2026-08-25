# Future Research Directions

For a research collaboration (for example with Prof. R. Upadrasta).
These are post-V0.2. V0.2's spikes (see
[14_V0.2_CLOSEOUT.md](14_V0.2_CLOSEOUT.md)) showed there is no
single-node runtime win to open-source that PyTorch does not already
give. So these directions are framed as research, not as adoption
claims.

## 0. The reframed research question (read first)

A review of the V0.2 spikes found that they establish that raggedness
can matter, but they do **not** yet establish that Tensora has a
research contribution. The directions below are **candidates to
evaluate**, not commitments. Three corrections shape all of them:

- **Do not pitch "Tensora does ragged" or "Tensora is faster than
  PyTorch on ragged."** The spikes push against both. CoRa already
  occupies "ragged compiler + generated kernels." PyTorch's jagged
  `NestedTensor` already captures the tested Transformer region.
- **The honest question is:** *is there a semantic property Tensora can
  represent and verify that existing tensor compilers (TACO, Tiramisu,
  DISTAL) do not make explicit, and does making it explicit measurably
  change optimization, distribution, or correctness?* Raggedness becomes
  one test case, not the whole thesis. The general categories to probe
  are **axis independence**, **axis equivalence**, and **reduction
  algebra**.
- **Do not build the sub-polyhedral ragged scheduler yet.** First do the
  capability matrix in §0.1, then pick the first workload with a genuine
  gap — rather than retrofitting a scheduler onto the one gap
  (`cumsum`/indexing) that happened to appear in the probe.

### 0.1 First step: a capability matrix, before any building

**The full matrix is built:** `research/ragged_spike/MATRIX.md` (on the
`v0.3-research` branch).
It found two candidate missing rows: **static semantic verification**
(Tensora's only "yes", no other system has it) and **multiple
independent ragged dimensions** (PyTorch jagged cannot represent it --
tested; serving systems are single-ragged-axis). The small table below is
the empirical PyTorch column from
`research/ragged_spike/pytorch_nested/coverage_spike.py` (GPU, torch
2.13); the other systems' cells in the full matrix are from papers, not
runs.

| workload | PyTorch nested (GPU) | PyTorch nested (CPU) | CoRa | TACO | Tiramisu | DISTAL |
|---|---|---|---|---|---|---|
| self / causal attention | OK | breaks (no backend) | ? | ? | ? | ? |
| cross-attention (Q!=KV len) | OK | breaks | ? | ? | ? | ? |
| KV-cache decode | **breaks** | breaks | ? | ? | ? | ? |
| GQA / MQA | **breaks** | breaks | ? | ? | ? | ? |
| two ragged dims | breaks | breaks | ? | ? | ? | ? |
| autograd | OK | breaks | ? | ? | ? | ? |
| torch.compile | OK | breaks | ? | ? | ? | ? |

The empirical finding: core PyTorch's ragged path covers GPU dense-batch
attention but breaks on the inference-serving path (KV-cache, GQA/MQA)
and on CPU entirely. Those gaps are already served by specialized
systems (vLLM, llama.cpp), so a research pitch must compare against
those, not core PyTorch.

Extend this with more real workloads (10-20). For each, record: what the
programmer must specify manually, what can go wrong, what information is
implicit, what optimization is hidden. Only a cell where every existing
system (including the specialized ones) is weak is a real research
opening.

### 0.2 The five holes to close before claiming a contribution

1. **No representation experiment.** Tensora's TCG does not yet carry
   substantially more than shapes + op types. Show the extra facts
   (independence, reduction algebra, ragged extent, axis identity) exist
   in the IR and that an optimizer consumes them — not as comments.
2. **No competing implementation.** Compare what each of Tensora, TACO,
   Tiramisu, DISTAL, and PyTorch/Inductor can *express* before claiming a
   missing capability.
3. **No generated schedule.** Show the same computation, two schedules,
   one chosen using semantic information, and the performance difference.
4. **No difficult benchmark.** Attention is easy to dismiss (PyTorch
   invested there). Find a ragged workload with an op `NestedTensor`
   does not handle cleanly AND a meaningful iteration-space
   optimization — and check it is actually important and expensive.
5. **No user evidence.** Collect real ragged kernels and show what is
   hard to express or easy to get wrong, before claiming verification is
   valuable.

### 0.3 raggedness is not sparsity (precise boundary)

A ragged tensor is dense within each object but has data-dependent
extents. It is not necessarily sparse in the TACO sense (no zero-valued
structure). So "TACO supports sparse, therefore TACO solves ragged" is
wrong, and so is the reverse. The precise research boundary is
**irregular iteration spaces that are dense within each object but have
data-dependent bounds.**

## The core thesis

One idea unifies all of these: **semantic axis facts, statically
verified and preserved into the IR, are optimization information.** Not
the names — the facts. For example:

```
batch  = independent          (no cross-batch dependence)
head   = independent
model  = reduction<sum>       (associative reduction)
Q.batch == K.batch == V.batch (axis identity across tensors)
head   : shardable
model  : do not shard         (it is a reduction)
```

The claim to test: these facts let a downstream compiler prove
parallelism, prove distribution legality, cut its search space, or
reject illegal schedules — things a shape-only IR must rediscover.

The framing, in order (do not lead with "beat PyTorch"):

```
more semantic knowledge
   -> less ambiguity
   -> fewer legal schedules / less search
   -> less unnecessary computation
   -> potentially better runtime
```

Use PyTorch, CoRa, and TACO as reference points, not as the thing to
beat everywhere. If Tensora beats PyTorch in one narrow important regime
(highly heterogeneous sequence lengths), that is the headline. If not,
"semantic facts reduce search or computation" is still a real result.

## Direction 1 — Semantic-guided distributed execution (strongest)

The one direction V0.2's spikes could not dominate. Two spikes support
it. `shard_analysis.py` modeled the communication; `dist_spike.py` then
**measured** it, with real `torch.distributed` (gloo) collectives across
the 22 CPU cores: sharding the independent `batch` axis needed zero
communication; sharding the `hidden` reduction axis cost a real
~11 ms/step all-reduce. The semantic fact predicts this before running.
Feed these facts (independence, reduction algebra, axis identity,
distribution constraints) to a distributed mapper.

- Best fit: **DISTAL** (distributed tensor algebra; it already separates
  data distribution from computation scheduling). Tensora would supply
  a verified higher-level constraint system; DISTAL derives the machine
  mapping.
- The hypothesis is already measured locally (co-located gloo, a lower
  bound vs a real network). The remaining steps: scale the process
  count, run on a real multi-node cluster, and compare against DISTAL's
  own mapping -- that is where the real speedup claim is settled.

## Direction 2 — Semantic-guided polyhedral scheduling

Feed axis roles to a polyhedral scheduler as search constraints and
priorities, rather than letting it rediscover them.

- Best fit: **Tiramisu** (polyhedral, multi-target). Tensora supplies
  "batch → parallel candidate, height/width → tile candidate, channel →
  vector/reduction candidate"; Tiramisu does the legal transformation
  and schedule search.
- The professor's **sub-polyhedral (UTVPI) scheduling** (POPL 2013;
  IMPACT 2011/2012) is the technique for keeping that search fast.
- Ragged/variable-length scheduling connects here; see
  `research/ragged_spike/V0.3_PITCH.md` (on the `v0.3-research` branch).

## Direction 3 — Verification as the product

Un-dominated by any V0.2 spike, because the spikes disprove a speed
niche, not a correctness one. Static named-axis / contract checking that
PyTorch does not do: catch wrong-axis contractions, computing over
padding, illegal fusion of same-shape-different-meaning tensors. V0.1's
semantic analyzer already demonstrates the base (`demos/03`). A
`fusion-legality` experiment: two tensors with the same shape but
different logical axes (`A[batch,hidden]` vs `B[hidden,batch]`) — Tensora
rejects the illegal transform statically; a shape-only view cannot.

## Direction 4 — Quantization-axis semantics

The same tensor shape does not fix legal or efficient low-level
transforms. `W[out,in]` with `quantization_axis = out` vs `= in` are
both valid shapes but mean different per-channel quantization. Test
whether Tensora can carry `out = quantization axis` through
matmul → bias → activation → requantize without losing the
correspondence. A second, non-ragged domain for the same thesis.

## Direction 5 — Semantics-preserving multi-target ML compiler

Evolve Tensora from a language compiler into a model-to-deployment
compiler with one semantic IR and many importers/backends:

```
PyTorch / ONNX / TF  ->  Tensora Semantic IR  ->  CPU / GPU / Web / MCU / Apple
```

Do not pitch this as "convert models to every platform" — that is
**IREE** and **TVM** already. The research novelty must be specific: the
IR preserves axis identity, quantization contracts, layout, precision,
sparsity, raggedness, and distribution as explicit facts, reused by
target backends instead of rediscovered. Paper question: does preserving
these facts across the high-level-to-low-level boundary cut compiler
search cost or improve generated code across heterogeneous targets,
versus a conventional importer pipeline?

## The recommended single experiment family

Rather than six unrelated experiments, one benchmark family: **attention
with increasing semantic information.** Measure what each layer buys.

| version | new information | expected effect |
|---|---|---|
| plain shape | none | baseline |
| named axes | axis identity | better verification |
| axis roles | reduction / independence | fewer legal-schedule candidates |
| cross-tensor identity | axis equivalence | safer fusion / distribution |
| ragged | variable extent | less actual computation |

Metrics beyond runtime: transformations considered, transformations
rejected, dependence analyses performed, IR rewrites, compile time,
communication volume, kernel launches, memory traffic. These test the
semantic-IR thesis more cleanly than trying to beat a GEMM kernel.

## The question each existing system forces (answer crisply before pitching)

A compiler professor will ask exactly these. The pitch must answer each,
not restate the system:

- **TACO** already separates *what is computed* from *how tensors are
  stored*, over index notation, and generates the loops. Question: *what
  does a semantic-axis layer add to TACO?* A ragged extent is an
  iteration-bound property, not a storage-format property — that is a
  candidate answer, but it must be shown, not asserted.
- **Tiramisu** already separates *computation* from *schedule* in the
  polyhedral model. Question: *"semantic facts → schedule" is the world
  Tiramisu already lives in — what specifically is missing?* The novelty
  cannot be the arrow itself.
- **DISTAL** already separates *computation* from *distribution and
  communication*. Question: *is there semantic information about
  irregular/ragged iteration spaces that DISTAL cannot represent
  conveniently or exploit correctly?* The `shard_analysis.py` insight is
  what DISTAL was built to operationalize, so "semantic facts help
  sharding" is far too broad.

## Prior art to cite (do not reinvent)

- **CoRa** — ragged tensor compilation with minimal padding, MLSys 2022.
  1.6x over PyTorch (Transformer encoder, V100), 1.37x over TensorFlow
  (MHA, 64-core ARM). Extends tensor-compiler abstractions to understand
  dimension relationships and variable loop bounds.
  https://www.pdl.cmu.edu/PDL-FTP/BigLearning/MLSys-2022-the-cora-tensor-compiler.pdf
- **TACO** — the Tensor Algebra Compiler, OOPSLA 2017. Compiles
  dense/sparse tensor algebra from index notation; index roles (free vs
  contraction) drive the generated loops. http://tensor-compiler.org
- **DISTAL** — distributed tensor algebra; explicit mapping of tensor
  dimensions to machine dimensions, with scheduling for tiling,
  distribution, and communication.
- **Tiramisu** — polyhedral compiler; loop and data-layout transforms
  for CPU/GPU/FPGA/distributed.
- **Sub-polyhedral scheduling** — Upadrasta & Cohen, POPL 2013; IMPACT
  2011/2012. Fast (UTVPI) scheduling.
- **PolyDL** — polyhedral optimizations for DL primitives.
- **IREE, TVM** — the existing multi-target ML compiler baselines for
  Direction 5.
- **"A Faster Generalized Two-Stage Approximate Top-K"** (TMLR 2026) —
  an example where axis roles (reduction axis vs top-k axis) decide
  fusion and which accelerator subsystem is the bottleneck. Supports the
  "axis semantics as optimization information" thesis.

## Appendix — what a polyhedral compiler does (reference)

For `for i,j: C[i,j] = A[i,j] + B[i,j]`, the compiler treats each run as
a point `(i,j)` in the iteration space and transforms it while keeping
dependencies legal:

- **Tiling** — split into blocks (e.g. 32x32) so a small region stays in
  cache/SRAM.
- **Interchange** — swap loop order (`i,j` -> `j,i`) when dependencies
  allow, to match memory layout.
- **Fusion** — merge two loops over the same space so an intermediate is
  consumed where produced, avoiding materializing it.
- **Parallelization** — run independent points on different threads
  (legal here; illegal for `A[i]=A[i-1]+1`).
- **Distribution** — split points across machines; legal with no
  cross-partition dependence, else the compiler must add communication
  or reject the split.

The semantic facts above are what let a compiler know, up front, which
of these are legal and worth doing.
