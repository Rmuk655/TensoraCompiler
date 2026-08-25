# V0.3 research proposal (1 page, revised after reviewer round 2)

Distilled from the V0.2 P0 spikes ([MATRIX.md](MATRIX.md)) and two
reviewer rounds. This is the falsifiable version. It deliberately does
**not** claim to be first at size-dependent typing (Futhark is) or at
consuming runtime bounds (MLIR is).

## Question (modest, survives Futhark/JAX/MLIR)

Can an ML type system establish and verify relationships among the
**contents** of per-instance metadata tensors and the tensors they
describe (`lengths[b]` IS `x[b]`'s extent; `offsets` = prefix-sum of
`lengths`; `Ly[b] = conv_extent(Lx[b])`) -- relationships that MLIR's
generic SSA / value-bound / affine machinery cannot infer from the
lowered program alone -- and does preserving those verified relationships
enable a correctness guarantee or optimization?

## Prior art that bounds the novelty (must be in the matrix)

- **Futhark**: compile-time **size-dependent types** over whole arrays,
  and (Aug 2026) **`flatmap`** for nonuniform nested parallelism -- inner
  dims that vary per outer iteration, returned with a shape vector
  (essentially `lengths`/`cu_seqlens`). That is the exact `L[b]`
  territory, so "typed extent relationships" and even "per-instance
  irregular parallelism" are **not novel**. Two honest reads: the space
  is actively worked by a mature compiler, AND Futhark's own `flatmap`
  is still "significantly worse than hand-flattened" -- so it is hard and
  not solved. Difference to establish: this in an **ML-specific** kernel
  compiler, embedded in ML apps (see [POSITIONING.md](POSITIONING.md)).
- **JAX shape polymorphism / StableHLO**: this needs a precise
  three-level distinction (terminology matters -- see below). JAX verifies
  **global symbolic dimension** relations; StableHLO represents dynamic /
  bounded dims and runtime shape operands, but its spec says dynamic-shape
  correctness is **the producer's responsibility**. Measured
  (`jax_stablehlo/jax_boundary.py`): JAX rejected `(4,7)` against `(n,2*k)`
  (global relation enforced) but ran a **wrong `cu_seqlens` with no error**
  (per-instance contract unverified). So JAX/StableHLO bound the novelty
  but leave the exact hole open.
- **MLIR `ValueBoundsOpInterface` + affine dynamic bounds**: two things
  MLIR already does, so do not reinvent them. (1) Affine loops take
  runtime bounds (`affine.for %i = 0 to %N`) and support the
  `+`,`-`,`c*`,`floordiv c` arithmetic conv lengths need. (2)
  `ValueBoundsOpInterface` already lets each op expose how its result
  relates to its operands and **propagates** those relations -- so
  "each operator supplies a transfer relation and the compiler
  propagates it" is **not novel** either. So both "the backend cannot
  obtain L[b]" and "per-op transfer propagation is new" are false.
  But: `L[b]` (array lookup indexed by an induction var) is an
  *uninterpreted* runtime function, not an affine symbol, so the global
  domain `{(b,i): 0<=i<L[b]}` is **not one affine set** -- confirmed with
  islpy (`extent/affine_boundary.py`): affine-in-b bounds are one set;
  arbitrary `L[b]` needs a union of per-b boxes with runtime values.
- **TensorFlow RaggedTensor**: multiple ragged dims already exist.
- **2026 staged shape-dependent types** (arXiv 2604.23807): compile-time
  tensor shape checking — adjacent, must be cited.

## Terminology: three kinds of dynamism (do not conflate)

| kind | example | JAX shape poly | StableHLO | Tensora aim |
|---|---|---|---|---|
| symbolic **global** dim | `[B,S,D]`, unknown `S` | verifies | represents | (reuse) |
| **per-instance runtime extent** | `[B,Smax,D]` + `L[b]` | no (L is data) | carries values; producer owns correctness | **verify + propagate** |
| **data-derived extent** | `len(nonzero(x))` | no | some bounded support | out of scope |

Crucial: `L[b]` coming from an input `lengths` array is a *per-instance
runtime extent* (metadata), **not** "data-dependent" in StableHLO's
`nonzero` sense. Reserve "data-dependent" for the third row. This avoids
an argument with anyone who knows XLA terminology.

## Corrected contribution (relationships among metadata CONTENTS)

Not a new loop representation, not runtime affine bounds, not per-op
transfer propagation (MLIR ValueBounds/Affine have all three). Not
size-dependent typing (Futhark). Not global symbolic shapes (JAX). The
remaining, narrower candidate:

> **Verified relationships among the CONTENTS of per-instance metadata
> tensors and the tensors they describe** -- e.g. "`lengths[b]` IS the
> valid extent of axis `time` of `x[b]`", "`offsets` is the prefix-sum
> partition of `lengths`", "`Ly[b] = conv_extent(Lx[b])`" -- established
> and checked in an ML type system, then lowered.

The key distinction: ValueBounds reasons about *values and dimensions*;
Affine about *bounds*. Neither captures a semantic relationship among the
*contents* of metadata tensors (offsets = prefix-sum of lengths;
lengths = the extent of another tensor). Those are the facts a producer
maintains by hand today (and JAX runs a wrong `cu_seqlens` silently,
per `jax_stablehlo/jax_boundary.py`).

Do **not** claim "Tensora hands the backend an affine domain it cannot
obtain" -- too strong (the global ragged domain is not affine for anyone;
per-instance it is a trivial runtime bound MLIR already accepts).

## Established this round (runnable)

- **Extent propagation works** (`extent/extent_propagation.py`): conv
  transfer functions verified vs real output; a wrong one caught before
  running; 2.63x padded waste.
- **Scope of the f(L) model** (`extent/operator_taxonomy.py`): covers
  A (unchanged), B (`f(L_in)`: conv/pool/concat), C (`f(L_in, metadata)`:
  gather/pad) — but **not D** (nonzero/filter/MoE/dynamic-decode: extent
  is a runtime VALUE of the data). Honest scope: a large sub-language,
  not all of ML.
- **Two contracts, not one** (`extent/contracts.py`): the relational
  contract `L'=f(L)` (static) does **not** imply representation
  consistency (`L[b]` matches the buffer) — a wrong length is silently
  wrong. Need a **runtime boundary check** + static internal propagation.
  Also: `cu_seqlens` = prefix-sum(L) is a crisp checkable invariant.
- **JAX/StableHLO boundary** (`jax_stablehlo/jax_boundary.py`): JAX
  verifies a global symbolic relation (rejects `(4,7)` for `(n,2*k)`) but
  runs a **wrong `cu_seqlens` with no error** — per-instance extent
  correctness is the producer's job. The exact hole.
- **Ragged domain is not affine** (`extent/affine_boundary.py`): an
  affine-in-b bound is one ISL set; arbitrary `L[b]` needs a union of
  per-b boxes with runtime values. So "hand the backend an affine domain"
  is wrong; the contribution is about metadata *contents*, not the domain.

## The decisive experiment (not yet runnable here)

Compile the **same** varlen computation two ways, same runtime metadata:
- **A**: `L[b]`/`cu_seqlens` as opaque runtime SSA into MLIR;
- **B**: Tensora verifies provenance + transfer, then lowers.

Inspect generated MLIR/IR and measure: does B enable a transformation or
remove a runtime check that A cannot? Outcomes:
- identical IR → the contribution is a **type/checking/language** result
  (still a paper, but not a codegen one);
- B enables an optimization or removes a guard → a **compiler** result.

This needs the MLIR Python bindings / an MLIR build (not installed here).
It is the single next experiment, and it decides which kind of paper
this is. Do not start V0.3 implementation before it.

## StableHLO as a backend, not a competitor (and possible upstreaming)

Architecture: Tensora (ML semantics + verified extent contracts) →
**StableHLO** (portable dynamic-shape IR) → XLA/IREE → CPU/GPU/edge.
StableHLO already represents dynamic/bounded dims and runtime shape
operands, so Tensora should **prove the contract, then lower and let the
proof-only information be erased** where the backend does not need it --
normal compiler layering, not a StableHLO deficiency.

Upstreaming is a *possible outcome, not a V0.3 goal*: first make Tensora
the lab; if a specific verified relationship (e.g. a checked
`lengths`↔`cu_seqlens` link) turns out to be generically useful to
XLA/IREE, prototype it via `custom_call` / an auxiliary dialect, and only
then consider a StableHLO RFC. Do not design StableHLO extensions
speculatively. See [POSITIONING.md](POSITIONING.md).

## Benchmarks

1. Primary: wav2vec2-style **audio → subsampling → attention →
   pack/unpack**, where `L[b]`, masks, and offsets interact (not plain
   attention — too simple to show leverage).
2. Applied: ragged attention with `cu_seqlens`.
3. Stretch (demoted): ragged × distributed as a **two-level scheduling**
   study (batch balance + intra-sequence split), not a headline.

## Baselines / ablation

padded+mask · hand-written varlen (`cu_seqlens` by hand) · Tensora
without the contract · Tensora with it. The 3-vs-4 ablation credits the
contract, not "a new implementation."

## Falsification (kill it if)

- a wider search finds a system already doing per-instance verified
  extent propagation through an ML graph (not just Futhark's whole-array
  sizes);
- the decisive A/B experiment yields identical IR **and** the contract
  catches no bug tests/type-checkers already catch;
- the D-operator class turns out to dominate the target workloads (then
  the f(L) model is too narrow to matter).

## System goal (separate from the science)

Tensora is an **embedded ML kernel compiler** (the Futhark model): the
user keeps their PyTorch app and replaces one compute-intensive region
with a compiled Tensora kernel called over an FFI boundary -- not a
whole-model rewrite. The system-goal experiment (distinct from the
scientific A/B one): *the smallest PyTorch app where swapping one region
for a Tensora kernel gives a tangible benefit* (the wav2vec2 conv region).
Full framing: [POSITIONING.md](POSITIONING.md).

## Non-goals (first paper)

Routing to TACO/Tiramisu/DISTAL is a second-stage demo. Beating
vLLM/cuBLAS on kernel speed is not the claim. Replacing PyTorch is not
the goal -- Tensora is an embedded kernel compiler, not a framework.
