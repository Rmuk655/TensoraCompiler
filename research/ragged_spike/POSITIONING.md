# Positioning: Tensora as an embedded ML kernel compiler (the Futhark model)

The clarifying frame from reviewer round 3: Tensora is **not** a PyTorch
replacement. It is a small compiled kernel language, embedded in a
Python/PyTorch application, like **Futhark**.

## The three roles (do not conflate)

```
PyTorch / Python            = the application framework
  model, training, data, autograd, distributed, serving
      |
      | one compute-intensive region, by hand
      v
Tensora                     = the embedded kernel compiler
  small .tns -> typed IR + verified extent contracts
      |
      v
StableHLO / MLIR            = portable dynamic-shape IR (a BACKEND, not a
  represents dynamic/bounded dims, runtime shape operands   competitor)
      |
      v
XLA / IREE / LLVM -> CPU/GPU/edge, and vendor libs (cuBLAS/cuDNN/
  FlashAttention) that Tensora does NOT try to beat
```

Tensora **proves** the extent contract, then lowers to StableHLO/MLIR and
lets proof-only information be erased where the backend does not need it
(normal compiler layering). StableHLO already carries the dynamic values;
its spec leaves their *correctness* to the producer -- which is exactly
the gap Tensora's verification fills. Upstreaming a generically-useful
piece to StableHLO (via `custom_call` first, then an RFC) is a possible
long-term outcome, not a V0.3 goal.

Target UX (Futhark-style FFI/module boundary):

```python
import torch, tensora_kernels
y = tensora_kernels.mlp(x, w)          # x,w are torch tensors; y comes back
```

The application owns everything else; Tensora owns one region. This also
gives the `.tns` language a clear reason to exist: it is the kernel DSL,
not a whole-model language. It reconnects v0.1 (the language) to a real
role.

## Two questions, kept separate

- **Scientific:** does verified per-instance extent information (see
  [PROPOSAL.md](PROPOSAL.md)) let the compiler generate/verify code for
  irregular ML computation that conventional dense representations do
  not? (The A-vs-B MLIR experiment decides this.)
- **System/product:** can Tensora be a useful *embedded kernel compiler*
  for PyTorch -- compile one region, call it transparently? (The
  integration experiment below decides this.)

Do not let a good answer to one masquerade as an answer to the other.

## Futhark: precedent and prior-art bound

Futhark is the precedent for "small compiled kernel language embedded in
a larger app" (Python modules / C FFI, reuse the host ecosystem). It is
also a **prior-art bound**:

- Futhark has compile-time **size-dependent types** (whole-array). So
  "typed size relationships" is not novel.
- Futhark's **`flatmap`** work (Aug 2026) targets *nonuniform nested
  data parallelism* -- inner parallel dims that vary per outer iteration,
  returned with a shape vector (essentially `lengths`/`cu_seqlens`).
  That is the exact `L[b]` territory. **And even there, performance is
  "significantly worse than hand-flattened code."**

Read that honestly both ways: the space is (1) actively worked by a
mature compiler, so not open ground to claim casually; and (2) still
**not solved for performance**, even in Futhark -- so it is a real,
hard, open problem, not a settled one.

## Where Tensora can differentiate (not "runtime extents" alone)

Futhark is a *general* data-parallel array language. Tensora's possible
differentiation is being **ML-specific**:

- ML operators as first-class (matmul, conv, attention, gather) with
  their extent-transfer functions built in;
- ML axis semantics (batch/head/reduction/ragged) as typed facts;
- embeddable specifically into the PyTorch/ML workflow;
- (later) ML-oriented distributed placement.

The claim is not "we invented irregular nested parallelism." It is
"we bring verified extent reasoning into an *ML* kernel compiler that
embeds in ML applications."

## Design principle (from Futhark's own warnings)

Futhark deliberately restricts language generality because generality
hurts efficient GPU codegen. **A small, closed DSL is an asset, not a
limitation.** Tensora should stay a deliberately restricted
high-performance region language, not drift toward a full framework.

## The missing system experiment (the Futhark-style integration test)

Not "can Tensora replace PyTorch?" but:

> **What is the smallest PyTorch application where replacing one
> compute-intensive region with a Tensora kernel produces a tangible
> benefit or capability?**

Concretely: take the wav2vec2 conv-subsampling region (where `L[b]`,
masks, and offsets interact), compile just that region with Tensora,
call it from PyTorch, and compare against PyTorch eager / `torch.compile`
on correctness, the manual bookkeeping removed, and runtime/memory.

This is a build task (a Tensora->native kernel + a Python/FFI boundary),
and it is the system-goal counterpart to the scientific A/B experiment.
Both are V0.3; neither is done yet.
