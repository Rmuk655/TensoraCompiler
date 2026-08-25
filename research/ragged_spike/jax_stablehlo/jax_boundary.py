#!/usr/bin/env python3
"""JAX/StableHLO boundary experiment.

Reviewer's killer test: JAX shape polymorphism verifies GLOBAL symbolic
dimension relationships. But a PER-INSTANCE extent (L[b] from a lengths
array) is just opaque int data to JAX -- it does not verify a contract
like cu[i+1]-cu[i] == L[i] or L1 = conv_extent(L0). StableHLO's own spec
says dynamic-shape correctness is the PRODUCER's responsibility. So this
tests exactly where Tensora could add value.
"""
import numpy as np
import jax, jax.numpy as jnp
from jax import export

print(f"jax {jax.__version__}\n")

# 1. GLOBAL symbolic dimension relationship -- JAX verifies it.
print("1. JAX shape polymorphism verifies a GLOBAL symbolic relationship:")
try:
    (n, d) = export.symbolic_shape("n, d")
    def f(x):                      # requires last dim even: reshape [n, 2k]->[n,k,2]
        return x.reshape(x.shape[0], x.shape[1] // 2, 2)
    exp = export.export(jax.jit(f))(
        jax.ShapeDtypeStruct(export.symbolic_shape("n, 2*k"), np.float32))
    print("   exported f with symbolic shape (n, 2*k) -- relationship captured")
    ok = exp.call(np.zeros((4, 8), np.float32))          # 8 even: OK
    print(f"   call with (4,8): OK, out shape {ok.shape}")
    try:
        exp.call(np.zeros((4, 7), np.float32))           # 7 odd: violates 2*k
        print("   call with (4,7): accepted (unexpected)")
    except Exception as e:
        print(f"   call with (4,7): REJECTED -> {type(e).__name__} "
              f"(global relation enforced)")
except Exception as e:
    print(f"   (shape-poly API differed: {type(e).__name__}: {str(e)[:50]})")

# 2. PER-INSTANCE extent contract -- JAX does NOT verify it.
print("\n2. Per-instance extent contract (L[b], cu_seqlens): JAX does NOT verify:")
B, Smax, D = 4, 512, 8
lengths = jnp.array([500, 317, 91, 460])     # per-row real extents (metadata)
# correct prefix-sum offsets
cu_good = jnp.concatenate([jnp.zeros(1, int), jnp.cumsum(lengths)])
# a WRONG cu_seqlens (off by one) -- JAX happily accepts and uses it
cu_bad = cu_good.at[2].add(1)
def use_offsets(cu):
    return cu[-1]                              # "total packed tokens"
print(f"   lengths      = {lengths.tolist()}")
print(f"   cu (correct) = {cu_good.tolist()}")
print(f"   cu (wrong)   = {cu_bad.tolist()}")
print(f"   jax runs with the WRONG cu_seqlens, result={int(use_offsets(cu_bad))}, "
      f"no error")
print("   -> JAX has no notion that cu must satisfy cu[i+1]-cu[i]==L[i]; the")
print("      offsets are ordinary int data. Correctness is the PRODUCER's job.")
print("      That is exactly the per-instance extent contract Tensora would")
print("      make first-class and verify. Not shape polymorphism; not nonzero.")
