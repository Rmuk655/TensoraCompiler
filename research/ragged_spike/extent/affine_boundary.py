#!/usr/bin/env python3
"""Is the ragged domain {(b,i): 0<=i<L[b]} an affine/polyhedral set?

Reviewer's precise MLIR point: affine loops DO take runtime bounds
(affine.for %i = 0 to %N) and affine exprs support +,-,c*,floordiv c.
So a bound that is AFFINE IN b (e.g. 2b+1) is fine. But L[b] -- an
arbitrary array lookup indexed by the induction variable -- is an
UNINTERPRETED runtime function of b, not an affine symbol. So the global
domain {(b,i): 0<=i<L[b]} is NOT one classical affine/polyhedral set.
Per fixed b it is a trivial runtime-bound loop; globally it is not.

This is why ragged scheduling differs from ordinary dynamic-shape
compilation, and it sharpens the Tensora claim: the contribution is not
"give the backend an affine domain it cannot obtain" (too strong). It is
about verified relationships among the CONTENTS of metadata tensors
(lengths, offsets) -- which ValueBounds (values/dims) and Affine (bounds)
do not capture.

Uses islpy (the real ISL library). See ../tiramisu/SETUP.md.
"""
import islpy as isl


def main():
    print(f"islpy {isl.__version__}\n")

    # 1. Bound AFFINE in b -> one symbolic parametric set (fine for affine/ISL)
    print("1. Bound affine in b (e.g. i < 2b+1): a single symbolic set:")
    s = isl.Set("[B] -> { [b,i] : 0<=b<B and 0<=i<2b+1 }")
    print(f"   {s}")
    concrete = isl.Set("{ [b,i] : 0<=b<4 and 0<=i<2b+1 }")
    print(f"   count at B=4: {int(concrete.count_val().to_python())} points "
          f"(representable, schedulable)\n")

    # 2. Arbitrary L[b] -> NOT expressible as one affine set.
    print("2. Arbitrary L[b] (array lookup): NOT one affine set.")
    print("   You cannot write '{ [b,i] : 0<=i<L[b] }' in ISL/affine with L an")
    print("   uninterpreted array -- L[b] is not an affine function of b.")
    L = [3, 7, 2, 5]
    # the only way to form it is to INSTANTIATE per b, using the runtime VALUES:
    terms = " or ".join(f"(b={b} and 0<=i<{L[b]})" for b in range(len(L)))
    inst = isl.Set(f"{{ [b,i] : {terms} }}")
    print(f"   with concrete values L={L}, as a union of per-b boxes:")
    print(f"   {inst}")
    print(f"   count: {int(inst.count_val().to_python())} points")
    print("   -> this needed the runtime values; there is no single symbolic")
    print("      affine form. Per fixed b it is trivial; globally it is not one")
    print("      polyhedral set.\n")

    print("Consequence for the pitch:")
    print("- Drop 'Tensora hands the backend an affine domain it cannot obtain.'")
    print("- MLIR already: runtime affine bounds (affine.for .. to %N), and")
    print("  ValueBounds already propagates per-op transfer relations. Do not")
    print("  reinvent either.")
    print("- The narrower open question: can Tensora verify relationships among")
    print("  the CONTENTS of metadata tensors (lengths[b] IS x[b]'s extent;")
    print("  offsets[b+1]-offsets[b]=lengths[b]) -- facts about contents, not")
    print("  dimensions -- that generic SSA/value-bound machinery cannot infer")
    print("  from the lowered program, and does preserving them help?")


if __name__ == "__main__":
    main()
