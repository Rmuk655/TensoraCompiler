#!/usr/bin/env python3
"""Two contracts, and the cu_seqlens contract (research-risk tests).

The reviewer's sharp point: verifying L' = f(L) is not the whole story.

Contract 1 (relational, STATIC): L_out = f(L_in). Compile-time.
Contract 2 (representation consistency, RUNTIME): the declared length
matches the actual buffer, 0 <= L[b] <= allocated[b].

A compile-time proof of Contract 1 does NOT prove Contract 2. If the
buffer holds 120 tokens but metadata says 140, the relational proof still
"holds" and the result is silently wrong. So a real system needs a
runtime boundary check for Contract 2, then static propagation inside.

This also tests the cu_seqlens contract: cu_seqlens must be the exact
prefix sum of the declared lengths; a wrong one corrupts a packed kernel.

Usage: python contracts.py
"""
import torch


def contract1_vs_contract2():
    print("1. Relational contract holds, but representation contract can fail:")
    # buffer actually has 120 real rows, but we CLAIM length 140
    actual = 120
    claimed_L = 140
    buf = torch.randn(160, 8)          # allocated 160
    K, S, P = 5, 2, 2
    f = lambda L: (L + 2*P - K)//S + 1
    # Contract 1 (relational) is satisfied for the CLAIMED length:
    print(f"   claimed L_in={claimed_L} -> L_out=f={f(claimed_L)} "
          f"(relational contract: satisfied)")
    # but pooling/reducing over claimed 140 reads 20 rows of garbage/pad:
    real_mean = buf[:actual].mean().item()
    claimed_mean = buf[:claimed_L].mean().item()
    print(f"   mean over real {actual} rows = {real_mean:+.4f}")
    print(f"   mean over claimed {claimed_L} rows = {claimed_mean:+.4f}  "
          f"(SILENTLY WRONG)")
    print(f"   -> Contract 1 (static) cannot catch this; only a RUNTIME check")
    print(f"      0<=L<=allocated and L matches the buffer can. Two contracts.\n")


def cu_seqlens_contract():
    print("2. cu_seqlens contract: must be the exact prefix sum of lengths:")
    L = torch.tensor([8, 16, 4, 32])
    good = torch.zeros(len(L) + 1, dtype=torch.long)
    good[1:] = torch.cumsum(L, 0)
    print(f"   lengths       = {L.tolist()}")
    print(f"   cu_seqlens    = {good.tolist()}  (good)")
    # invariants a compiler could check symbolically:
    inv = (good[0].item() == 0 and good[-1].item() == int(L.sum())
           and all((good[i+1]-good[i]).item() == L[i].item() for i in range(len(L))))
    print(f"   invariants (cu[0]=0, cu[-1]=sum L, cu[i+1]-cu[i]=L[i]): "
          f"{'HOLD' if inv else 'FAIL'}")
    # a wrong cu_seqlens (off-by-one accumulation) violates the invariant:
    bad = good.clone(); bad[2] += 1
    bad_ok = all((bad[i+1]-bad[i]).item() == L[i].item() for i in range(len(L)))
    print(f"   corrupted cu_seqlens {bad.tolist()}: invariant "
          f"{'holds' if bad_ok else 'VIOLATED (caught)'}")
    print("   -> the compiler can verify cu_seqlens is the correct prefix-sum")
    print("      representation of the declared extent before using it in a")
    print("      packed kernel. A crisp, checkable contract.\n")


def main():
    contract1_vs_contract2()
    cu_seqlens_contract()
    print("Takeaway: the language/compiler story needs BOTH -- a runtime check")
    print("at the data boundary (representation consistency) and static")
    print("propagation of the relational contract inside. The proposal must say")
    print("this explicitly; 'static verification' alone overclaims.")


if __name__ == "__main__":
    main()
