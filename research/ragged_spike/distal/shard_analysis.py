#!/usr/bin/env python3
"""Sharding-communication analysis (no cluster needed).

The distributed direction (DISTAL) is the one future direction that
could be un-dominated by single-node PyTorch. It needs a cluster to run,
but the KEY point can be shown analytically on any machine: a semantic
fact decides whether a chosen sharding is communication-free or
communication-heavy, and a shape-only view does not know that fact.

Workload: Y = X @ W
  X : Tensor[batch=B, seq=S, hidden=D]
  W : Tensor[hidden=D, out=F]
  hidden=D is the reduction (contraction) axis. batch=B is independent.

Two ways to split across P devices:
  - shard batch  (an INDEPENDENT axis): each device owns B/P rows,
    computes its whole result locally. No cross-device reduction.
    Communication = 0.
  - shard hidden (the REDUCTION axis): each device owns D/P of the
    contraction, produces a PARTIAL result, and an all-reduce over the
    output [B,S,F] is required to sum the partials.

The semantic facts (batch=independent, hidden=reduction) are exactly
what tells a compiler this, with no dependence analysis at run time. A
shape-only IR sees two axes of some size and must rediscover it.

Usage: python shard_analysis.py
"""

def analyze(B, S, D, F, P, bytes_per=2):
    total_macs = B * S * D * F  # multiply-accumulates for the matmul
    out_elems = B * S * F

    # shard batch: independent axis -> no communication
    batch_local_macs = total_macs // P
    batch_comm_bytes = 0

    # shard hidden: reduction axis -> all-reduce the output partials
    hidden_local_macs = total_macs // P
    # ring all-reduce moves ~2*(P-1)/P * (message size) bytes per device
    hidden_comm_bytes = int(2 * (P - 1) / P * out_elems * bytes_per)

    return {
        "total_macs": total_macs,
        "out_elems": out_elems,
        "batch": (batch_local_macs, batch_comm_bytes),
        "hidden": (hidden_local_macs, hidden_comm_bytes),
    }


def main():
    B, S, D, F = 32, 2048, 4096, 4096
    print("workload: Y = X @ W")
    print(f"  X[batch={B}, seq={S}, hidden={D}]  W[hidden={D}, out={F}]")
    print("  semantic facts: batch = independent, hidden = reduction\n")
    print(f"{'devices':>7} {'shard-batch comm':>18} {'shard-hidden comm':>18}   note")
    print("-" * 72)
    for P in [2, 4, 8, 16, 32]:
        r = analyze(B, S, D, F, P)
        bc = r["batch"][1]
        hc = r["hidden"][1]
        print(f"{P:>7} {bc/1e6:14.1f} MB {hc/1e6:15.1f} MB   "
              f"batch axis is free; hidden axis needs an all-reduce")
    print()
    print("The point: both shardings do the same local compute (total/P MACs),")
    print("but shard-batch moves 0 bytes and shard-hidden moves hundreds of MB")
    print("per step. The semantic facts (batch=independent, hidden=reduction)")
    print("decide this up front. A shape-only IR sees [32,2048,4096] @ [4096,4096]")
    print("and must rediscover that batch carries no cross-device dependence.")
    print()
    print("This is analytical only. A real speedup claim needs a real cluster")
    print("and a real distributed backend (e.g. DISTAL). That is future research,")
    print("not a laptop spike.")


if __name__ == "__main__":
    main()
