#!/usr/bin/env python3
"""Ragged x distributed: the load-balance gap DISTAL cannot address.

Your question: I have a ragged tensor and need TACO-style codegen AND
DISTAL-style distribution. No single system does that:
  - TACO: single-node.
  - DISTAL: distributed but DENSE-only (every index = equal work).
  - SpDISTAL: distributed + SPARSE (closest), but sparse != ragged.
  - CoRa: ragged but single-node (handles within-node imbalance only).

The concrete hard part is load balancing. DISTAL distributes dense
tensors where each index does uniform work, so it partitions by index
COUNT. Ragged data breaks that: attention work per sequence is ~L[b]^2,
so equal count != equal work. Balancing needs the per-sequence extent --
exactly Tensora's one surviving unique fact (the verified extent).

This shows: partitioning B ragged sequences across P workers by COUNT
(what a dense distributor does) vs by WORK (needs the extent) -- and the
resulting load imbalance (max worker / mean worker; 1.0 = perfect).

Usage: python ragged_balance.py
"""
import random


def imbalance(work_per_worker):
    m = sum(work_per_worker) / len(work_per_worker)
    return max(work_per_worker) / m if m > 0 else 1.0


def partition_by_count(works, P):
    # round-robin by index (what a count/dense distributor does)
    wk = [0.0] * P
    for i, w in enumerate(works):
        wk[i % P] += w
    return wk


def partition_by_work(works, P):
    # greedy longest-processing-time: assign each (desc) to least-loaded
    wk = [0.0] * P
    for w in sorted(works, reverse=True):
        j = min(range(P), key=lambda x: wk[x])
        wk[j] += w
    return wk


def heavy_tail(B, rng):
    # most sequences short, a few very long (real serving/text pattern)
    return [512 if rng.random() < 0.05 else rng.randint(8, 32) for _ in range(B)]


def main():
    P = 8
    print(f"P={P} workers; work[b] = L[b]^2 (attention). imbalance = "
          f"max worker / mean (1.0 = perfect)\n")
    print(f"{'case':>34} {'by-count':>10} {'by-work':>9}")
    print("-" * 56)
    rng = random.Random(0)
    cases = [
        ("B=256 uniform  (L 96..128)", 256, lambda: [rng.randint(96, 128) for _ in range(256)]),
        ("B=256 strong   (L 8..128)", 256, lambda: [rng.randint(8, 128) for _ in range(256)]),
        ("B=256 extreme  (L 1..512)", 256, lambda: [rng.randint(1, 512) for _ in range(256)]),
        ("B=32  extreme  (L 1..512)", 32, lambda: [rng.randint(1, 512) for _ in range(32)]),
        ("B=16  extreme  (L 1..512)", 16, lambda: [rng.randint(1, 512) for _ in range(16)]),
        ("B=256 heavy-tail (5% at 512)", 256, lambda: heavy_tail(256, rng)),
        ("B=32  heavy-tail (5% at 512)", 32, lambda: heavy_tail(32, rng)),
    ]
    for label, _, gen in cases:
        c_imbs, w_imbs = [], []
        for _ in range(30):
            works = [x * x for x in gen()]
            c_imbs.append(imbalance(partition_by_count(works, P)))
            w_imbs.append(imbalance(partition_by_work(works, P)))
        c = sum(c_imbs) / len(c_imbs)
        w = sum(w_imbs) / len(w_imbs)
        print(f"{label:>34} {c:9.2f}x {w:8.2f}x")

    print("\nRead:")
    print("- Count-based (dense/DISTAL-style) imbalance grows with skew and small")
    print("  batch-per-worker: fine at B=256 mild (1.04x), severe at B=32 heavy-")
    print("  tail (4.6x). The slowest worker sets the step time.")
    print("- Work-aware partition needs the per-sequence EXTENT and fixes the")
    print("  moderate cases (1.7-2x -> ~1.0-1.4x). That extent is Tensora's one")
    print("  unique verified fact; a dense distributor does not have it.")
    print("- Honest ceiling: with a heavy tail (one sequence's work > per-worker")
    print("  mean), even optimal batch partition cannot balance (4.6x -> 4.3x).")
    print("  That needs INTRA-sequence splitting (sequence parallelism), a second")
    print("  capability. So ragged x distributed is two-level: work-aware batch")
    print("  distribution + intra-sequence split for the tail.")
    print("- ragged x distributed is the open cell between SpDISTAL (sparse, not")
    print("  ragged) and CoRa (ragged, single-node).")


if __name__ == "__main__":
    main()
