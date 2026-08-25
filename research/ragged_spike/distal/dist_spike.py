#!/usr/bin/env python3
"""Distributed spike: measure the communication a semantic fact predicts.

Earlier, shard_analysis.py only MODELED communication. This MEASURES it,
with real collective communication across processes (torch.distributed,
gloo backend) on this machine's CPU cores. It is co-located, not a real
cluster -- gloo over loopback is faster than a network, so the measured
communication cost is a LOWER BOUND on a real cluster. But it is a real
all-reduce, not a hand-written estimate.

Workload: Y = X @ W,  X[batch=B, seq=S, hidden=D],  W[hidden=D, out=F].
  hidden=D is the reduction (contraction) axis; batch=B is independent.

Two shardings across P processes:
  - shard batch  (INDEPENDENT axis): each rank owns B/P rows, computes
    its whole slice locally. No collective. Communication = 0.
  - shard hidden (REDUCTION axis): each rank owns D/P of the contraction,
    computes a PARTIAL [B,S,F], then all-reduces (sum) to the full result.
    The all-reduce is the cost the semantic fact predicts.

The point: the semantic fact (batch=independent vs hidden=reduction)
decides which sharding pays the all-reduce -- measured here, not modeled.

Usage: python dist_spike.py --procs 4
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, P, B, S, D, F, iters, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=P)
    torch.set_num_threads(max(1, 22 // P))  # split the 22 cores across ranks

    g = torch.Generator().manual_seed(0)  # same data on every rank
    X = torch.randn(B, S, D, generator=g)
    W = torch.randn(D, F, generator=g)

    def bench(fn):
        fn()  # warmup
        dist.barrier()
        t = []
        for _ in range(iters):
            dist.barrier()
            t0 = time.perf_counter()
            fn()
            dist.barrier()
            t.append((time.perf_counter() - t0) * 1e3)
        t.sort()
        return t[len(t) // 2]

    # shard batch: independent axis, no communication
    b0, b1 = rank * (B // P), (rank + 1) * (B // P)
    def shard_batch():
        return X[b0:b1] @ W  # [B/P, S, F], stays local

    # shard hidden: reduction axis, needs an all-reduce
    d0, d1 = rank * (D // P), (rank + 1) * (D // P)
    Xh, Wh = X[:, :, d0:d1].contiguous(), W[d0:d1, :].contiguous()
    def shard_hidden():
        partial = Xh @ Wh  # [B, S, F] partial sum
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial

    # isolated all-reduce cost, for reference
    buf = torch.randn(B, S, F)
    def allreduce_only():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    batch_ms = bench(shard_batch)
    hidden_ms = bench(shard_hidden)
    ar_ms = bench(allreduce_only)

    # correctness: ALL ranks must call the collective, only rank 0 compares
    got = shard_hidden()
    if rank == 0:
        ref = X @ W
        err = (got - ref).abs().max().item()
        out_mb = B * S * F * 4 / 1e6
        print(f"\nworkload: X[{B},{S},{D}] @ W[{D},{F}]  across P={P} gloo ranks "
              f"({22 // P} threads each)")
        print(f"output all-reduce size: {out_mb:.0f} MB  (hidden all-reduce moves this)\n")
        print(f"{'sharding':16} {'axis kind':12} {'per-step ms':>12} {'communication':>14}")
        print("-" * 58)
        print(f"{'shard batch':16} {'independent':12} {batch_ms:10.1f}ms {'none':>14}")
        print(f"{'shard hidden':16} {'reduction':12} {hidden_ms:10.1f}ms "
              f"{'all-reduce':>14}")
        print(f"{'(all-reduce only)':16} {'':12} {ar_ms:10.1f}ms {'':>14}")
        print(f"\ncorrectness: shard-hidden vs single-process reference "
              f"max err {err:.1e}")
        overhead = hidden_ms - batch_ms
        print(f"\nMeasured: sharding the reduction axis costs ~{overhead:.1f} ms/step "
              f"more than\nsharding the independent axis, from the real all-reduce. "
              f"The semantic fact\n(batch=independent vs hidden=reduction) predicts this "
              f"before running.")
        print("Caveat: gloo over loopback is far faster than a real network, so this "
              f"\ncommunication cost is a LOWER BOUND on a real cluster.")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--F", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--port", type=int, default=29512)
    args = ap.parse_args()
    assert args.B % args.procs == 0 and args.D % args.procs == 0, "B and D must divide procs"
    mp.spawn(worker, args=(args.procs, args.B, args.S, args.D, args.F, args.iters, args.port),
             nprocs=args.procs, join=True)


if __name__ == "__main__":
    main()
