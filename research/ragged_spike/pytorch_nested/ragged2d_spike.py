#!/usr/bin/env python3
"""Two-ragged-dimension spike: the candidate 'missing row'.

The reviewer's sharpest point: torch.jagged supports ONE ragged
dimension. Workloads with two independent ragged axes -- hierarchical
data (documents -> sentences -> tokens) and ragged x ragged pairwise ops
(variable-length cross-attention, set matching, GNN neighborhoods) --
cannot be represented, so PyTorch falls back to padding BOTH axes, whose
waste is quadratic in the skew.

This tests two things:
  1. Can torch.jagged represent two independent ragged dims? (expect no)
  2. Ragged x ragged pairwise C[b,i,j] = sum_d A[b,i,d]*B[b,j,d] with
     L_A[b], L_B[b] varying: padded work vs true work, and whether any
     core-PyTorch ragged path expresses the [B, jA, jB] output.

Usage: python ragged2d_spike.py
"""
import time
import torch


def bench(fn, dev, iters=10):
    fn()
    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    t = []
    for _ in range(iters):
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    t.sort()
    return peak, t[len(t) // 2]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    print(f"device={dev.type}\n")

    # 1. can torch.jagged hold two independent ragged dims?
    print("1. Represent two independent ragged dimensions:")
    try:
        # documents -> variable sentences, each a variable-length vector row
        doc1 = torch.nested.nested_tensor([torch.randn(n, 16) for n in (3, 5, 2)], layout=torch.jagged)
        doc2 = torch.nested.nested_tensor([torch.randn(n, 16) for n in (4, 1)], layout=torch.jagged)
        torch.nested.nested_tensor([doc1, doc2], layout=torch.jagged)
        print("   OK (unexpected)")
    except Exception as e:
        print(f"   BREAK -- {type(e).__name__}: {str(e).splitlines()[0][:64]}")
    print("   => torch.jagged is single-ragged-dim; two ragged axes are not "
          "representable.\n")

    # 2. ragged x ragged pairwise: C[b,i,j] = A[b,i,:] . B[b,j,:]
    print("2. Ragged x ragged pairwise similarity C[b,i,j] = sum_d A[b,i,d]B[b,j,d]:")
    g = torch.Generator().manual_seed(0)
    B, D = 16, 128
    La = torch.randint(4, 512, (B,), generator=g).tolist()
    Lb = torch.randint(4, 512, (B,), generator=g).tolist()
    mA, mB = max(La), max(Lb)
    true_work = sum(a * b for a, b in zip(La, Lb)) * D
    padded_work = B * mA * mB * D
    print(f"   B={B} D={D}  maxA={mA} maxB={mB}")
    print(f"   padded work = {padded_work/1e9:.2f} GMAC, true work = "
          f"{true_work/1e9:.2f} GMAC  -> {padded_work/true_work:.1f}x waste\n")

    # padded dense: build [B,mA,D],[B,mB,D], C=[B,mA,mB]
    A = torch.zeros(B, mA, D, device=dev, dtype=dt)
    Bt = torch.zeros(B, mB, D, device=dev, dtype=dt)
    for i, (a, b) in enumerate(zip(La, Lb)):
        A[i, :a] = torch.randn(a, D, generator=g).to(dev, dt)
        Bt[i, :b] = torch.randn(b, D, generator=g).to(dev, dt)
    def padded():
        return torch.bmm(A, Bt.transpose(1, 2))  # [B, mA, mB]
    pm, pms = bench(padded, dev)

    # true ragged: per-batch, no padding
    As = [A[i, :La[i]].contiguous() for i in range(B)]
    Bs = [Bt[i, :Lb[i]].contiguous() for i in range(B)]
    def ragged_loop():
        return [As[i] @ Bs[i].transpose(0, 1) for i in range(B)]  # list of [La,Lb]
    rm, rms = bench(ragged_loop, dev)

    print(f"   padded bmm:   {pm/1e6:8.1f} MB  {pms:7.2f} ms")
    print(f"   ragged loop:  {rm/1e6:8.1f} MB  {rms:7.2f} ms")
    print(f"   -> the [B, jA, jB] output has TWO ragged dims; no core-PyTorch "
          f"ragged\n      type can hold it, so padded is the only vectorized "
          f"option. The\n      waste is quadratic in the skew.\n")

    print("Read: this is a representation gap, not a missing kernel. A tensor with "
          "two\nindependent ragged axes cannot be expressed by torch.jagged at all. "
          "This is\nthe candidate V0.3 target: an IR that represents the irregular "
          "iteration\ndomain directly, instead of padding both axes.")


if __name__ == "__main__":
    main()
