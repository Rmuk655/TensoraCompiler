#!/usr/bin/env python3
"""Compute-bound GEMM spike.

The scheduling spike (sched_spike.py) only tested bandwidth-bound kernels
and showed Inductor near the copy-bandwidth ceiling. Fair challenge (from
a reviewer): that says nothing about COMPUTE-bound kernels. So test a
large square matmul, which is compute-bound and runs on tensor cores.

The question: can torch.compile beat plain torch.matmul (cuBLAS) on a
compute-bound GEMM? If not, cuBLAS is the practical ceiling and there is
no headroom there either -- the "near-optimal" claim extends to the
compute-bound case. Reports achieved TFLOP/s.

Usage: python gemm_spike.py
"""
import statistics
import torch


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); fn(); t1.record(); torch.cuda.synchronize()
        t.append(t0.elapsed_time(t1))
    return statistics.median(t)


def main():
    if not torch.cuda.is_available():
        print("needs CUDA"); return
    dev = torch.device("cuda")
    dt = torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}  dtype=fp16\n")
    print(f"{'M=N=K':>7} {'variant':10} {'ms':>9} {'TFLOP/s':>9}")
    print("-" * 40)
    for S in (2048, 4096, 8192):
        A = torch.randn(S, S, device=dev, dtype=dt)
        B = torch.randn(S, S, device=dev, dtype=dt)
        flops = 2.0 * S * S * S
        eager_ms = bench(lambda: A @ B)
        print(f"{S:>7} {'eager':10} {eager_ms:7.2f}ms {flops/eager_ms/1e9:8.1f}")
        try:
            cf = torch.compile(lambda x, y: x @ y)
            cms = bench(lambda: cf(A, B))
            print(f"{S:>7} {'compile':10} {cms:7.2f}ms {flops/cms/1e9:8.1f}")
        except Exception as e:
            print(f"{S:>7} {'compile':10}  FAILED  {type(e).__name__}")
        torch.cuda.empty_cache()
    print("\nRead: if compile does not beat eager torch.matmul, cuBLAS is the\n"
          "practical ceiling on this compute-bound kernel -- no headroom for a\n"
          "semantic scheduler here either. Peak fp16 for this laptop GPU is a few\n"
          "tens of TFLOP/s, so a high achieved number also means near the roofline.")


if __name__ == "__main__":
    main()
