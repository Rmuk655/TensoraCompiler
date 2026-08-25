#!/usr/bin/env python3
"""Scheduling-headroom spike.

Future research directions (semantic axis types feeding TACO / Tiramisu /
DISTAL / Inductor) all rest on one claim: explicit semantic facts
(independence, reduction algebra, axis equivalence) let a compiler
produce a better single-node schedule than shape/graph inference alone.

Before building any of that, ask the honest question with numbers: on a
single GPU, how much headroom does PyTorch's own compiler (Inductor, via
torch.compile) leave? If Inductor is already near the hardware limit,
a semantic scheduler cannot beat it on one node -- it could only match
it more cheaply or more safely. The distributed direction (DISTAL) is
the interesting one, but it needs a cluster and is not tested here.

Method:
  1. Measure a practical bandwidth ceiling (a large device-to-device
     copy). Pointwise ops cannot go faster than moving their bytes.
  2. For pointwise and reduction kernels that the semantic facts would
     target, compare eager vs torch.compile, and report each one's
     achieved bandwidth as a percentage of the ceiling.
  3. If torch.compile is already near the ceiling, there is no single-
     node headroom for a semantic scheduler on these ops.

Usage:
  python sched_spike.py
"""
import statistics
import torch
import torch.nn.functional as F


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); fn(); t1.record(); torch.cuda.synchronize()
        ts.append(t0.elapsed_time(t1))
    return statistics.median(ts)


def gbps(bytes_moved, ms):
    return bytes_moved / (ms / 1e3) / 1e9


def main():
    if not torch.cuda.is_available():
        print("needs CUDA"); return
    dev = torch.device("cuda")
    dt = torch.float16
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} ({p.total_memory/(1024**3):.1f} GB)\n")

    # 1. bandwidth ceiling: device-to-device copy of a big tensor
    n = 64 * 1024 * 1024  # 64M fp16 = 128 MB
    a = torch.randn(n, device=dev, dtype=dt)
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a))
    ceiling = gbps(a.numel() * 2 * 2, ms)  # read + write
    print(f"bandwidth ceiling (d2d copy): {ceiling:6.0f} GB/s\n")

    B, S, H = 32, 1024, 4096  # a big activation tensor
    x = torch.randn(B, S, H, device=dev, dtype=dt)
    scale = torch.randn(H, device=dev, dtype=dt)
    bias = torch.randn(H, device=dev, dtype=dt)

    print(f"{'kernel':28} {'variant':10} {'latency':>10} {'GB/s':>8} {'% ceil':>8}")
    print("-" * 70)

    # 2a. pointwise chain -- all axes independent, fully fusable.
    def pointwise(t):
        return F.gelu(t * scale + bias) * 0.5
    bytes_pw = x.numel() * 2 * 2  # read x, write y (fused ideal)
    eager_ms = bench(lambda: pointwise(x))
    print(f"{'pointwise chain':28} {'eager':10} {eager_ms:8.3f}ms {gbps(bytes_pw, eager_ms):7.0f} "
          f"{100*gbps(bytes_pw, eager_ms)/ceiling:7.0f}%")
    try:
        cpw = torch.compile(pointwise)
        cms = bench(lambda: cpw(x))
        print(f"{'pointwise chain':28} {'compile':10} {cms:8.3f}ms {gbps(bytes_pw, cms):7.0f} "
              f"{100*gbps(bytes_pw, cms)/ceiling:7.0f}%")
    except Exception as e:
        print(f"{'pointwise chain':28} {'compile':10}  FAILED  {type(e).__name__}: {str(e)[:30]}")

    # 2b. RMSNorm -- reduction over the hidden axis (reduction algebra).
    w = torch.randn(H, device=dev, dtype=dt)
    def rmsnorm(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    bytes_rms = x.numel() * 2 * 2
    eager_ms = bench(lambda: rmsnorm(x))
    print(f"{'rmsnorm (reduction)':28} {'eager':10} {eager_ms:8.3f}ms {gbps(bytes_rms, eager_ms):7.0f} "
          f"{100*gbps(bytes_rms, eager_ms)/ceiling:7.0f}%")
    try:
        crms = torch.compile(rmsnorm)
        cms = bench(lambda: crms(x))
        print(f"{'rmsnorm (reduction)':28} {'compile':10} {cms:8.3f}ms {gbps(bytes_rms, cms):7.0f} "
              f"{100*gbps(bytes_rms, cms)/ceiling:7.0f}%")
    except Exception as e:
        print(f"{'rmsnorm (reduction)':28} {'compile':10}  FAILED  {type(e).__name__}: {str(e)[:30]}")

    # 2c. fused norm+pointwise -- fusion across a reduction and elementwise.
    def normchain(t):
        h = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5) * w
        return F.gelu(h * scale + bias)
    eager_ms = bench(lambda: normchain(x))
    print(f"{'norm+pointwise fused':28} {'eager':10} {eager_ms:8.3f}ms {gbps(bytes_rms, eager_ms):7.0f} "
          f"{100*gbps(bytes_rms, eager_ms)/ceiling:7.0f}%")
    try:
        cnc = torch.compile(normchain)
        cms = bench(lambda: cnc(x))
        print(f"{'norm+pointwise fused':28} {'compile':10} {cms:8.3f}ms {gbps(bytes_rms, cms):7.0f} "
              f"{100*gbps(bytes_rms, cms)/ceiling:7.0f}%")
    except Exception as e:
        print(f"{'norm+pointwise fused':28} {'compile':10}  FAILED  {type(e).__name__}: {str(e)[:30]}")

    print("\nRead: if 'compile' reaches a high % of the bandwidth ceiling, "
          "Inductor is\nalready near the single-node optimum for that kernel -- "
          "no headroom for a\nsemantic scheduler to beat it on one GPU.")


if __name__ == "__main__":
    main()
