#!/usr/bin/env python3
"""TACO-hypothesis spike (not TACO the system).

The real TACO (Stanford) is a C++ compiler; the PyPI `pytaco` is an
unrelated package, and TACO needs a source build. So, as with the
distributed spike, test the HYPOTHESIS behind TACO instead of the
system: exposing a tensor's structure/format to the compiler lets it
generate a kernel whose work is proportional to the nonzeros, not the
dense size. Here the "format as information" is a CSR sparse matrix, and
the kernel is a real sparse @ dense matmul (torch.sparse), compared with
the same computation done densely.

This is TACO's core claim (structure -> less work), measured with a real
kernel. It is not TACO's codegen, and sparsity is not raggedness (a
ragged tensor is dense within each object). It tests the principle only.

Usage: python taco_hypothesis.py
"""
import time
import torch


def bench(fn, iters=10, dev="cuda"):
    fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    t.sort()
    return t[len(t) // 2]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32
    N, K, F = 4096, 4096, 512
    X = torch.randn(K, F, device=dev, dtype=dt)
    print(f"device={dev}  M[{N},{K}] @ X[{K},{F}]  (M has a given density)\n")
    print(f"{'density':>8} {'dense ms':>10} {'sparse ms':>11} {'speedup':>9} "
          f"{'dense MB':>9} {'sparse MB':>10}")
    print("-" * 62)
    for density in [0.5, 0.1, 0.02, 0.005]:
        # dense M
        M = (torch.rand(N, K, device=dev, dtype=dt) < density).to(dt) * \
            torch.randn(N, K, device=dev, dtype=dt)
        Mcsr = M.to_sparse_csr()
        dense_mb = M.numel() * 4 / 1e6
        nnz = Mcsr.values().numel()
        sparse_mb = (nnz * 4 + nnz * 4 + (N + 1) * 4) / 1e6  # values+cols+rowptr

        ref = M @ X
        def dense():
            return M @ X
        def sparse():
            return torch.sparse.mm(Mcsr, X)
        try:
            got = torch.sparse.mm(Mcsr, X)
            err = (got - ref).abs().max().item()
        except Exception as e:
            print(f"{density:8.3f}  sparse mm failed: {type(e).__name__}: {str(e)[:30]}")
            continue
        dms = bench(dense, dev=dev)
        sms = bench(sparse, dev=dev)
        print(f"{density:8.3f} {dms:9.2f}ms {sms:10.2f}ms {dms/sms:8.1f}x "
              f"{dense_mb:8.0f} {sparse_mb:9.0f}   err {err:.1e}")
    print("\nRead: as the matrix gets sparser, the sparse (format-aware) kernel "
          "does\nless work and uses less memory -- work tracks nonzeros, not the "
          "dense size.\nThat is TACO's premise: structure exposed to the compiler "
          "is optimization\ninformation. (This is a proxy; sparsity != raggedness, "
          "and it is not TACO's\nown codegen -- real TACO needs a C++ source build.)")


if __name__ == "__main__":
    main()
