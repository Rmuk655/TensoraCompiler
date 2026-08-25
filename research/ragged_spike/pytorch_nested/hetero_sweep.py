#!/usr/bin/env python3
"""Heterogeneity sweep for ragged attention.

Vary ONLY the length distribution. Everything else is fixed: batch size,
maximum length (128), heads, head_dim, dtype, GPU. An earlier version
also grew the maximum length across cases, which changed absolute work
and kernel behaviour -- so it was not a clean causal test. Here maxlen
stays 128 in every case, so the only thing that changes is how uneven
the lengths are.

The clean signal is the theoretical "attention work eliminated"
(1 - sum(Li^2) / (B*maxlen^2)). Wall-time at these small sizes is noisy
(launch overhead dominates), so read it as a rough check, not a curve.
A fuller study would sample many batches from length distributions
rather than use five hand-picked cases.

Reuses the attention variants from spike.py.

Usage: python hetero_sweep.py
"""
import torch
from spike import make_qkv, flash_padded, ragged_loop, measure

# maxlen fixed at 128 in every case; only the distribution changes.
CASES = {
    "A uniform    [128,128,128,128]": [128, 128, 128, 128],
    "B mild       [128,120,112,104]": [128, 120, 112, 104],
    "C moderate   [128,96,64,32]   ": [128, 96, 64, 32],
    "D strong     [128,64,32,8]    ": [128, 64, 32, 8],
    "E extreme    [128,32,16,8]    ": [128, 32, 16, 8],
}


def main():
    if not torch.cuda.is_available():
        print("needs CUDA"); return
    dev = torch.device("cuda")
    dt = torch.float16
    heads, head_dim = 8, 64
    print(f"GPU: {torch.cuda.get_device_name(0)}   heads={heads} head_dim={head_dim} fp16\n")
    print(f"{'case':34} {'pad ratio':>9} {'work elim':>9} {'flash ms':>9} "
          f"{'ragged ms':>10} {'speedup':>8}")
    print("-" * 84)
    for name, lengths in CASES.items():
        B = len(lengths)
        maxlen = max(lengths)
        pad_ratio = B * maxlen / sum(lengths)
        work_elim = 1.0 - sum(L * L for L in lengths) / (B * maxlen * maxlen)
        qs, ks, vs = make_qkv(lengths, heads, head_dim, dt, dev)
        pm, pms, _ = measure(flash_padded, qs, ks, vs, maxlen, dev, iters=10)
        qs, ks, vs = make_qkv(lengths, heads, head_dim, dt, dev)
        rm, rms, _ = measure(ragged_loop, qs, ks, vs, maxlen, dev, iters=10)
        speedup = pms / rms if rms > 0 else float("nan")
        print(f"{name:34} {pad_ratio:8.2f}x {100*work_elim:7.0f}% "
              f"{pms:8.2f}ms {rms:9.2f}ms {speedup:7.1f}x")
        torch.cuda.empty_cache()
    print("\nRead: 'work elim' (theoretical) is the clean signal -- it rises "
          "with skew\nat fixed maxlen. Wall-time is noisy at these sizes; do not "
          "read it as a curve.")


if __name__ == "__main__":
    main()
