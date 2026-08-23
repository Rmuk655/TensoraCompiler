#!/usr/bin/env python3
"""Heterogeneity sweep for ragged attention.

The plain ragged spike used one skewed batch. This shows the causal
curve: as the batch gets more heterogeneous (lengths more uneven), the
padding waste grows, the attention work eliminated grows, and the ragged
speedup grows. A uniform batch shows almost no gain; a very skewed batch
shows a large one. This is the honest "when does ragged help" picture.

Reuses the attention variants from spike.py.

Usage: python hetero_sweep.py
"""
import torch
from spike import make_qkv, flash_padded, ragged_loop, measure

CASES = {
    "A uniform      [128,128,128,128]": [128, 128, 128, 128],
    "B one short    [100,100,100,20] ": [100, 100, 100, 20],
    "C graded       [128,64,32,8]    ": [128, 64, 32, 8],
    "D very skewed  [512,64,16,4]    ": [512, 64, 16, 4],
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
    print("\nRead: gain tracks heterogeneity. A uniform batch gains ~nothing; "
          "a very\nskewed batch gains a lot. This is the regime where ragged "
          "(and Tensora) can win.")


if __name__ == "__main__":
    main()
