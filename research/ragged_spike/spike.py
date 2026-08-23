#!/usr/bin/env python3
"""Ragged-attention measurement spike.

Goal: find out if there is a real memory or speed gap between padded
attention and ragged (no-padding) attention on a skewed batch, before
building any tool. This is a go/no-go for the V0.2 direction. It uses
only core PyTorch (no flash_attn, no vllm), so it measures what a stock
user actually has.

It runs one self-attention workload four ways on the same skewed batch:

  1. eager_padded   -- naive attention: builds the full [B,H,L,L] score
                       matrix on padded data. Many hand-written attention
                       blocks do this.
  2. flash_padded   -- torch's fused scaled_dot_product_attention on
                       padded data. This is PyTorch's OWN good kernel;
                       it does not build the full score matrix.
  3. ragged_nested  -- scaled_dot_product_attention on a jagged
                       NestedTensor. This is PyTorch's OWN ragged path,
                       if this build supports it.
  4. ragged_loop    -- one SDPA call per sequence, no padding at all. A
                       robust ragged baseline that always runs.

For each variant it reports peak GPU memory and latency, and checks the
ragged variants match the padded result on the real (non-pad) positions.
A sweep mode grows the batch until each variant runs out of memory, and
prints the crossover point (where padded fails but ragged still fits).

Usage:
  python spike.py                 # one config, all four variants
  python spike.py --sweep         # grow batch until OOM; print crossover
"""
import argparse
import statistics
import sys

import torch
import torch.nn.functional as F


def gen_lengths(batch, maxlen, seed):
    # A skewed batch: most sequences short, a few long. This is the real
    # case padding wastes the most on. The longest is pinned to maxlen so
    # the padded shape is exactly [batch, maxlen].
    g = torch.Generator().manual_seed(seed)
    short = torch.randint(8, 64, (batch,), generator=g)
    short[0] = maxlen  # one full-length sequence forces the padded size
    # make a few more long, to be realistic (not a single outlier)
    n_long = max(1, batch // 8)
    idx = torch.randperm(batch, generator=g)[:n_long]
    short[idx] = torch.randint(maxlen // 2, maxlen + 1, (n_long,), generator=g)
    return short.tolist()


def waste_factor(lengths, maxlen):
    padded_slots = len(lengths) * maxlen
    real_slots = sum(lengths)
    return padded_slots / real_slots, padded_slots, real_slots


def make_qkv(lengths, heads, head_dim, dtype, device, seed=0):
    # Per-sequence Q, K, V as a list of [len_i, heads, head_dim] tensors.
    g = torch.Generator(device="cpu").manual_seed(seed)
    qs, ks, vs = [], [], []
    for L in lengths:
        q = torch.randn(L, heads, head_dim, generator=g).to(device=device, dtype=dtype)
        k = torch.randn(L, heads, head_dim, generator=g).to(device=device, dtype=dtype)
        v = torch.randn(L, heads, head_dim, generator=g).to(device=device, dtype=dtype)
        qs.append(q); ks.append(k); vs.append(v)
    return qs, ks, vs


def pad_stack(seqs, maxlen):
    # Stack a list of [len_i, H, D] into [B, H, maxlen, D], zero-padded,
    # plus a [B, maxlen] key-padding mask (True = real token).
    B = len(seqs)
    L = maxlen
    H, D = seqs[0].shape[1], seqs[0].shape[2]
    out = seqs[0].new_zeros(B, L, H, D)
    mask = torch.zeros(B, L, dtype=torch.bool, device=seqs[0].device)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = True
    return out.transpose(1, 2).contiguous(), mask  # [B, H, L, D], [B, L]


def eager_padded(qs, ks, vs, maxlen):
    q, kmask = pad_stack(qs, maxlen)
    k, _ = pad_stack(ks, maxlen)
    v, _ = pad_stack(vs, maxlen)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,L,L] -- the waste
    # mask out padded keys
    keymask = kmask[:, None, None, :]  # [B,1,1,L]
    scores = scores.masked_fill(~keymask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn)  # rows that are all -inf (padded queries)
    return torch.matmul(attn, v)  # [B,H,L,D]


def flash_padded(qs, ks, vs, maxlen):
    q, kmask = pad_stack(qs, maxlen)
    k, _ = pad_stack(ks, maxlen)
    v, _ = pad_stack(vs, maxlen)
    attn_mask = kmask[:, None, None, :].expand(q.shape[0], q.shape[1], q.shape[2], q.shape[2])
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def ragged_nested(qs, ks, vs, maxlen):
    # PyTorch's own ragged path: jagged NestedTensor into SDPA.
    def nest(seqs):
        # each seq is [L, H, D]; want nested [B, H, jagged, D]
        per = [s.transpose(0, 1) for s in seqs]  # [H, L, D]
        nt = torch.nested.nested_tensor(per, layout=torch.jagged)
        return nt
    q, k, v = nest(qs), nest(ks), nest(vs)
    return F.scaled_dot_product_attention(q, k, v)


def ragged_loop(qs, ks, vs, maxlen):
    outs = []
    for q, k, v in zip(qs, ks, vs):
        # [L,H,D] -> [1,H,L,D]
        qh = q.transpose(0, 1).unsqueeze(0)
        kh = k.transpose(0, 1).unsqueeze(0)
        vh = v.transpose(0, 1).unsqueeze(0)
        outs.append(F.scaled_dot_product_attention(qh, kh, vh))
    return outs  # list of [1,H,Li,D]


VARIANTS = {
    "eager_padded": eager_padded,
    "flash_padded": flash_padded,
    "ragged_nested": ragged_nested,
    "ragged_loop": ragged_loop,
}


def measure(fn, qs, ks, vs, maxlen, device, iters=5):
    # Returns (peak_mem_bytes, median_ms) or raises OOM.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # warmup
    out = fn(qs, ks, vs, maxlen)
    if device.type == "cuda":
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    # timing
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        out = fn(qs, ks, vs, maxlen)
        if device.type == "cuda":
            t1.record(); torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
    return peak, (statistics.median(times) if times else float("nan")), out


def real_positions_output(out, lengths, variant):
    # Extract each sequence's real (non-pad) attention output as a flat
    # [sum(Li), H, D] tensor, so variants can be compared for correctness.
    if variant in ("eager_padded", "flash_padded"):
        rows = [out[i, :, : L, :].transpose(0, 1) for i, L in enumerate(lengths)]  # [Li,H,D]
        return torch.cat(rows, dim=0).float()
    if variant == "ragged_loop":
        rows = [o[0].transpose(0, 1) for o in out]  # [Li,H,D]
        return torch.cat(rows, dim=0).float()
    if variant == "ragged_nested":
        # out is nested [B,H,jagged,D]; unbind and reshape
        rows = []
        for i, L in enumerate(lengths):
            t = out[i]  # [H, Li, D]
            rows.append(t.transpose(0, 1))  # [Li,H,D]
        return torch.cat(rows, dim=0).float()
    raise ValueError(variant)


def fmt_mb(b):
    return f"{b / (1024**2):8.1f} MB"


def run_once(args, device):
    lengths = gen_lengths(args.batch, args.maxlen, args.seed)
    wf, padded_slots, real_slots = waste_factor(lengths, args.maxlen)
    dtype = getattr(torch, args.dtype)
    print(f"\nconfig: batch={args.batch} maxlen={args.maxlen} heads={args.heads} "
          f"head_dim={args.head_dim} dtype={args.dtype} device={device.type}")
    print(f"skewed lengths: min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths)/len(lengths):.0f}")
    print(f"padding waste: {padded_slots} padded token slots vs {real_slots} real "
          f"tokens = {wf:.2f}x")
    print()
    print(f"{'variant':16} {'peak memory':>12} {'latency':>12}   {'result'}")
    print("-" * 60)

    ref = None
    for name, fn in VARIANTS.items():
        qs, ks, vs = make_qkv(lengths, args.heads, args.head_dim, dtype, device)
        try:
            peak, ms, out = measure(fn, qs, ks, vs, args.maxlen, device, args.iters)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).split("\n")[0][:40]
            print(f"{name:16} {'OOM/err':>12} {'-':>12}   {msg}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        # correctness vs first successful padded variant
        note = ""
        try:
            got = real_positions_output(out, lengths, name)
            if ref is None and name in ("eager_padded", "flash_padded"):
                ref = got
                note = "(reference)"
            elif ref is not None:
                maxerr = (got - ref).abs().max().item()
                note = f"max err vs ref {maxerr:.1e}"
        except Exception as e:
            note = f"(no check: {str(e)[:30]})"
        print(f"{name:16} {fmt_mb(peak):>12} {ms:9.2f} ms   {note}")
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_sweep(args, device):
    # We use a memory BUDGET, not a real hard OOM. On WSL2, CUDA spills
    # to shared system RAM instead of failing cleanly, then hard-crashes
    # much later -- so "peak > budget" is both more robust and the honest
    # question for an edge device ("does it fit in X GB?").
    budget_gb = args.mem_budget_gb
    if budget_gb <= 0 and device.type == "cuda":
        budget_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    budget_b = budget_gb * (1024**3)
    print(f"\nsweep: growing batch; a variant 'fails' when peak memory exceeds "
          f"the {budget_gb:.1f} GB budget")
    print(f"(fixed maxlen={args.maxlen} heads={args.heads} head_dim={args.head_dim} "
          f"dtype={args.dtype})\n")
    dtype = getattr(torch, args.dtype)
    names = list(VARIANTS.keys())
    alive = {n: True for n in names}
    first_fail = {n: None for n in names}
    header = f"{'batch':>6} " + " ".join(f"{n:>14}" for n in names)
    print(header)
    print("-" * len(header))
    batch = args.sweep_start
    while batch <= args.sweep_max and any(alive.values()):
        lengths = gen_lengths(batch, args.maxlen, args.seed)
        cells = []
        for n in names:
            if not alive[n]:
                cells.append(f"{'-':>14}")
                continue
            fn = VARIANTS[n]
            qs, ks, vs = make_qkv(lengths, args.heads, args.head_dim, dtype, device)
            peak = None
            try:
                peak, ms, out = measure(fn, qs, ks, vs, args.maxlen, device, iters=1)
                del out
            except Exception:
                # any allocation failure: this variant cannot run here
                alive[n] = False
                first_fail[n] = batch
                cells.append(f"{'OOM':>14}")
            if peak is not None:
                over = peak > budget_b
                cells.append(f"{peak/(1024**2):8.0f}MB" + ("!" if over else " "))
                if over:
                    alive[n] = False
                    first_fail[n] = batch
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"{batch:>6} " + " ".join(f"{c:>14}" for c in cells))
        # stop before the WSL hard-crash regime: once both padded variants
        # are out, the point is made.
        if not alive["eager_padded"] and not alive["flash_padded"]:
            break
        batch = int(batch * args.sweep_mul)
    print("\nfirst over-budget batch per variant ('!' = exceeded budget):")
    for n in names:
        print(f"  {n:16} {first_fail[n] if first_fail[n] else 'never (within sweep)'}")
    for padded in ("eager_padded", "flash_padded"):
        for ragged in ("ragged_nested", "ragged_loop"):
            if first_fail[padded] and (first_fail[ragged] is None or first_fail[ragged] > first_fail[padded]):
                print(f"\nCROSSOVER: {padded} exceeds the {budget_gb:.1f} GB budget at "
                      f"batch {first_fail[padded]}, but {ragged} still fits there.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-start", type=int, default=8)
    ap.add_argument("--sweep-max", type=int, default=4096)
    ap.add_argument("--sweep-mul", type=float, default=2.0)
    ap.add_argument("--mem-budget-gb", type=float, default=0.0,
                    help="fail a variant when peak memory exceeds this; 0 = use total VRAM")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB)")
    else:
        print("running on CPU (no peak-memory numbers)")

    if args.sweep:
        run_sweep(args, device)
    else:
        run_once(args, device)


if __name__ == "__main__":
    main()
