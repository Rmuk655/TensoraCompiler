#!/usr/bin/env python3
"""Fused-region ragged spike.

The plain-attention spike (spike.py) showed PyTorch's own NestedTensor
path already captures the ragged win for a single attention op. So the
real question for Tensora is: does that ragged path survive a whole
CUSTOM FUSED REGION, or does it break and force a fall back to padding?

Region tested (a realistic transformer block tail):

    attention  ->  RMSNorm  ->  Linear(D, 4D)  ->  GELU  ->  Linear(4D, D)  ->  + residual

Three variants on the same skewed batch:

  1. fused_padded      -- whole region on padded dense tensors (the
                          stock fallback). The [B, L, 4D] hidden is the
                          big padded intermediate.
  2. fused_nested      -- try to keep a jagged NestedTensor through the
                          WHOLE region. Report the first op that does not
                          support it (this is the gap Tensora targets).
  3. fused_ragged_loop -- per-sequence, no padding, whole region. What a
                          ragged-aware compiler could achieve.

Reports peak memory and latency, and where the nested path breaks.

Usage:
  python fused_spike.py
  python fused_spike.py --sweep
"""
import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_lengths(batch, maxlen, seed):
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(8, 64, (batch,), generator=g)
    lens[0] = maxlen
    n_long = max(1, batch // 8)
    idx = torch.randperm(batch, generator=g)[:n_long]
    lens[idx] = torch.randint(maxlen // 2, maxlen + 1, (n_long,), generator=g)
    return lens.tolist()


class Weights:
    # Shared region weights, so all variants compute the same function.
    def __init__(self, d_model, d_ff, dtype, device, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.norm_w = torch.randn(d_model, generator=g).to(device, dtype)
        self.w1 = (torch.randn(d_ff, d_model, generator=g) * (d_model ** -0.5)).to(device, dtype)
        self.b1 = torch.zeros(d_ff).to(device, dtype)
        self.w2 = (torch.randn(d_model, d_ff, generator=g) * (d_ff ** -0.5)).to(device, dtype)
        self.b2 = torch.zeros(d_model).to(device, dtype)
        self.eps = 1e-5


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def ffn_tail(x, W):
    # x: [..., D]  ->  region output [..., D]
    h = rmsnorm(x, W.norm_w, W.eps)
    h = F.linear(h, W.w1, W.b1)
    h = F.gelu(h)
    h = F.linear(h, W.w2, W.b2)
    return x + h


def make_qkv(lengths, heads, head_dim, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qs, ks, vs = [], [], []
    for L in lengths:
        qs.append(torch.randn(L, heads, head_dim, generator=g).to(device, dtype))
        ks.append(torch.randn(L, heads, head_dim, generator=g).to(device, dtype))
        vs.append(torch.randn(L, heads, head_dim, generator=g).to(device, dtype))
    return qs, ks, vs


def pad_stack(seqs, maxlen):
    B, L = len(seqs), maxlen
    H, D = seqs[0].shape[1], seqs[0].shape[2]
    out = seqs[0].new_zeros(B, L, H, D)
    mask = torch.zeros(B, L, dtype=torch.bool, device=seqs[0].device)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = True
    return out.transpose(1, 2).contiguous(), mask


def fused_padded(qs, ks, vs, maxlen, W, heads, head_dim):
    q, kmask = pad_stack(qs, maxlen)
    k, _ = pad_stack(ks, maxlen)
    v, _ = pad_stack(vs, maxlen)
    am = kmask[:, None, None, :].expand(q.shape[0], q.shape[1], q.shape[2], q.shape[2])
    ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=am)  # [B,H,L,Dh]
    B, H, L, Dh = ctx.shape
    x = ctx.transpose(1, 2).reshape(B, L, H * Dh)  # [B,L,D]
    return ffn_tail(x, W)  # [B,L,D] (padded rows included)


def fused_nested(qs, ks, vs, maxlen, W, heads, head_dim):
    # Attempt the whole region on a jagged NestedTensor. Return
    # (output, None) on success, or (None, step_name) at the first op
    # that does not support the nested layout.
    def nest(seqs):
        return torch.nested.nested_tensor([s.transpose(0, 1) for s in seqs], layout=torch.jagged)
    try:
        q, k, v = nest(qs), nest(ks), nest(vs)
    except Exception as e:
        return None, f"nested_tensor build ({type(e).__name__})"
    try:
        ctx = F.scaled_dot_product_attention(q, k, v)  # nested [B,H,jagged,Dh]
    except Exception as e:
        return None, f"attention ({type(e).__name__})"
    try:
        x = ctx.transpose(1, 2)  # [B, jagged, H, Dh]
        x = x.flatten(2)         # [B, jagged, D]  -- merge heads on jagged
    except Exception as e:
        return None, f"merge heads / reshape ({type(e).__name__})"
    try:
        h = rmsnorm(x, W.norm_w, W.eps)  # reduction over feature dim on jagged
    except Exception as e:
        return None, f"rmsnorm ({type(e).__name__})"
    try:
        h = F.linear(h, W.w1, W.b1)
        h = F.gelu(h)
        h = F.linear(h, W.w2, W.b2)
        out = x + h
    except Exception as e:
        return None, f"ffn linear/gelu ({type(e).__name__})"
    return out, None


def fused_ragged_loop(qs, ks, vs, maxlen, W, heads, head_dim):
    outs = []
    for q, k, v in zip(qs, ks, vs):
        qh = q.transpose(0, 1).unsqueeze(0)  # [1,H,L,Dh]
        kh = k.transpose(0, 1).unsqueeze(0)
        vh = v.transpose(0, 1).unsqueeze(0)
        ctx = F.scaled_dot_product_attention(qh, kh, vh)  # [1,H,L,Dh]
        _, H, L, Dh = ctx.shape
        x = ctx.transpose(1, 2).reshape(1, L, H * Dh)
        outs.append(ffn_tail(x, W))
    return outs


def measure(fn, args_tuple, device, iters=5):
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    out = fn(*args_tuple)
    if device.type == "cuda":
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        out = fn(*args_tuple)
        if device.type == "cuda":
            t1.record(); torch.cuda.synchronize(); times.append(t0.elapsed_time(t1))
    return peak, (statistics.median(times) if times else float("nan")), out


def run_once(args, device):
    lengths = gen_lengths(args.batch, args.maxlen, args.seed)
    padded_slots = len(lengths) * args.maxlen
    real_slots = sum(lengths)
    dtype = getattr(torch, args.dtype)
    d_model = args.heads * args.head_dim
    W = Weights(d_model, 4 * d_model, dtype, device)
    print(f"\nregion: attention -> RMSNorm -> Linear({d_model},{4*d_model}) -> GELU "
          f"-> Linear({4*d_model},{d_model}) -> +residual")
    print(f"config: batch={args.batch} maxlen={args.maxlen} heads={args.heads} "
          f"head_dim={args.head_dim} d_model={d_model} dtype={args.dtype}")
    print(f"padding waste: {padded_slots} slots vs {real_slots} real tokens "
          f"= {padded_slots/real_slots:.2f}x\n")
    print(f"{'variant':18} {'peak memory':>12} {'latency':>12}   note")
    print("-" * 66)

    ref = None
    # padded
    qs, ks, vs = make_qkv(lengths, args.heads, args.head_dim, dtype, device)
    peak, ms, out = measure(fused_padded, (qs, ks, vs, args.maxlen, W, args.heads, args.head_dim), device, args.iters)
    ref = torch.cat([out[i, :L].float() for i, L in enumerate(lengths)], 0)
    print(f"{'fused_padded':18} {peak/(1024**2):8.1f} MB {ms:9.2f} ms   (reference)")
    del out
    if device.type == "cuda": torch.cuda.empty_cache()

    # nested
    qs, ks, vs = make_qkv(lengths, args.heads, args.head_dim, dtype, device)
    try:
        peak, ms, out = measure(lambda *a: fused_nested(*a)[0] if fused_nested(*a)[1] is None else None,
                                (qs, ks, vs, args.maxlen, W, args.heads, args.head_dim), device, 1)
        _, breakstep = fused_nested(qs, ks, vs, args.maxlen, W, args.heads, args.head_dim)
        if breakstep is not None:
            print(f"{'fused_nested':18} {'BREAKS':>12} {'-':>12}   nested path fails at: {breakstep}")
        else:
            got = torch.cat([out[i].float() for i in range(len(lengths))], 0)
            err = (got - ref).abs().max().item()
            print(f"{'fused_nested':18} {peak/(1024**2):8.1f} MB {ms:9.2f} ms   max err {err:.1e}")
    except Exception as e:
        print(f"{'fused_nested':18} {'BREAKS':>12} {'-':>12}   {type(e).__name__}: {str(e)[:34]}")
    if device.type == "cuda": torch.cuda.empty_cache()

    # ragged loop
    qs, ks, vs = make_qkv(lengths, args.heads, args.head_dim, dtype, device)
    peak, ms, out = measure(fused_ragged_loop, (qs, ks, vs, args.maxlen, W, args.heads, args.head_dim), device, args.iters)
    got = torch.cat([o[0].float() for o in out], 0)
    err = (got - ref).abs().max().item()
    print(f"{'fused_ragged_loop':18} {peak/(1024**2):8.1f} MB {ms:9.2f} ms   max err {err:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} ({p.total_memory/(1024**3):.1f} GB)")
    else:
        print("running on CPU (no peak-memory numbers)")
    run_once(args, device)


if __name__ == "__main__":
    main()
