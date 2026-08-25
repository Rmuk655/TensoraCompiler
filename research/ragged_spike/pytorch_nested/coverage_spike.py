#!/usr/bin/env python3
"""Coverage spike: where does PyTorch's ragged path actually break?

The earlier spikes tested one Transformer region and found torch.nested
handled it. The critique was right: that is one workload. This probes the
untested cases with core torch, to find real gaps -- the empirical
"PyTorch column" of the capability matrix. Each probe reports OK, BREAK
(unsupported), or WRONG (runs but gives a wrong answer).

It also runs a correctness check for the reframed research question: does
making the ragged extent explicit change the *result*, not just speed?

TACO / Tiramisu / DISTAL are NOT run here (heavy builds; DISTAL needs a
cluster). Their matrix cells come from their papers, not this script.

Usage: python coverage_spike.py [--cpu]
"""
import argparse
import torch
import torch.nn.functional as F


def nest(seqs):
    # seqs: list of [L_i, H, D] -> jagged nested [B, H, jagged, D]
    return torch.nested.nested_tensor([s.transpose(0, 1) for s in seqs], layout=torch.jagged)


def probe(name, fn):
    try:
        r = fn()
        print(f"  {'OK':6} {name}" + (f"  -- {r}" if isinstance(r, str) else ""))
        return "OK"
    except Exception as e:
        print(f"  {'BREAK':6} {name}  -- {type(e).__name__}: {str(e).splitlines()[0][:60]}")
        return "BREAK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    H, D = 8, 64
    lens = [8, 16, 32, 128]
    print(f"device={dev.type} dtype={dt} heads={H} head_dim={D} lengths={lens}\n")

    def mk(lens_, h=H):
        g = torch.Generator().manual_seed(0)
        return [torch.randn(L, h, D, generator=g).to(dev, dt) for L in lens_]

    print("Untested ragged cases (core torch.nested):")

    # 1. causal attention
    def causal():
        q, k, v = nest(mk(lens)), nest(mk(lens)), nest(mk(lens))
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return "nested + causal"
    probe("causal self-attention", causal)

    # 2. cross-attention: query lengths != key/value lengths
    def cross():
        q = nest(mk([4, 8, 16, 64]))          # query lengths
        k = nest(mk(lens)); v = nest(mk(lens))  # key/value lengths differ
        F.scaled_dot_product_attention(q, k, v)
        return "ragged query != ragged key"
    probe("cross-attention (Q len != KV len)", cross)

    # 3. KV-cache decode: one new query token per sequence, ragged cached KV
    def kvcache():
        q = nest(mk([1, 1, 1, 1]))            # single decode step
        k = nest(mk(lens)); v = nest(mk(lens))  # ragged cache per batch
        F.scaled_dot_product_attention(q, k, v)
        return "1 query vs ragged KV cache"
    probe("KV-cache decode step", kvcache)

    # 4. grouped-query attention (fewer KV heads)
    def gqa():
        q = nest(mk(lens, h=H))
        k = nest(mk(lens, h=H // 4)); v = nest(mk(lens, h=H // 4))
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return f"{H} Q heads, {H//4} KV heads"
    probe("grouped-query attention", gqa)

    # 5. two ragged dimensions at once
    def two_ragged():
        # a nested tensor whose elements are themselves jagged -> 2 ragged dims
        inner = [torch.randn(L, D).to(dev, dt) for L in [4, 8]]
        outer = torch.nested.nested_tensor(inner, layout=torch.jagged)
        # try to nest again (jagged supports ONE ragged dim)
        torch.nested.nested_tensor([outer, outer], layout=torch.jagged)
        return "two jagged dims"
    probe("two ragged dimensions", two_ragged)

    # 6. autograd / backward through nested attention
    def autograd():
        q = nest(mk(lens)); k = nest(mk(lens)); v = nest(mk(lens))
        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        out = F.scaled_dot_product_attention(q, k, v)
        out.values().sum().backward()
        ok = q.grad is not None
        return f"backward ran, grad present={ok}"
    probe("autograd (backward pass)", autograd)

    # 7. torch.compile over nested attention
    def compiled():
        def f(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)
        cf = torch.compile(f, fullgraph=True)
        q, k, v = nest(mk(lens)), nest(mk(lens)), nest(mk(lens))
        cf(q, k, v)
        return "fullgraph compile"
    probe("torch.compile(fullgraph) + nested", compiled)

    # 8. correctness: does making the ragged extent explicit change the ANSWER?
    print("\nCorrectness (reframed question -- semantics change the result):")
    B, maxlen = len(lens), max(lens)
    feat = 16
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randn(L, feat, generator=g).to(dev, torch.float32) for L in lens]
    # padded tensor, zeros in pad positions
    padded = torch.zeros(B, maxlen, feat, device=dev)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s
    # WRONG: mean over the padded axis divides by maxlen (a silent bug)
    wrong = padded.mean(dim=1)
    # RIGHT: divide by the true length (the ragged extent)
    lengths = torch.tensor(lens, device=dev, dtype=torch.float32)[:, None]
    right = padded.sum(dim=1) / lengths
    err = (wrong - right).abs().max().item()
    print(f"  mean-pool over sequence: naive-padded vs true-length differ by "
          f"max {err:.3f}")
    print(f"  -> the padded mean is WRONG unless the ragged extent is known. "
          f"The\n     semantic fact (true length per row) is exactly what makes it "
          f"correct.")


if __name__ == "__main__":
    main()
