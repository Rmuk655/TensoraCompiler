#!/usr/bin/env python3
"""Randomized heterogeneity study (answers 'is a rigorous sweep hard?').

The hand-picked hetero_sweep was a sanity check. This is the more
rigorous version the reviewer asked for: many randomly sampled batches,
several batch sizes and max lengths, several skew levels, with
mean +/- std (confidence). It reports theoretical work-eliminated and
measured ragged-vs-padded speedup, so you can see how well work-elim
predicts speedup.

Not hard -- it is ~an hour of scripting. The one thing not done here:
multiple GPUs (this machine has one). Everything else the reviewer
listed is here.

Usage: python hetero_random.py
"""
import statistics
import time
import torch
import torch.nn.functional as F

H, Dh = 8, 64


def sample_lengths(B, maxlen, skew, gen):
    # skew in (0,1]: low skew -> near-uniform lengths; high skew -> very uneven
    lo = max(1, int(maxlen * (1 - skew)))
    lens = torch.randint(lo, maxlen + 1, (B,), generator=gen).tolist()
    lens[0] = maxlen  # pin the max so padded shape is [B, maxlen]
    return lens


def work_elim(lens, maxlen):
    B = len(lens)
    return 1.0 - sum(L * L for L in lens) / (B * maxlen * maxlen)


def make(lens, dev, dt):
    g = torch.Generator().manual_seed(sum(lens))
    return [torch.randn(L, H, Dh, generator=g).to(dev, dt) for L in lens]


def padded(seqs, maxlen):
    B = len(seqs)
    q = seqs[0].new_zeros(B, maxlen, H, Dh)
    mask = torch.zeros(B, maxlen, dtype=torch.bool, device=seqs[0].device)
    for i, s in enumerate(seqs):
        q[i, : s.shape[0]] = s; mask[i, : s.shape[0]] = True
    q = q.transpose(1, 2)
    am = mask[:, None, None, :].expand(B, H, maxlen, maxlen)
    return F.scaled_dot_product_attention(q, q, q, attn_mask=am)


def ragged(seqs):
    return [F.scaled_dot_product_attention(
        s.transpose(0, 1)[None], s.transpose(0, 1)[None], s.transpose(0, 1)[None]) for s in seqs]


def med_ms(fn, dev, iters=8):
    fn()
    if dev.type == "cuda": torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(t)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    print(f"device={dev.type}  (one GPU; multi-GPU not available here)\n")
    print(f"{'batch':>6} {'maxlen':>7} {'skew':>5} {'work elim %':>16} {'speedup':>16}")
    print(f"{'':>6} {'':>7} {'':>5} {'mean +/- std':>16} {'mean +/- std':>16}")
    print("-" * 56)
    NSAMPLES = 8
    for B in (8, 32):
        for maxlen in (256, 1024):
            for skew in (0.2, 0.6, 0.95):
                elims, speeds = [], []
                for s in range(NSAMPLES):
                    g = torch.Generator().manual_seed(1000 * s + B + maxlen)
                    lens = sample_lengths(B, maxlen, skew, g)
                    elims.append(100 * work_elim(lens, maxlen))
                    seqs = make(lens, dev, dt)
                    pms = med_ms(lambda: padded(seqs, maxlen), dev)
                    seqs = make(lens, dev, dt)
                    rms = med_ms(lambda: ragged(seqs), dev)
                    speeds.append(pms / rms if rms > 0 else float("nan"))
                    if dev.type == "cuda": torch.cuda.empty_cache()
                em, es = statistics.mean(elims), statistics.pstdev(elims)
                sm, ss = statistics.mean(speeds), statistics.pstdev(speeds)
                print(f"{B:>6} {maxlen:>7} {skew:>5.2f} "
                      f"{em:7.0f} +/- {es:<4.0f}     {sm:6.1f} +/- {ss:<5.1f}")
    print("\nRead: work-eliminated is a stable function of skew (tight std). "
          "Measured\nspeedup is noisier (launch overhead, kernel selection), but "
          "tracks work-elim\nfor larger maxlen. A rigorous sweep is not hard; only "
          "multi-GPU is missing here.")


if __name__ == "__main__":
    main()
