#!/usr/bin/env python3
"""Does the L'=f(L) extent-transfer model actually cover ML? (closure test)

The proposed IR assumes each operator has an extent-transfer function
L_out = f(L_in). That is only true for some operators. This classifies
real ops and TESTS the classification:

  A  extent unchanged            (relu, gelu, softmax over feature)
  B  L_out = f(L_in)             (conv, pool, downsample, flatten)
  C  L_out = f(L_in, metadata)   (concat, pad, gather -- needs an extra
                                  known array/param, not just L_in)
  D  L_out depends on DATA VALUES (nonzero, boolean_mask, topk-dynamic,
                                  unique) -- NOT a function of extents

If a large fraction of real ops are D, the f(L) model covers only a
"nice" sub-language, and that is a scientific limit worth knowing.

Usage: python operator_taxonomy.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_B():
    print("B (L_out = f(L_in)) -- verify transfer function vs real output:")
    conv = nn.Conv1d(4, 4, 5, stride=2, padding=2)
    pool = lambda x: F.max_pool1d(x, 2)
    ok = True
    for L in [7, 13, 40, 128]:
        x = torch.randn(1, 4, L)
        rc = conv(x).shape[-1];  fc = (L + 2*2 - 5)//2 + 1
        rp = pool(x).shape[-1];  fp = L // 2
        ok = ok and rc == fc and rp == fp
    print(f"   conv1d, maxpool: transfer functions {'VERIFIED' if ok else 'WRONG'}")
    # concat is B across two extents: L_out = L_a + L_b
    a, b = torch.randn(1, 4, 10), torch.randn(1, 4, 25)
    print(f"   concat: L_out={torch.cat([a,b],-1).shape[-1]} == 10+25 "
          f"{'VERIFIED' if torch.cat([a,b],-1).shape[-1]==35 else 'WRONG'}\n")


def check_C():
    print("C (L_out = f(L_in, metadata)) -- needs an extra known array:")
    table = torch.randn(100, 8)
    idx = torch.tensor([3, 7, 1, 42])          # metadata: the index array
    out = table[idx]
    print(f"   gather: L_out={out.shape[0]} == len(indices)={idx.numel()} "
          f"(depends on the index array, not the table extent)\n")


def check_D():
    print("D (L_out depends on DATA VALUES) -- NOT a function of extents:")
    # same shape, different data -> different output extent
    x1 = torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0])
    x2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    n1, n2 = torch.nonzero(x1).shape[0], torch.nonzero(x2).shape[0]
    print(f"   nonzero: same input shape 5, output extent {n1} vs {n2} "
          f"(data-dependent)")
    m1 = x1[x1 > 2]; m2 = x2[x2 > 2]
    print(f"   boolean_mask: output extent {m1.numel()} vs {m2.numel()} "
          f"(data-dependent)")
    print("   -> f(L_in) cannot express these; they need L_out as a runtime")
    print("      VALUE produced by the op, not derived from input extents.\n")


def main():
    print("Operator taxonomy for the extent-transfer model:\n")
    check_B(); check_C(); check_D()
    print("Classification of ~common ops:")
    rows = [
        ("relu / gelu / softmax(feat)", "A", "extent unchanged"),
        ("conv / pool / stride", "B", "L_out = f(L_in) -- verified above"),
        ("flatten / reshape(fixed)", "B", "static reassociation"),
        ("concat", "B", "L_out = sum of input extents"),
        ("pad / slice(fixed)", "B/C", "f(L_in, pad/slice params)"),
        ("gather / embedding", "C", "L_out = len(index array)"),
        ("cumsum over seq", "A", "extent unchanged, but scan dependence"),
        ("topk (static k)", "A", "k is static"),
        ("topk (dynamic k)", "C/D", "k from metadata or data"),
        ("nonzero / boolean_mask / filter", "D", "data-dependent extent"),
        ("unique / compaction / MoE routing", "D", "data-dependent extent"),
    ]
    print(f"   {'op':34} {'class':7} note")
    for op, c, note in rows:
        print(f"   {op:34} {c:7} {note}")
    print("\nRead: A/B/C cover the feed-forward + conv + attention + gather core")
    print("(the f(L)/f(L,meta) model works). D ops (nonzero, filter, MoE routing,")
    print("dynamic decode) are genuinely data-dependent -- the extent is a runtime")
    print("VALUE, not a function of extents. So the abstraction covers a large,")
    print("useful sub-language but NOT all of ML. That boundary (B/C vs D) is the")
    print("honest scope of the proposed contract system.")


if __name__ == "__main__":
    main()
