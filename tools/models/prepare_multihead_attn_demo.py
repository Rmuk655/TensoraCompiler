#!/usr/bin/env python3
"""Prepare the multi-head self-attention Associative Recall demo.

The rank-4, multi-head sibling of prepare_text_attn_demo.py (single-head
topic classification), but a different task chosen to give multi-head
attention something distinctive to demonstrate: Associative Recall
(AR) -- a standard synthetic benchmark in the mechanistic-
interpretability and long-context literature (Ba et al. 2016; Arora et
al.'s "Zoology", arXiv:2312.04927, which generalizes it to the
multi-query MQAR variant this demo does not implement -- see
examples/attn_multihead_retrieval.tns's header for the full citation),
and a one-hop version of the retrieval step behind real next-token
prediction.

Each example is a sequence of 6 (key, value) slots in random order
(the key->value binding is a random permutation, reshuffled every
example -- a fixed global lookup table cannot be memorized), followed
by one query token naming a key. Target: the value bound to the
queried key *in this sequence*. A slot is embedded as
concat(key_embedding, value_embedding); the query token is embedded as
concat(key_embedding, zeros) -- so a head that learns to compare "key
subspaces" and copy from the matching slot's "value subspace" solves
the task in one self-attention layer (see
examples/attn_multihead_retrieval.tns's header for the full
architecture, which this script mirrors op-for-op: batched matmul,
transpose, softmax, sum -- no reshape/concat/gather inside the part
PyTorch and Tensora must agree on).

Exports a batch of 30 held-out sequences:

  demos/artifacts/attn_mh/eval/
      Xq.npy Xk.npy Wq.npy Wk.npy Wv.npy Wo.npy Sel.npy
      Wc1.npy bc1.npy Wc2.npy bc2.npy
                          -> a self-contained --data-dir (Xq == Xk,
                             each a batch of 30 embedded sequences,
                             with the head=1 broadcast axis already
                             added; Sel is a fixed, untrained one-hot
                             selector -- see the .tns file's header)
      expected.npy        -> the PyTorch reference logits [30, 6]
      meta.json           -> the 30 sequences (readable key/value
                             names) and their true target values

The eval dir is directly usable as `tensora --run
examples/attn_multihead_retrieval.tns --data-dir <dir>`: one run
retrieves all 30 held-out queries and the demo compares every row
against PyTorch. Run once, on a machine with PyTorch (see
tools/models/requirements.txt):
    python3 tools/models/prepare_multihead_attn_demo.py
Safe to re-run: skips everything if demos/artifacts/attn_mh/eval/
already exists (pass --force to redo).
"""
import json
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("error: PyTorch is required. Install with: pip install -r tools/models/requirements.txt")

SEED = 0
NUM_SLOTS = 6           # one slot per key, all 6 keys present every example
SEQ = NUM_SLOTS + 1     # + 1 query token = 7, matching seq_q/seq_k in the .tns
KEY_DIM = 16
VALUE_DIM = 16
DIM = KEY_DIM + VALUE_DIM  # 32, matching d=32 in the .tns
HEADS = 4
HEAD_DIM = DIM // HEADS  # 8
SCALE = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(8); must match examples/attn_multihead_retrieval.tns
HID = 16
KEYS = ["red", "blue", "green", "yellow", "purple", "orange"]
VALUES = ["one", "two", "three", "four", "five", "six"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "attn_mh")

TRAIN_BATCHES = 800
BATCH = 64
HELDOUT = 30  # matching b=30 in the .tns


def make_example(rng):
    """One associative-recall example: 6 shuffled (key, value) slots,
    a random query key, and the value bound to it in this example."""
    key_order = rng.permutation(NUM_SLOTS)          # which key sits in each slot position
    value_binding = rng.permutation(NUM_SLOTS)      # key i (0..5) is bound to value_binding[i]
    slot_keys = key_order
    slot_values = value_binding[key_order]
    query_key = int(rng.integers(NUM_SLOTS))
    target_value = int(value_binding[query_key])
    return slot_keys, slot_values, query_key, target_value


class Retrieval(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_emb = nn.Embedding(NUM_SLOTS, KEY_DIM)
        self.val_emb = nn.Embedding(NUM_SLOTS, VALUE_DIM)
        # Per-head projection weights, carrying an explicit `head` axis
        # from the start -- exactly the layout
        # examples/attn_multihead_retrieval.tns declares (Tensor[b=1,
        # head=4, d=32, dk=8]). See prepare_multihead_attn_demo.py's
        # predecessor comment for why: training and export then agree
        # with the .tns program's own operations bit for bit (up to
        # float rounding).
        self.Wq = nn.Parameter(torch.empty(HEADS, DIM, HEAD_DIM))
        self.Wk = nn.Parameter(torch.empty(HEADS, DIM, HEAD_DIM))
        self.Wv = nn.Parameter(torch.empty(HEADS, DIM, HEAD_DIM))
        self.Wo = nn.Parameter(torch.empty(HEADS, HEAD_DIM, DIM))
        for w in (self.Wq, self.Wk, self.Wv, self.Wo):
            nn.init.xavier_uniform_(w)
        self.Wc1 = nn.Linear(DIM, HID)
        self.Wc2 = nn.Linear(HID, NUM_SLOTS)

    def embed(self, slot_keys, slot_values, query_key):
        # slot_keys/slot_values: [B, NUM_SLOTS] int64; query_key: [B] int64.
        slot_x = torch.cat([self.key_emb(slot_keys), self.val_emb(slot_values)], dim=-1)  # [B, 6, 32]
        qk = self.key_emb(query_key)                          # [B, 16]
        qv = torch.zeros_like(qk)                             # value unknown at query time
        query_x = torch.cat([qk, qv], dim=-1).unsqueeze(1)    # [B, 1, 32]
        return torch.cat([slot_x, query_x], dim=1)            # [B, 7, 32]

    def forward_from_x(self, x):
        # x: [B, SEQ, DIM]. Mirrors examples/attn_multihead_retrieval.tns
        # exactly: per-head projection in one batched matmul, scaled
        # dot-product attention, per-head output projection summed
        # over heads, then "select the last (query) position" via a
        # one-hot mask multiply + sum -- the same operation as
        # `merged[:, -1, :]` below, computed the .tns program's way so
        # both sides run an identical op sequence.
        Q = torch.einsum("bsd,hde->bhse", x, self.Wq)
        K = torch.einsum("bsd,hde->bhse", x, self.Wk)
        V = torch.einsum("bsd,hde->bhse", x, self.Wv)
        scores = torch.einsum("bhqe,bhke->bhqk", Q, K) * SCALE
        A = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bhqk,bhke->bhqe", A, V)          # [B, HEADS, SEQ, HEAD_DIM]
        proj = torch.einsum("bhqe,hed->bhqd", ctx, self.Wo)  # [B, HEADS, SEQ, DIM]
        merged = proj.sum(dim=1)                              # [B, SEQ, DIM]
        selected = merged[:, -1, :]                           # the query position's hidden state
        return self.Wc2(torch.relu(self.Wc1(selected)))

    def forward(self, slot_keys, slot_values, query_key):
        return self.forward_from_x(self.embed(slot_keys, slot_values, query_key))


def main():
    marker = os.path.join(OUT_DIR, "eval", "meta.json")
    if os.path.exists(marker) and "--force" not in sys.argv:
        print(f"Already prepared: {os.path.normpath(marker)} exists "
              "(delete demos/artifacts/attn_mh/, or pass --force, to redo)")
        return
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Retrieval()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(TRAIN_BATCHES):
        batch = [make_example(rng) for _ in range(BATCH)]
        slot_keys = torch.tensor(np.stack([b[0] for b in batch]))
        slot_values = torch.tensor(np.stack([b[1] for b in batch]))
        query_key = torch.tensor([b[2] for b in batch])
        target = torch.tensor([b[3] for b in batch])
        opt.zero_grad()
        loss = loss_fn(model(slot_keys, slot_values, query_key), target)
        loss.backward()
        opt.step()
    model.eval()

    # A fresh, larger batch for a reliable accuracy estimate (never
    # trained on, though the task's held-out/train distinction is
    # moot here -- every example is freshly random).
    check_batch = [make_example(rng) for _ in range(2000)]
    with torch.no_grad():
        sk = torch.tensor(np.stack([b[0] for b in check_batch]))
        sv = torch.tensor(np.stack([b[1] for b in check_batch]))
        qk = torch.tensor([b[2] for b in check_batch])
        tgt = np.array([b[3] for b in check_batch])
        acc = (model(sk, sv, qk).argmax(1).numpy() == tgt).mean()
    print(f"Trained associative-retrieval attention: accuracy on 2000 fresh examples = {acc:.3f}")

    with torch.no_grad():
        weights = {
            # Add the b=1 broadcast axis the .tns file declares.
            "Wq": model.Wq.numpy().reshape(1, HEADS, DIM, HEAD_DIM).astype(np.float32),
            "Wk": model.Wk.numpy().reshape(1, HEADS, DIM, HEAD_DIM).astype(np.float32),
            "Wv": model.Wv.numpy().reshape(1, HEADS, DIM, HEAD_DIM).astype(np.float32),
            "Wo": model.Wo.numpy().reshape(1, HEADS, HEAD_DIM, DIM).astype(np.float32),
            "Wc1": model.Wc1.weight.t().contiguous().numpy().astype(np.float32),
            "bc1": model.Wc1.bias.numpy().astype(np.float32),
            "Wc2": model.Wc2.weight.t().contiguous().numpy().astype(np.float32),
            "bc2": model.Wc2.bias.numpy().astype(np.float32),
        }
        # Fixed, untrained one-hot selector: 1.0 at the query position
        # (the last of the SEQ=7 positions), 0.0 elsewhere. See the
        # .tns file's header for why this replaces indexing.
        sel = np.zeros((1, SEQ, 1), dtype=np.float32)
        sel[0, -1, 0] = 1.0
        weights["Sel"] = sel

    heldout = [make_example(rng) for _ in range(HELDOUT)]
    sk = torch.tensor(np.stack([b[0] for b in heldout]))
    sv = torch.tensor(np.stack([b[1] for b in heldout]))
    qk = torch.tensor([b[2] for b in heldout])
    tgt = np.array([b[3] for b in heldout])
    with torch.no_grad():
        x = model.embed(sk, sv, qk).numpy().astype(np.float32)          # [N, SEQ, DIM]
        logits = model.forward_from_x(torch.tensor(x)).numpy().astype(np.float32)  # [N, 6]

    os.makedirs(OUT_DIR, exist_ok=True)
    d = os.path.join(OUT_DIR, "eval")
    os.makedirs(d, exist_ok=True)
    # Xq/Xk: add the head=1 broadcast axis the .tns file declares.
    x_h1 = x[:, None, :, :]  # [N, 1, SEQ, DIM]
    np.save(os.path.join(d, "Xq.npy"), x_h1)
    np.save(os.path.join(d, "Xk.npy"), x_h1)
    for name, arr in weights.items():
        np.save(os.path.join(d, f"{name}.npy"), arr)
    np.save(os.path.join(d, "expected.npy"), logits)

    sk_np, sv_np = sk.numpy(), sv.numpy()
    meta = {
        "count": HELDOUT,
        "sequences": [
            {
                "slots": [f"{KEYS[k]}:{VALUES[v]}" for k, v in zip(sk_np[i], sv_np[i])],
                "query_key": KEYS[int(qk[i])],
                "true_value": VALUES[int(tgt[i])],
            }
            for i in range(HELDOUT)
        ],
    }
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    pt_acc = float((logits.argmax(1) == tgt).mean())
    print(f"Wrote {HELDOUT}-sequence eval batch to {os.path.normpath(OUT_DIR)} "
          f"(PyTorch accuracy on this batch = {pt_acc:.3f})")


if __name__ == "__main__":
    main()
