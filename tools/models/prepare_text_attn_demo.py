#!/usr/bin/env python3
"""Prepare the self-attention text-classifier real-inference demo.

Trains a small single-head self-attention classifier once in PyTorch on
a generated 3-topic corpus (weather / food / sports), using ONLY the
operators Tensora supports (matmul, softmax, transpose, mean, linear --
no layernorm or GELU), so PyTorch and Tensora compute the same function
and agree within tolerance. Then exports a batch of 30 held-out
sentences:

  demos/artifacts/attn/eval/
      Xq.npy Xk.npy Wq.npy Wk.npy Wv.npy Wc1.npy bc1.npy Wc2.npy bc2.npy
                          -> a self-contained --data-dir (Xq == Xk,
                             each a batch of 30 sentence embeddings)
      expected.npy        -> the PyTorch reference logits [30, 3]
      meta.json           -> the 30 sentences and their true topics

The eval dir is directly usable as `tensora --run
examples/attn_text_classifier.tns --data-dir <dir>`: one run classifies
all 30 held-out sentences and the demo compares every row against
PyTorch. Run once, on a machine with PyTorch (see
tools/models/requirements.txt):
    python3 tools/models/prepare_text_attn_demo.py
"""
import json
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("error: PyTorch is required. Install with: pip install -r tools/models/requirements.txt")

SEED = 0
SEQ = 16
DIM = 32
SCALE = 0.1767767  # must match the literal in examples/attn_text_classifier.tns
TOPICS = ["weather", "food", "sports"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "attn")

TRAIN_PER_TOPIC = 60
HELDOUT_PER_TOPIC = 10  # 3 topics -> 30 held-out, matching b=30 in the .tns

# Topic-specific vocabularies (the signal) plus shared neutral connector
# words (no signal). Sentences are composed by sampling from these, so
# the corpus is reproducible and large enough for a statistically
# meaningful Tensora-vs-PyTorch comparison. Real learned parameters,
# understandable inputs -- not a pretrained model.
TOPIC_WORDS = {
    0: "rain snow wind clouds fog sunny storm forecast drizzle cold heat humid thunder frost sky "
       "temperature blizzard breeze overcast hail".split(),
    1: "bread curry pizza salad soup noodles cake pastries cheese rice beans grilled baked roasted "
       "lemon sauce dinner recipe spicy dessert".split(),
    2: "goal race track ball match championship team player sprint marathon tennis basketball score "
       "win kick coach league striker referee tournament".split(),
}
CONNECTORS = "the a and with of in on for was were will today we she he they".split()


def make_sentence(topic, rng):
    n_topic = rng.integers(6, 11)
    words = list(rng.choice(TOPIC_WORDS[topic], size=n_topic)) + list(rng.choice(CONNECTORS, size=3))
    rng.shuffle(words)
    return " ".join(words)


def make_corpus(per_topic, rng):
    return [(make_sentence(t, rng), t) for t in TOPICS_IDX for _ in range(per_topic)]


TOPICS_IDX = [0, 1, 2]


def build_vocab(sentences):
    vocab = {"<pad>": 0}
    for s in sentences:
        for w in s.split():
            vocab.setdefault(w, len(vocab))
    return vocab


def encode(sentence, vocab):
    ids = [vocab.get(w, 0) for w in sentence.split()][:SEQ]
    ids += [0] * (SEQ - len(ids))
    return ids


HID = 16


class Attn(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, DIM)
        self.Wq = nn.Linear(DIM, DIM, bias=False)
        self.Wk = nn.Linear(DIM, DIM, bias=False)
        self.Wv = nn.Linear(DIM, DIM, bias=False)
        self.Wc1 = nn.Linear(DIM, HID)
        self.Wc2 = nn.Linear(HID, 3)

    def embed(self, ids):
        # Pad/truncate to SEQ, then look up embeddings -> [1, SEQ, DIM].
        return self.emb(ids)

    def forward_from_x(self, x):
        # x: [1, SEQ, DIM]; mirrors examples/attn_text_classifier.tns exactly.
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = (Q @ K.transpose(-1, -2)) * SCALE
        A = torch.softmax(scores, dim=-1)
        pooled = (A @ V).mean(dim=1)
        return self.Wc2(torch.relu(self.Wc1(pooled)))

    def forward(self, ids):
        return self.forward_from_x(self.embed(ids))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    corpus = make_corpus(TRAIN_PER_TOPIC, rng)
    heldout = make_corpus(HELDOUT_PER_TOPIC, rng)  # distinct RNG draws -> distinct sentences
    # make_corpus groups by topic; shuffle so the held-out batch interleaves
    # topics (the demo then shows spread-out, varied rows).
    heldout = [heldout[i] for i in rng.permutation(len(heldout))]
    vocab = build_vocab([s for s, _ in corpus] + [s for s, _ in heldout])
    model = Attn(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()

    ids = torch.tensor([encode(s, vocab) for s, _ in corpus])
    labels = torch.tensor([y for _, y in corpus])
    for epoch in range(300):
        opt.zero_grad()
        loss = loss_fn(model(ids), labels)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(ids).argmax(1) == labels).float().mean().item()
    print(f"Trained attention classifier: training accuracy = {acc:.3f}")

    # Export projection/classifier weights in Tensora's layout: Tensora
    # computes X @ W, so each W is [in, out] -- the transpose of PyTorch's
    # [out, in] nn.Linear.weight. The batched-matmul projections carry a
    # leading batch axis of 1 to match the .tns weight shapes.
    with torch.no_grad():
        weights = {
            "Wq": model.Wq.weight.t().contiguous().numpy().reshape(1, DIM, DIM).astype(np.float32),
            "Wk": model.Wk.weight.t().contiguous().numpy().reshape(1, DIM, DIM).astype(np.float32),
            "Wv": model.Wv.weight.t().contiguous().numpy().reshape(1, DIM, DIM).astype(np.float32),
            "Wc1": model.Wc1.weight.t().contiguous().numpy().astype(np.float32),
            "bc1": model.Wc1.bias.numpy().astype(np.float32),
            "Wc2": model.Wc2.weight.t().contiguous().numpy().astype(np.float32),
            "bc2": model.Wc2.bias.numpy().astype(np.float32),
        }

    # Embed all held-out sentences and classify the whole batch in one
    # Tensora run (batch axis b in examples/attn_text_classifier.tns), so
    # the Tensora-vs-PyTorch agreement is statistically meaningful.
    sents = [s for s, _ in heldout]
    topics = [t for _, t in heldout]
    tok = torch.tensor([encode(s, vocab) for s in sents])
    with torch.no_grad():
        x = model.embed(tok).numpy().astype(np.float32)  # [N, SEQ, DIM]
        logits = model.forward_from_x(torch.tensor(x)).numpy().astype(np.float32)  # [N, 3]

    os.makedirs(OUT_DIR, exist_ok=True)
    d = os.path.join(OUT_DIR, "eval")
    os.makedirs(d, exist_ok=True)
    # Xq and Xk are the same embedding batch (see the .tns header).
    np.save(os.path.join(d, "Xq.npy"), x)
    np.save(os.path.join(d, "Xk.npy"), x)
    for name, arr in weights.items():
        np.save(os.path.join(d, f"{name}.npy"), arr)
    np.save(os.path.join(d, "expected.npy"), logits)
    meta = {
        "count": len(sents),
        "sentences": sents,
        "true_topics": [TOPICS[t] for t in topics],
    }
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    pt_acc = float((logits.argmax(1) == np.array(topics)).mean())
    print(f"Wrote {len(sents)}-sentence eval batch to {os.path.normpath(OUT_DIR)} "
          f"(PyTorch accuracy on this batch = {pt_acc:.3f})")


if __name__ == "__main__":
    main()
