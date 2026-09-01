#!/usr/bin/env python3
"""Prepare the MLP student-placement real-inference demo.

Trains a small MLP once in PyTorch on the UCI Student Performance
dataset (Cortez & Silva, 2008; https://archive.ics.uci.edu/dataset/320)
to predict placement-eligibility (on track to pass, G3 >= 10) from 8
academic features, then exports:

  demos/artifacts/mlp/eval/
      x.npy W1.npy b1.npy W2.npy b2.npy   -> a self-contained --data-dir
                                             (x is a batch of 100 students)
      expected.npy                        -> the PyTorch reference logits [100,2]
      meta.json                           -> true labels + a few readable rows

The eval dir is directly usable as `tensora --run examples/mlp_placement.tns
--data-dir <dir>`: one run classifies all 100 held-out students, and the
demo compares every row against PyTorch. This is a compiler demonstration,
not a real placement or grading system.

Run once, on a machine with PyTorch (see tools/models/requirements.txt):
    python3 tools/models/prepare_mlp_demo.py
Safe to re-run: skips everything if demos/artifacts/mlp/eval/ already
exists (pass --force to redo).
"""
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("error: PyTorch is required. Install with: pip install -r tools/models/requirements.txt")

SEED = 0
BATCH = 100  # held-out records evaluated in one run; matches examples/mlp_placement.tns
FEATURES = ["age", "Medu", "Fedu", "studytime", "failures", "absences", "G1", "G2"]
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "mlp")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def load_dataset():
    """Downloads and parses student-por.csv (semicolon-separated)."""
    print("Downloading UCI Student Performance dataset ...")
    try:
        raw = urllib.request.urlopen(DATASET_URL, timeout=30).read()
    except Exception as e:
        raise SystemExit(f"error: could not download the dataset ({e}). Check your network and retry.")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open("student-por.csv") as f:
            lines = f.read().decode("utf-8").splitlines()
    header = lines[0].split(";")
    rows = [dict(zip(header, [c.strip('"') for c in ln.split(";")])) for ln in lines[1:]]
    return rows


def main():
    marker = os.path.join(OUT_DIR, "eval", "meta.json")
    if os.path.exists(marker) and "--force" not in sys.argv:
        print(f"Already prepared: {os.path.normpath(marker)} exists "
              "(delete demos/artifacts/mlp/, or pass --force, to redo)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rows = load_dataset()

    X = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([1 if float(r["G3"]) >= 10 else 0 for r in rows], dtype=np.int64)

    # Deterministic train/test split.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_test = max(30, len(X) // 5)
    Xtr, ytr = X[:-n_test], y[:-n_test]
    Xte, yte = X[-n_test:], y[-n_test:]

    # Standardize on train statistics; the same transform is applied to
    # the demo inputs (preprocessing lives here, not in the .tns program).
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-6
    Xtr_s = (Xtr - mean) / std
    Xte_s = (Xte - mean) / std

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(Xtr_s)
    yt = torch.tensor(ytr)
    for epoch in range(300):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(torch.tensor(Xte_s)).argmax(1).numpy() == yte).mean()
    print(f"Trained MLP: held-out accuracy = {acc:.3f}")

    # Export weights in Tensora's linear layout: y = x @ W + b, so W is
    # [in, out] -- the transpose of PyTorch's [out, in] nn.Linear.weight.
    with torch.no_grad():
        weights = {
            "W1": model.fc1.weight.t().contiguous().numpy().astype(np.float32),
            "b1": model.fc1.bias.numpy().astype(np.float32),
            "W2": model.fc2.weight.t().contiguous().numpy().astype(np.float32),
            "b2": model.fc2.bias.numpy().astype(np.float32),
        }

    # Evaluate a whole batch of held-out students in one Tensora run, so
    # the Tensora-vs-PyTorch agreement is statistically meaningful. BATCH
    # matches the batch axis in examples/mlp_placement.tns.
    if len(Xte_s) < BATCH:
        raise SystemExit(f"error: only {len(Xte_s)} held-out records; need {BATCH}")
    # The held-out records are already in shuffled order (from the split
    # permutation above), so the batch is not sorted by label.
    xb = Xte_s[:BATCH].astype(np.float32)  # [BATCH, 8], standardized
    yb = yte[:BATCH]
    Xraw = Xte[:BATCH]  # unstandardized features, for display
    with torch.no_grad():
        logits = model(torch.tensor(xb)).numpy().astype(np.float32)  # [BATCH, 2]
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    # Show three genuinely different students -- a confident pass, the
    # most borderline, and a confident fail -- at their real (spread-out)
    # positions in the batch, not reordered to the front.
    pass_conf = probs[:, 1]
    disp = sorted({int(np.argmax(pass_conf)), int(np.argmin(np.abs(pass_conf - 0.5))), int(np.argmin(pass_conf))})

    os.makedirs(OUT_DIR, exist_ok=True)
    d = os.path.join(OUT_DIR, "eval")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x.npy"), xb)
    for name, arr in weights.items():
        np.save(os.path.join(d, f"{name}.npy"), arr)
    np.save(os.path.join(d, "expected.npy"), logits)  # [BATCH, 2]
    meta = {
        "count": BATCH,
        "true_labels": ["placement-eligible" if y == 1 else "not eligible" for y in yb.tolist()],
        "display": [
            {
                "index": i,
                "features": {f: float(Xraw[i][k]) for k, f in enumerate(FEATURES)},
                "true_label": "placement-eligible" if yb[i] == 1 else "not eligible",
                "pytorch_prob_eligible": float(probs[i][1]),
            }
            for i in disp
        ],
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pt_acc = float((logits.argmax(1) == yb).mean())
    print(f"Wrote {BATCH}-record eval batch to {os.path.normpath(OUT_DIR)} "
          f"(PyTorch accuracy on this batch = {pt_acc:.3f})")


if __name__ == "__main__":
    main()
