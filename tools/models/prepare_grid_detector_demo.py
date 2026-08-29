#!/usr/bin/env python3
"""Prepare the fixed-grid CNN object-detection real-inference demo.

Trains a small CNN once in PyTorch to solve a fixed-grid detection task
(early-YOLO style, see examples/cnn_grid_detector.tns for the exact
architecture and design rationale), on synthetic 32x32 RGB images
containing 1-3 colored shapes (circle / square / triangle), each
placed inside its own cell of a 4x4 grid so cells never overlap.
Exports the trained weights and a held-out eval batch:

  demos/artifacts/grid_detector/
      eval/
          image.npy Wc1.npy bc1.npy Wc2.npy bc2.npy Wc3.npy bc3.npy
          Whead.npy bhead.npy      -> a self-contained --data-dir (EVAL_N images)
          expected.npy             -> PyTorch reference per-cell probabilities
          cells_true.npy           -> ground-truth per-cell class ids (for scoring)
          image_<i>.png            -> a raw, unannotated display image
          meta.json                -> which images are displayed

`eval/` is directly usable as `tensora --run
examples/cnn_grid_detector.tns --data-dir demos/artifacts/grid_detector/eval`.
Run once (see tools/models/requirements.txt, no download needed -- the
"dataset" is synthesized on the fly, unlike the STL-10/ViT demos):
    python3 tools/models/prepare_grid_detector_demo.py
Safe to re-run: skips everything if demos/artifacts/grid_detector/eval/
already exists (pass --force to redo).

The data is synthetic (not a real photo dataset) because v1's grid
detector is meant to demonstrate WHAT the language/compiler can express
(conv2d/relu/maxpool/softmax composed into a spatial, per-cell
classifier), not to be a state-of-the-art detector; the honesty
boundary is the same as every other demos/*_demo.sh: 100% of the neural
network inference (every conv2d/relu/maxpool/softmax) executes inside
Tensora. Python only synthesizes the input pixels (offline, before
Tensora ever sees them) and later draws the output boxes on screen
(after Tensora has already produced the per-cell class predictions).
"""
import json
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("error: PyTorch is required. "
                     "Install with: pip install -r tools/models/requirements.txt")
from PIL import Image, ImageDraw

SEED = 0
IMG = 32           # image is IMG x IMG pixels
GRID = 4           # GRID x GRID detection cells (matches the .tns backbone: 32->16->8->4)
CELL = IMG // GRID  # 8x8 pixels per cell
CLASSES = ["background", "circle", "square", "triangle"]
TRAIN_N = 4000
EPOCHS = 30
EVAL_N = 30
DISPLAY = 4
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "grid_detector")
MODEL_PT = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "grid-detector-model.pt")

# Per-class base hue (RGB), jittered per instance so the network cannot
# trivially memorize one exact color -- it has to look at shape.
BASE_COLOR = {"circle": (200, 30, 30), "square": (30, 90, 200), "triangle": (30, 160, 60)}


def draw_shape(draw, cls, cx, cy, r, rng):
    color = tuple(int(np.clip(c + rng.integers(-40, 41), 0, 255)) for c in BASE_COLOR[cls])
    if cls == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif cls == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:  # triangle
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


def make_example(rng):
    """One synthetic image: a light background plus 1-3 shapes, each
    centered in a distinct, randomly-chosen grid cell so no two objects
    share a cell (this is what makes NMS-free per-cell decoding valid)."""
    bg = tuple(int(c) for c in rng.integers(230, 256, size=3))
    img = Image.new("RGB", (IMG, IMG), bg)
    draw = ImageDraw.Draw(img)
    cells_true = np.zeros((GRID, GRID), dtype=np.int64)  # 0 = background
    n_objects = rng.integers(1, 4)  # 1..3 inclusive
    all_cells = [(r, c) for r in range(GRID) for c in range(GRID)]
    rng.shuffle(all_cells)
    for (row, col) in all_cells[:n_objects]:
        cls_idx = int(rng.integers(1, len(CLASSES)))  # never background
        cx = col * CELL + CELL // 2 + int(rng.integers(-1, 2))
        cy = row * CELL + CELL // 2 + int(rng.integers(-1, 2))
        r = CELL // 2 - int(rng.integers(1, 3))  # stays inside the cell
        draw_shape(draw, CLASSES[cls_idx], cx, cy, r, rng)
        cells_true[row, col] = cls_idx
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # CHW, [0,1]
    return arr, cells_true, img


class GridDetector(nn.Module):
    # Mirrors examples/cnn_grid_detector.tns exactly: 3x [conv3x3(pad1)
    # -> relu -> maxpool2] backbone (32 -> 16 -> 8 -> 4), then a 1x1 conv
    # head producing 4 per-cell class logits (background + 3 shapes).
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.head = nn.Conv2d(32, len(CLASSES), 1)

    def forward(self, x):
        x = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv3(x)), 2)
        return self.head(x)  # [B, classes, GRID, GRID] logits


def main():
    marker = os.path.join(OUT_DIR, "eval", "meta.json")
    if os.path.exists(marker) and "--force" not in sys.argv:
        print(f"Already prepared: {os.path.normpath(marker)} exists "
              "(delete demos/artifacts/grid_detector/, or pass --force, to redo)")
        return
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print(f"Synthesizing {TRAIN_N} training images ({GRID}x{GRID} grid, {IMG}x{IMG} pixels) ...")
    train_x = np.zeros((TRAIN_N, 3, IMG, IMG), dtype=np.float32)
    train_y = np.zeros((TRAIN_N, GRID, GRID), dtype=np.int64)
    for i in range(TRAIN_N):
        arr, cells, _ = make_example(rng)
        train_x[i], train_y[i] = arr, cells
    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_y)

    model = GridDetector()
    cached_ok = False
    if os.path.exists(MODEL_PT):
        state = torch.load(MODEL_PT)
        if state.get("head.weight") is not None and state["head.weight"].shape[0] == len(CLASSES):
            model.load_state_dict(state)
            cached_ok = True
            print(f"Loaded cached model from {os.path.normpath(MODEL_PT)} (delete it to retrain)")
    if not cached_ok:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(train_x_t, train_y_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                logits = model(xb)  # [B, classes, GRID, GRID]
                # Per-cell 4-way cross entropy, averaged over all cells and the batch.
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            print(f"  epoch {epoch + 1}/{EPOCHS} done (loss {loss.item():.4f})")
        torch.save(model.state_dict(), MODEL_PT)
    model.eval()

    # Held-out per-cell accuracy over a fresh synthetic batch.
    eval_x = np.zeros((EVAL_N, 3, IMG, IMG), dtype=np.float32)
    eval_y = np.zeros((EVAL_N, GRID, GRID), dtype=np.int64)
    eval_imgs = []
    for i in range(EVAL_N):
        arr, cells, img = make_example(rng)
        eval_x[i], eval_y[i] = arr, cells
        eval_imgs.append(img)
    eval_x_t = torch.from_numpy(eval_x)
    with torch.no_grad():
        logits = model(eval_x_t)  # [EVAL_N, classes, GRID, GRID]
        probs = F.softmax(logits, dim=1).numpy().astype(np.float32)
    cell_acc = float((logits.argmax(1).numpy() == eval_y).mean())
    print(f"Grid detector: held-out PER-CELL accuracy = {cell_acc:.3f} "
          f"({GRID * GRID} cells x {EVAL_N} images)")

    with torch.no_grad():
        weights = {
            "Wc1": model.conv1.weight.numpy().astype(np.float32),
            "bc1": model.conv1.bias.numpy().astype(np.float32),
            "Wc2": model.conv2.weight.numpy().astype(np.float32),
            "bc2": model.conv2.bias.numpy().astype(np.float32),
            "Wc3": model.conv3.weight.numpy().astype(np.float32),
            "bc3": model.conv3.bias.numpy().astype(np.float32),
            "Whead": model.head.weight.numpy().astype(np.float32),
            "bhead": model.head.bias.numpy().astype(np.float32),
        }

    display = sorted({round(k * (EVAL_N - 1) / (DISPLAY - 1)) for k in range(DISPLAY)})

    d = os.path.join(OUT_DIR, "eval")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "image.npy"), eval_x)
    for wname, arr in weights.items():
        np.save(os.path.join(d, f"{wname}.npy"), arr)
    np.save(os.path.join(d, "expected.npy"), probs)       # [EVAL_N, classes, GRID, GRID]
    np.save(os.path.join(d, "cells_true.npy"), eval_y)    # [EVAL_N, GRID, GRID]
    json.dump({"classes": CLASSES, "img_size": IMG, "grid": GRID, "cell": CELL},
              open(os.path.join(OUT_DIR, "classes.json"), "w"), indent=2)

    for b in display:
        eval_imgs[b].save(os.path.join(d, f"image_{b}.png"))

    meta = {"count": EVAL_N, "display": display}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    print(f"Wrote {EVAL_N}-image eval batch to {os.path.normpath(OUT_DIR)}; "
          f"showing {len(display)} images")


if __name__ == "__main__":
    main()
