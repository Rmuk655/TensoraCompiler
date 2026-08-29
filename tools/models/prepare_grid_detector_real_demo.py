#!/usr/bin/env python3
"""Prepare the REAL-photograph fixed-grid CNN object-localization demo.

This is the "does this generalize past toy shapes?" companion to
examples/cnn_grid_detector.tns (kept as the minimal/synthetic teaching
example -- see its header comment). Same architecture family, same
frozen v1 operators, same "one class per grid cell, no box regression,
no NMS" design -- but now trained on real photographs with real,
human-annotated object locations, not shapes this script drew itself.

Ground truth comes from Pascal VOC 2012's per-pixel segmentation masks
(fetched via the "nateraw/pascal-voc-2012" dataset's lightweight
HuggingFace datasets-server preview API -- small per-image JPEGs, no
multi-GB dataset download). For each image, every VOC class's largest
connected mask region is reduced to its bounding-box CENTER (exactly
the reduction docs/language/... describes: real annotation -> center
point -> grid cell), for five common, recognizable classes: person,
car, dog, bicycle, bus. A 128x128 input with the same 3x
[conv2d(pad=1) -> relu -> maxpool(2)] backbone as the synthetic demo
produces a 16x16 grid (128 -> 64 -> 32 -> 16), i.e. finer localization
than the synthetic demo's 4x4, without adding a fourth stage.

Honesty boundary (same as every other demos/*_demo.sh): 100% of the
neural network inference (every conv2d/relu/maxpool/softmax) executes
inside Tensora. This script (and the VOC dataset itself) supplies real
photographs and real object locations; Tensora is never given
synthetic pixels for this demo. Python's only other job, in the demo
runner, is drawing the predicted boxes after Tensora has already
produced the per-cell class predictions.

Run once (needs PyTorch + scipy + Pillow, and internet access to fetch
~1500 small preview images from HuggingFace, one time, no dataset
download):
    python3 tools/models/prepare_grid_detector_real_demo.py
Safe to re-run: skips everything if
demos/artifacts/grid_detector_real/eval/ already exists (pass --force
to redo).
"""
import json
import os
import sys
import urllib.request

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("error: PyTorch is required. "
                     "Install with: pip install -r tools/models/requirements.txt")
from PIL import Image

SEED = 0
IMG = 128           # image is IMG x IMG pixels (resized, aspect not preserved -- see below)
GRID = 16           # GRID x GRID detection cells: 128 -> 64 -> 32 -> 16, same 3-stage backbone as the synthetic demo
CELL = IMG // GRID  # 8x8 pixels per cell
MIN_CELL_FRACTION = 0.12  # a cell needs >=12% of its pixels in one kept class to be labeled that class
TRAIN_N = 1300
EVAL_N = 24
DISPLAY = 4
EPOCHS = 40
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "grid_detector_real")
CACHE_DIR = os.path.join(OUT_DIR, "voc_cache")  # raw downloaded image/mask JPEGs (gitignored, re-used across runs)
MODEL_PT = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "grid-detector-real-model.pt")

# VOC class index -> name (index 0 is background). We only KEEP five of
# these (below); the rest of VOC's 20 classes are treated as
# background for this demo, same as any pixel with no object at all.
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
KEEP = ["person", "car", "dog", "bicycle", "bus"]  # common, recognizable, and each has enough VOC instances
CLASSES = ["background"] + KEEP
KEEP_VOC_IDS = {VOC_CLASSES.index(c): CLASSES.index(c) for c in KEEP}  # VOC id -> our compact id

BOX_COLOR = {
    "person": (220, 40, 40), "car": (40, 100, 220), "dog": (40, 170, 70),
    "bicycle": (200, 140, 20), "bus": (150, 40, 200),
}


def voc_colormap():
    """The standard Pascal VOC devkit color palette (bit-interleaved
    RGB), used to turn each mask pixel's color back into a class id."""
    def bitget(v, i):
        return (v >> i) & 1
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


CMAP = voc_colormap()
CMAP[255] = [224, 224, 192]  # VOC's "void"/boundary color, not a real class
PALETTE_IDS = list(range(21)) + [255]
PALETTE_COLORS = CMAP[PALETTE_IDS].astype(np.int32)


def snap_mask_to_palette(mask_rgb):
    """VOC masks are served here as lossy JPEGs, so pixel colors drift
    slightly off the exact palette; snap every pixel to its NEAREST
    palette color (palette entries are far apart in RGB space, so this
    fully recovers the intended class id -- verified during design)."""
    flat = mask_rgb.reshape(-1, 3).astype(np.int32)
    d = ((flat[:, None, :] - PALETTE_COLORS[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d, axis=1)
    return np.array(PALETTE_IDS)[idx].reshape(mask_rgb.shape[:2])


def fetch_voc_rows(split, offset, length):
    import time
    url = (f"https://datasets-server.huggingface.co/rows?dataset=nateraw%2Fpascal-voc-2012"
           f"&config=default&split={split}&offset={offset}&length={length}")
    last_err = None
    for attempt in range(5):
        try:
            return json.load(urllib.request.urlopen(url, timeout=30))["rows"]
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def cached_download(url, path):
    import time
    if os.path.exists(path):
        return path
    last_err = None
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def fetch_all_rows():
    """VOC-2012-segmentation (via HF) has 1464 train + 1449 val rows;
    the preview API caps each request at 100 rows, so page through both
    splits. Only the small per-row image.jpg/mask.jpg preview JPEGs are
    downloaded -- never the ~330MB source parquet file itself."""
    rows = []
    for split, total in [("train", 1464), ("val", 1449)]:
        off = 0
        while off < total:
            page = fetch_voc_rows(split, off, min(100, total - off))
            for r in page:
                r["_split"] = split
            rows.extend(page)
            off += 100
            print(f"  listed {split} {off}/{total}")
    return rows


def label_one_image(mask_rgb):
    """Real annotation -> grid cells, DENSE: snap the mask to VOC's
    palette, resize the resulting label map (nearest-neighbor -- exact,
    no blending of class ids) to IMG x IMG (the same frame the photo
    itself is resized into), then for every GRID x GRID cell take
    whichever of our five kept classes covers the LARGEST share of that
    cell's pixels, labeling the cell with it if that share clears
    MIN_CELL_FRACTION (otherwise background). This is a per-cell
    majority vote over the real segmentation mask -- not just one
    center point per object -- so a cell is labeled whenever the real
    object actually overlaps it, giving both denser training signal and
    boxes that approximate the object's real extent when drawn."""
    idmask = snap_mask_to_palette(mask_rgb)
    compact = np.zeros(256, dtype=np.int64)
    for voc_id, our_id in KEEP_VOC_IDS.items():
        compact[voc_id] = our_id
    compact_full = compact[idmask]  # native mask resolution, values in 0..len(CLASSES)-1
    resized = np.array(
        Image.fromarray(compact_full.astype(np.uint8)).resize((IMG, IMG), Image.NEAREST))
    blocks = resized.reshape(GRID, CELL, GRID, CELL).transpose(0, 2, 1, 3).reshape(GRID, GRID, CELL * CELL)
    cell = np.zeros((GRID, GRID), dtype=np.int64)
    for r in range(GRID):
        for c in range(GRID):
            counts = np.bincount(blocks[r, c], minlength=len(CLASSES))
            best = int(np.argmax(counts[1:])) + 1 if counts[1:].any() else 0
            if best != 0 and counts[best] >= MIN_CELL_FRACTION * CELL * CELL:
                cell[r, c] = best
    return cell if cell.any() else None


class GridDetector(nn.Module):
    # Identical architecture family to examples/cnn_grid_detector.tns's
    # PyTorch mirror, just a larger spatial size throughout (128 grid
    # instead of 32) -- the backbone stays 3 stages deep either way.
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv3(x)), 2)
        return self.head(x)


def main():
    marker = os.path.join(OUT_DIR, "eval", "meta.json")
    if os.path.exists(marker) and "--force" not in sys.argv:
        print(f"Already prepared: {os.path.normpath(marker)} exists "
              "(delete demos/artifacts/grid_detector_real/, or pass --force, to redo)")
        return
    torch.manual_seed(SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Listing Pascal VOC 2012 rows (HuggingFace datasets-server, no dataset download) ...")
    rows = fetch_all_rows()
    print(f"  {len(rows)} rows available (train+val)")

    images, labels = [], []
    n_scanned = 0
    for r in rows:
        row = r["row"]
        split = r["_split"]
        idx = r["row_idx"]
        img_path = os.path.join(CACHE_DIR, f"{split}_{idx}_image.jpg")
        mask_path = os.path.join(CACHE_DIR, f"{split}_{idx}_mask.jpg")
        try:
            cached_download(row["image"]["src"], img_path)
            cached_download(row["mask"]["src"], mask_path)
        except Exception:
            continue
        n_scanned += 1
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        cell = label_one_image(mask_rgb)
        if cell is None:
            continue
        img = Image.open(img_path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
        images.append(img_path)  # keep path; resize again lazily below to save memory
        labels.append(cell)
        if len(images) % 100 == 0:
            print(f"  usable images so far: {len(images)} (scanned {n_scanned}/{len(rows)})")
        if len(images) >= TRAIN_N + EVAL_N + 50:  # a little headroom for the eval split below
            break
    print(f"Usable images (>=1 of {KEEP} present): {len(images)} out of {n_scanned} scanned")
    if len(images) < TRAIN_N // 4:
        raise SystemExit(f"error: only found {len(images)} usable images -- too few to train on. "
                          "Check network access to huggingface.co.")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(images))
    n_train = min(TRAIN_N, len(images) - EVAL_N)
    train_idx, eval_idx = order[:n_train], order[n_train:n_train + EVAL_N]

    def load_batch(idxs):
        xs = np.zeros((len(idxs), 3, IMG, IMG), dtype=np.float32)
        ys = np.zeros((len(idxs), GRID, GRID), dtype=np.int64)
        for i, j in enumerate(idxs):
            img = Image.open(images[j]).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
            xs[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            ys[i] = labels[j]
        return xs, ys

    print(f"Loading {len(train_idx)} train / {len(eval_idx)} eval images into memory ...")
    train_x, train_y = load_batch(train_idx)
    eval_x, eval_y = load_batch(eval_idx)

    # Augment the (small, real-photo) training set with a left-right
    # flip of every image -- flipping the label grid's COLUMN axis to
    # match, since the label is spatial. This doubles the effective
    # training data at no extra download/labeling cost, the standard,
    # cheapest way to reduce overfitting on a few hundred real photos;
    # only the TRAIN set is flipped, never eval (eval must reflect the
    # real, unmodified photo Tensora is checked against).
    train_x = np.concatenate([train_x, train_x[:, :, :, ::-1]], axis=0)
    train_y = np.concatenate([train_y, train_y[:, :, ::-1]], axis=0)
    train_x = np.ascontiguousarray(train_x)
    train_y = np.ascontiguousarray(train_y)

    model = GridDetector(len(CLASSES))
    cached_ok = False
    if os.path.exists(MODEL_PT):
        state = torch.load(MODEL_PT)
        if state.get("head.weight") is not None and state["head.weight"].shape[0] == len(CLASSES):
            model.load_state_dict(state)
            cached_ok = True
            print(f"Loaded cached model from {os.path.normpath(MODEL_PT)} (delete it to retrain)")
    if not cached_ok:
        # Grid cells are overwhelmingly background (16x16=256 cells,
        # only a handful overlap a real object), so weight the loss by
        # inverse class frequency -- computed from the actual training
        # labels, not a fixed guess -- to counter that imbalance;
        # otherwise the model trivially predicts all-background.
        counts = np.bincount(train_y.reshape(-1), minlength=len(CLASSES)).astype(np.float64)
        class_weight = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
        # Cap the upper end: an extreme weight (e.g. bicycle's raw
        # ~14x, from its rarity) pushes the model to over-predict that
        # class everywhere, flooding the output with false positives:
        # a real cost (this is a per-cell classifier reporting a
        # confidence, not a hand-tuned threshold), not just cosmetic.
        class_weight = torch.clamp(class_weight, max=5.0)
        print(f"  per-cell class counts (train, incl. flip augmentation): "
              f"{dict(zip(CLASSES, counts.astype(int).tolist()))}")
        print(f"  class weights (capped at 5.0): {dict(zip(CLASSES, class_weight.tolist()))}")
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.3)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
            print(f"  epoch {epoch + 1}/{EPOCHS} done (loss {loss.item():.4f})")
        torch.save(model.state_dict(), MODEL_PT)
    model.eval()

    with torch.no_grad():
        eval_logits = model(torch.from_numpy(eval_x))
        probs = F.softmax(eval_logits, dim=1).numpy().astype(np.float32)
    pred = eval_logits.argmax(1).numpy()
    # Object-cell recall/precision are the meaningful numbers here (raw
    # per-cell accuracy is dominated by trivial background cells).
    obj_mask = eval_y != 0
    recall = float((pred[obj_mask] == eval_y[obj_mask]).mean()) if obj_mask.any() else float("nan")
    print(f"Grid detector (real photos): held-out object-cell recall = {recall:.3f} "
          f"({int(obj_mask.sum())} labeled object cells across {EVAL_N} images)")

    with torch.no_grad():
        weights = {
            "Wc1": model.conv1.weight.numpy().astype(np.float32), "bc1": model.conv1.bias.numpy().astype(np.float32),
            "Wc2": model.conv2.weight.numpy().astype(np.float32), "bc2": model.conv2.bias.numpy().astype(np.float32),
            "Wc3": model.conv3.weight.numpy().astype(np.float32), "bc3": model.conv3.bias.numpy().astype(np.float32),
            "Whead": model.head.weight.numpy().astype(np.float32), "bhead": model.head.bias.numpy().astype(np.float32),
        }

    display = sorted({round(k * (EVAL_N - 1) / (DISPLAY - 1)) for k in range(DISPLAY)})
    d = os.path.join(OUT_DIR, "eval")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "image.npy"), eval_x)
    for wname, arr in weights.items():
        np.save(os.path.join(d, f"{wname}.npy"), arr)
    np.save(os.path.join(d, "expected.npy"), probs)
    np.save(os.path.join(d, "cells_true.npy"), eval_y)
    json.dump({"classes": CLASSES, "img_size": IMG, "grid": GRID, "cell": CELL, "box_color": BOX_COLOR},
              open(os.path.join(OUT_DIR, "classes.json"), "w"), indent=2)
    for b in display:
        img = Image.open(images[eval_idx[b]]).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
        img.save(os.path.join(d, f"image_{b}.png"))
    json.dump({"count": EVAL_N, "display": display}, open(os.path.join(d, "meta.json"), "w"), indent=2)
    print(f"Wrote {EVAL_N}-image eval batch to {os.path.normpath(OUT_DIR)}; showing {len(display)} images")


if __name__ == "__main__":
    main()
