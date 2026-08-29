#!/usr/bin/env python3
"""Bonus scientific-comparison script for the real-photo grid detector.

This script is OUTSIDE the Tensora pipeline: nothing it produces is
ever fed into the compiler or the .tns programs. It exists only to put
examples/cnn_grid_detector_real.tns's numbers in honest context by
comparing two PyTorch models trained/fine-tuned on the exact same ~900
VOC photographs:

  1. The Tensora-compatible "from scratch" model: a tiny 3-stage
     conv/relu/maxpool backbone plus a 1x1-conv classification head,
     one class per 8x8 grid cell, no box regression, no BatchNorm/
     LayerNorm (Tensora v1 has neither op) -- exactly what
     tools/models/prepare_grid_detector_real_demo.py trains and what
     Tensora itself executes in the shipped demo.
  2. A real, production-grade object detector -- Ultralytics YOLO11s,
     pretrained on COCO -- fine-tuned on the same photographs, doing
     real box regression + non-max suppression, neither of which
     Tensora v1's frozen operator set can express (no `gather`/index
     select, no NMS, no integer tensors -- see
     docs/language/... roadmap).

Reported side by side: precision/recall (grid detector) vs mAP50-95
(YOLO11s), plus annotated example images from both. The intent is to
be transparent about the gap between a from-scratch, operator-frozen,
~900-image, compiler-scope model and a real transfer-learning
pipeline, not to claim the from-scratch model is competitive.

Ground truth for BOTH models comes from the same VOC segmentation
masks already cached by prepare_grid_detector_real_demo.py (run that
script first) -- this script derives ordinary axis-aligned bounding
boxes (per connected mask region, not grid cells) for YOLO's format,
instead of collapsing them into 8x8-cell labels.

Run once (needs ultralytics, in addition to the demo script's deps;
uses CUDA automatically if available):
    python3 tools/models/prepare_grid_detector_real_demo.py   # first, to populate the VOC cache
    python3 tools/models/compare_yolo_baseline.py
"""
import glob
import json
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage

import prepare_grid_detector_real_demo as grid  # reuse VOC palette/class-id logic

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "yolo_baseline")
YOLO_DATA_DIR = os.path.join(OUT_DIR, "yolo_dataset")
CACHE_DIR = grid.CACHE_DIR
CLASSES = grid.KEEP  # YOLO format wants 0-based ids with NO background class
MIN_BOX_PIXELS = 24 * 24  # ignore tiny/noisy mask fragments (native VOC mask resolution)


def boxes_from_mask(mask_rgb):
    """Real annotation -> ordinary [class, x1, y1, x2, y2] boxes: snap
    the mask to VOC's palette, then take every separate connected
    region of each of our five kept classes as one box (unlike the grid
    detector, which only needs a per-cell majority vote, YOLO needs
    real box coordinates in the original image's pixel space)."""
    idmask = grid.snap_mask_to_palette(mask_rgb)
    h, w = idmask.shape
    boxes = []
    for voc_id, our_id in grid.KEEP_VOC_IDS.items():
        binary = idmask == voc_id
        if not binary.any():
            continue
        labeled, n = ndimage.label(binary)
        for region_id in range(1, n + 1):
            ys, xs = np.where(labeled == region_id)
            if len(ys) < MIN_BOX_PIXELS:
                continue
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            boxes.append((our_id - 1, x1 / w, y1 / h, x2 / w, y2 / h))  # our_id-1: drop background offset
    return boxes


def write_yolo_split(rows, split, images_dir, labels_dir):
    written = 0
    for r in rows:
        img_path = os.path.join(CACHE_DIR, f"{r['_split']}_{r['idx']}_image.jpg")
        mask_path = os.path.join(CACHE_DIR, f"{r['_split']}_{r['idx']}_mask.jpg")
        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            continue
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        boxes = boxes_from_mask(mask_rgb)
        if not boxes:
            continue
        stem = f"{r['_split']}_{r['idx']}"
        shutil.copy(img_path, os.path.join(images_dir, stem + ".jpg"))
        with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
            for cls_id, x1, y1, x2, y2 in boxes:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bw, bh = x2 - x1, y2 - y1
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        written += 1
    return written


def build_yolo_dataset():
    """Re-derive the same usable-image list prepare_grid_detector_real_demo.py
    used (person/car/dog/bicycle/bus present), but keep ordinary boxes
    instead of grid cells, and split 85/15 train/val with the same seed."""
    cached_names = {os.path.basename(p) for p in glob.glob(os.path.join(CACHE_DIR, "*_image.jpg"))}
    rows = []
    for name in sorted(cached_names):
        split, idx_str, _ = name.split("_", 2)
        rows.append({"_split": split, "idx": int(idx_str)})
    rng = np.random.RandomState(grid.SEED)
    rng.shuffle(rows)

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = os.path.join(YOLO_DATA_DIR, sub)
        os.makedirs(d, exist_ok=True)

    n_val = max(24, int(0.15 * len(rows)))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    n_train = write_yolo_split(train_rows, "train",
                                os.path.join(YOLO_DATA_DIR, "images/train"),
                                os.path.join(YOLO_DATA_DIR, "labels/train"))
    n_val_written = write_yolo_split(val_rows, "val",
                                      os.path.join(YOLO_DATA_DIR, "images/val"),
                                      os.path.join(YOLO_DATA_DIR, "labels/val"))
    print(f"YOLO dataset: {n_train} train images, {n_val_written} val images "
          f"(images with none of our 5 classes present are skipped)")

    data_yaml = os.path.join(YOLO_DATA_DIR, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(YOLO_DATA_DIR)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"names: {json.dumps(CLASSES)}\n")
    return data_yaml, n_train, n_val_written


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Deriving ordinary bounding boxes from the same cached VOC masks "
          "(no re-download needed) ...")
    data_yaml, n_train, n_val = build_yolo_dataset()
    if n_train < 50:
        raise SystemExit(f"error: only {n_train} usable training images -- "
                          f"run prepare_grid_detector_real_demo.py first to populate the VOC cache")

    from ultralytics import YOLO
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning YOLO11s (COCO-pretrained) on {n_train} train / {n_val} val images, "
          f"device={'cuda' if device == 0 else 'cpu'} ...")
    model = YOLO("yolo11s.pt")
    results = model.train(
        data=data_yaml, epochs=40, imgsz=320, batch=16, device=device,
        project=OUT_DIR, name="finetune", exist_ok=True, verbose=False, patience=10,
    )
    metrics = model.val(data=data_yaml, device=device, project=OUT_DIR, name="eval", exist_ok=True)
    summary = {
        "model": "YOLO11s (COCO-pretrained, fine-tuned)",
        "train_images": n_train,
        "val_images": n_val,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_map50": {c: float(v) for c, v in zip(CLASSES, metrics.box.maps)},
    }
    with open(os.path.join(OUT_DIR, "yolo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Render a few annotated val images for the README/demo comparison.
    sample_dir = os.path.join(OUT_DIR, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    val_images = sorted(glob.glob(os.path.join(YOLO_DATA_DIR, "images/val", "*.jpg")))[:6]
    preds = model.predict(val_images, device=device, conf=0.35, save=False, verbose=False)
    for p, path in zip(preds, val_images):
        out_path = os.path.join(sample_dir, os.path.basename(path).replace(".jpg", "_yolo.png"))
        p.save(filename=out_path)
    print(f"Wrote {len(val_images)} annotated YOLO sample images to {os.path.normpath(sample_dir)}")
    print(f"Summary written to {os.path.normpath(os.path.join(OUT_DIR, 'yolo_summary.json'))}")


if __name__ == "__main__":
    main()
