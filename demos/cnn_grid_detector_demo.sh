#!/usr/bin/env bash
# CNN grid-detector real-inference demo: fixed-grid object detection
# (early-YOLO style). Classifies every cell of a 4x4 grid over a 32x32
# synthetic image containing 1-3 colored shapes (examples/
# cnn_grid_detector.tns), checks every cell's prediction against the
# PyTorch reference, and draws the detected boxes + labels on top of a
# few of the held-out images. Only numpy + PIL are needed here (no
# torch); run tools/models/prepare_grid_detector_demo.py once first.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/grid_detector"
EVAL="$ART/eval"
TOL="1e-3"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch, no download):"
  echo "  python3 tools/models/prepare_grid_detector_demo.py"
  exit 1
fi

open_image() {
  local png="$1"
  [ -f "$png" ] || return 0
  if command -v wslview >/dev/null 2>&1; then wslview "$png" >/dev/null 2>&1 || true
  elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$(wslpath -w "$png" 2>/dev/null || echo "$png")" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$png" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then open "$png" >/dev/null 2>&1 || true
  fi
}

echo "=== Tensora CNN Real-World Demo: fixed-grid object detection (4x4 grid, 3 shape classes) ==="
echo "Model: examples/cnn_grid_detector.tns (3x [conv -> relu -> maxpool] -> 1x1 conv head -> softmax, per cell)"
echo

"$BUILD/tensora" --run examples/cnn_grid_detector.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

python3 - "$ART" "$EVAL" "$TOL" <<'PY'
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

art, d, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
info = json.load(open(os.path.join(art, "classes.json")))
CLASSES, IMG, GRID, CELL = info["classes"], info["img_size"], info["grid"], info["cell"]
meta = json.load(open(os.path.join(d, "meta.json")))
exp = np.load(os.path.join(d, "expected.npy"))       # [N, classes, GRID, GRID] PyTorch probs
got = np.load(os.path.join(d, "tensora_out.npy"))    # [N, classes, GRID, GRID] Tensora probs
truth = np.load(os.path.join(d, "cells_true.npy"))   # [N, GRID, GRID] ground-truth cell ids
N = exp.shape[0]

BOX_COLOR = {"circle": (200, 30, 30), "square": (30, 90, 200), "triangle": (30, 160, 60)}

def annotate(idx, pred_cells):
    """Draw Tensora's own predicted boxes (one per non-background cell,
    at that cell's fixed grid boundary -- this demo predicts a class
    per cell, not a continuous box, see examples/cnn_grid_detector.tns)
    on top of the raw held-out image, and save it next to the original."""
    src = os.path.join(d, f"image_{idx}.png")
    if not os.path.exists(src):
        return None
    img = Image.open(src).convert("RGB")
    scale = 6  # upscale so labels stay readable on a 32x32 source image
    img = img.resize((IMG * scale, IMG * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for row in range(GRID):
        for col in range(GRID):
            cls_idx = int(pred_cells[row, col])
            if cls_idx == 0:
                continue
            cls = CLASSES[cls_idx]
            x0, y0 = col * CELL * scale, row * CELL * scale
            x1, y1 = x0 + CELL * scale, y0 + CELL * scale
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BOX_COLOR[cls], width=3)
            draw.text((x0 + 3, y0 + 3), cls.upper(), fill=BOX_COLOR[cls])
    out = os.path.join(d, f"image_{idx}_detected.png")
    img.save(out)
    return out

print(f"Showing {len(meta['display'])} of {N} detected images:\n")
for b in meta["display"]:
    pred_cells = got[b].argmax(0)  # [GRID, GRID]
    objects = [(r, c, CLASSES[int(pred_cells[r, c])])
               for r in range(GRID) for c in range(GRID) if pred_cells[r, c] != 0]
    print(f"  Image (row {b} of {N}):")
    print(f"    Ground truth objects: {sum(1 for r in range(GRID) for c in range(GRID) if truth[b, r, c] != 0)}")
    print(f"    Tensora detections:   {len(objects)} -> " +
          ", ".join(f"{cls} @ cell({r},{c})" for r, c, cls in objects) if objects else "    Tensora detections:   0")
    out = annotate(b, pred_cells)
    if out:
        print(f"    Annotated image: {out}")
PY

for b in $(python3 -c "import json;print(' '.join(str(x) for x in json.load(open('$EVAL/meta.json'))['display']))"); do
  open_image "$EVAL/image_${b}_detected.png"
done

python3 - "$EVAL" "$TOL" <<'PY'
import sys
import numpy as np
d, tol = sys.argv[1], float(sys.argv[2])
exp = np.load(f"{d}/expected.npy")
got = np.load(f"{d}/tensora_out.npy")
truth = np.load(f"{d}/cells_true.npy")
N = exp.shape[0]

pt_cells = exp.argmax(1)   # [N, GRID, GRID]
tn_cells = got.argmax(1)
agree = int((pt_cells == tn_cells).sum())
total_cells = pt_cells.size
max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
pt_acc = float((pt_cells == truth).mean())
tn_acc = float((tn_cells == truth).mean())
print(f"\nOver all {N} images ({total_cells} grid cells):")
print(f"  Tensora vs PyTorch: {agree}/{total_cells} per-cell predictions agree")
print(f"  Max prob error:     {max_err:.2e}   mean {mean_err:.2e}   (tolerance {tol:g})")
print(f"  Per-cell accuracy:  PyTorch {pt_acc*100:.1f}%   Tensora {tn_acc*100:.1f}%")
ok = agree == total_cells and max_err <= tol
print("  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
