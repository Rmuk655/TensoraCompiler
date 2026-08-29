#!/usr/bin/env bash
# CNN grid-detector real-inference demo: fixed-grid object LOCALIZATION
# on real photographs (not shapes drawn by a script). Classifies every
# cell of a 16x16 grid over a 128x128 real Pascal-VOC-2012 photograph
# containing person/car/dog/bicycle/bus (examples/cnn_grid_detector_real.tns),
# checks every cell's prediction against the PyTorch reference, and
# draws the detected regions + labels on top of a few of the held-out
# photos. Only numpy + PIL are needed here (no torch); run
# tools/models/prepare_grid_detector_real_demo.py once first (needs
# PyTorch + scipy-free numpy + internet access to fetch small VOC
# preview images from HuggingFace, one time).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/grid_detector_real"
EVAL="$ART/eval"
TOL="1e-3"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch, internet access):"
  echo "  python3 tools/models/prepare_grid_detector_real_demo.py"
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

echo "=== Tensora CNN Real-World Demo: object localization on REAL photographs (16x16 grid, 5 classes) ==="
echo "Model: examples/cnn_grid_detector_real.tns (3x [conv -> relu -> maxpool] -> 1x1 conv head -> softmax, per cell)"
echo "Ground truth: real Pascal VOC 2012 segmentation masks -- not shapes this demo drew itself."
echo

"$BUILD/tensora" --run examples/cnn_grid_detector_real.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

python3 - "$ART" "$EVAL" "$TOL" <<'PY'
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

art, d, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
info = json.load(open(os.path.join(art, "classes.json")))
CLASSES, IMG, GRID, CELL = info["classes"], info["img_size"], info["grid"], info["cell"]
BOX_COLOR = {k: tuple(v) for k, v in info["box_color"].items()}
meta = json.load(open(os.path.join(d, "meta.json")))
exp = np.load(os.path.join(d, "expected.npy"))       # [N, classes, GRID, GRID] PyTorch probs
got = np.load(os.path.join(d, "tensora_out.npy"))    # [N, classes, GRID, GRID] Tensora probs
truth = np.load(os.path.join(d, "cells_true.npy"))   # [N, GRID, GRID] ground-truth cell ids
N = exp.shape[0]


def boxes_from_cells(cell_ids):
    """Group adjacent same-class cells into connected regions and draw
    ONE box per region (its bounding rectangle), instead of one small
    box per cell -- a real object usually spans several grid cells, so
    this reconstructs something closer to a real bounding box from the
    per-cell predictions, with no new Tensora computation involved."""
    seen = np.zeros_like(cell_ids, dtype=bool)
    boxes = []
    for r in range(GRID):
        for c in range(GRID):
            cls_idx = int(cell_ids[r, c])
            if cls_idx == 0 or seen[r, c]:
                continue
            # Flood-fill the 4-connected region of this class starting at (r, c).
            stack, region = [(r, c)], []
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                region.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < GRID and 0 <= nc < GRID and not seen[nr, nc] and cell_ids[nr, nc] == cls_idx:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in region]
            cols = [p[1] for p in region]
            boxes.append((cls_idx, min(rows), max(rows), min(cols), max(cols)))
    return boxes


def annotate(idx, pred_cells, probs):
    src = os.path.join(d, f"image_{idx}.png")
    if not os.path.exists(src):
        return None
    img = Image.open(src).convert("RGB")
    scale = 4
    img = img.resize((IMG * scale, IMG * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for cls_idx, r0, r1, c0, c1 in boxes_from_cells(pred_cells):
        cls = CLASSES[cls_idx]
        x0, y0 = c0 * CELL * scale, r0 * CELL * scale
        x1, y1 = (c1 + 1) * CELL * scale, (r1 + 1) * CELL * scale
        conf = float(probs[cls_idx, r0:r1 + 1, c0:c1 + 1].max())
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BOX_COLOR[cls], width=3)
        draw.text((x0 + 3, y0 + 3), f"{cls.upper()} {conf:.2f}", fill=BOX_COLOR[cls])
    out = os.path.join(d, f"image_{idx}_detected.png")
    img.save(out)
    return out


print(f"Showing {len(meta['display'])} of {N} localized photographs:\n")
for b in meta["display"]:
    pred_cells = got[b].argmax(0)
    regions = boxes_from_cells(pred_cells)
    n_true = len({(int(truth[b, r, c])) for r in range(GRID) for c in range(GRID) if truth[b, r, c] != 0})
    print(f"  Photo (row {b} of {N}):")
    print(f"    Ground-truth classes present: {n_true}")
    print(f"    Tensora regions detected:     {len(regions)} -> " +
          ", ".join(f"{CLASSES[c]}" for c, *_ in regions) if regions else "    Tensora regions detected:     0")
    out = annotate(b, pred_cells, got[b])
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

pt_cells = exp.argmax(1)
tn_cells = got.argmax(1)
agree = int((pt_cells == tn_cells).sum())
total_cells = pt_cells.size
max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
obj_mask = truth != 0
pt_recall = float((pt_cells[obj_mask] == truth[obj_mask]).mean()) if obj_mask.any() else float("nan")
tn_recall = float((tn_cells[obj_mask] == truth[obj_mask]).mean()) if obj_mask.any() else float("nan")
print(f"\nOver all {N} photographs ({total_cells} grid cells, {int(obj_mask.sum())} labeled object cells):")
print(f"  Tensora vs PyTorch: {agree}/{total_cells} per-cell predictions agree")
print(f"  Max prob error:     {max_err:.2e}   mean {mean_err:.2e}   (tolerance {tol:g})")
print(f"  Object-cell recall: PyTorch {pt_recall*100:.1f}%   Tensora {tn_recall*100:.1f}%   "
      f"(a small from-scratch CNN on real, diverse photos -- not a state-of-the-art detector)")
ok = agree == total_cells and max_err <= tol
print("  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
