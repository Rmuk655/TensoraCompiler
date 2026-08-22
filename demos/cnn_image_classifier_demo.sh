#!/usr/bin/env bash
# CNN real-inference demo: CIFAR-100 image classification (20 coarse
# superclasses).
#
# Classifies a whole batch of 40 held-out CIFAR-100 images
# (examples/cnn_cifar.tns) through the full Tensora pipeline (--run, JIT)
# in one go, checks every row against the PyTorch reference, and shows a
# few of them with their pictures. Only numpy is needed here (no torch);
# run tools/models/prepare_cifar_demo.py once first.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/cnn"
EVAL="$ART/eval"
TOL="1e-3"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch + torchvision):"
  echo "  python3 tools/models/prepare_cifar_demo.py"
  exit 1
fi

# Best-effort open of a PNG in a GUI viewer. On WSL, wslview or
# explorer.exe hands the file to the default Windows app; elsewhere,
# xdg-open/open. Never fails the demo if no viewer is available.
open_image() {
  local png="$1"
  [ -f "$png" ] || return 0
  if command -v wslview >/dev/null 2>&1; then wslview "$png" >/dev/null 2>&1 || true
  elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$(wslpath -w "$png" 2>/dev/null || echo "$png")" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$png" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then open "$png" >/dev/null 2>&1 || true
  fi
}

echo "=== Tensora CNN Real-World Demo: CIFAR-100 image classification (20 superclasses) ==="
echo "Model: examples/cnn_cifar.tns (4x [conv -> relu -> maxpool] -> flatten -> linear)"
echo

"$BUILD/tensora" --run examples/cnn_cifar.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

# Open the shown images in a GUI viewer (best effort).
for b in $(python3 -c "import json;print(' '.join(str(x['index']) for x in json.load(open('$EVAL/meta.json'))['display']))"); do
  open_image "$EVAL/image_$b.png"
done

python3 - "$EVAL" "$ART/classes.json" "$TOL" <<'PY'
import json, sys
import numpy as np
d, classes_path, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
CLASSES = json.load(open(classes_path))
meta = json.load(open(f"{d}/meta.json"))
exp  = np.load(f"{d}/expected.npy")      # [N,20] PyTorch logits
got  = np.load(f"{d}/tensora_out.npy")   # [N,20] Tensora logits
truth = meta["true_classes"]
N = exp.shape[0]

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

# A few images shown with their pictures (a diverse, correctly-classified subset).
print(f"Showing {len(meta['display'])} of {N} classified images:\n")
for item in meta["display"]:
    b = item["index"]
    thumb = np.load(f"{d}/thumb_{b}.npy")  # [H,W,3] uint8
    h, w, _ = thumb.shape
    step = max(1, w // 32)
    for y in range(0, h - step, 2 * step):
        line = "".join(
            f"\x1b[38;2;{thumb[y,x][0]};{thumb[y,x][1]};{thumb[y,x][2]}m"
            f"\x1b[48;2;{thumb[min(y+step,h-1),x][0]};{thumb[min(y+step,h-1),x][1]};{thumb[min(y+step,h-1),x][2]}m▀"
            for x in range(0, w, step))
        print("  " + line + "\x1b[0m")
    print(f"  Image (row {b} of {N}):")
    print(f"    True class:        {item['true_class']}")
    print(f"    PyTorch reference: {CLASSES[int(exp[b].argmax())]:<20} {softmax(exp[b]).max()*100:5.1f}%")
    print(f"    Tensora result:    {CLASSES[int(got[b].argmax())]:<20} {softmax(got[b]).max()*100:5.1f}%")
    print()

# Aggregate over the whole batch -- the statistically meaningful part.
agree = int((exp.argmax(1) == got.argmax(1)).sum())
max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
truth_idx = np.array([CLASSES.index(t) for t in truth])
pt_acc = float((exp.argmax(1) == truth_idx).mean())
tn_acc = float((got.argmax(1) == truth_idx).mean())
print(f"Over all {N} held-out images:")
print(f"  Tensora vs PyTorch: {agree}/{N} predictions agree")
print(f"  Max logit error:    {max_err:.2e}   mean {mean_err:.2e}   (tolerance {tol:g})")
print(f"  Accuracy:           PyTorch {pt_acc*100:.1f}%   Tensora {tn_acc*100:.1f}%")
ok = agree == N and max_err <= tol
print("  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
