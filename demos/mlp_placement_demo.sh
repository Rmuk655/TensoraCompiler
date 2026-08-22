#!/usr/bin/env bash
# MLP real-inference demo: student placement-eligibility.
#
# Classifies a whole held-out batch of 100 students
# (examples/mlp_placement.tns) through the full Tensora pipeline (--run,
# JIT) in one go, and checks every row against the PyTorch reference
# saved at prep time -- so the agreement is statistically meaningful, not
# just 3 examples. Only numpy is needed here (no torch); run
# tools/models/prepare_mlp_demo.py once first.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/mlp"
EVAL="$ART/eval"
TOL="1e-4"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch):"
  echo "  python3 tools/models/prepare_mlp_demo.py"
  exit 1
fi

echo "=== Tensora MLP Real-World Demo: student placement-eligibility ==="
echo "Model: examples/mlp_placement.tns (linear -> relu -> linear)"
echo

"$BUILD/tensora" --run examples/mlp_placement.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

python3 - "$EVAL" "$TOL" <<'PY'
import json, sys
import numpy as np
d, tol = sys.argv[1], float(sys.argv[2])
meta = json.load(open(f"{d}/meta.json"))
exp = np.load(f"{d}/expected.npy")        # [N,2] PyTorch logits
got = np.load(f"{d}/tensora_out.npy")     # [N,2] Tensora logits
labels = ["not eligible", "placement-eligible"]
truth = meta["true_labels"]
N = exp.shape[0]

# A few readable individual predictions (a clear pass, a borderline, and
# a clear fail; the full batch is scored in aggregate below).
print(f"Showing {len(meta['display'])} of {N} classified students (a pass, a borderline, a fail):\n")
for row in meta["display"]:
    i = row["index"]
    print(f"Student (row {i} of {N}):")
    for k, v in row["features"].items():
        print(f"    {k:<12} {v}")
    print(f"  True label:        {row['true_label']}")
    print(f"  PyTorch reference: {labels[int(exp[i].argmax())]}")
    print(f"  Tensora result:    {labels[int(got[i].argmax())]}")
    print()

# Aggregate over the whole batch -- the statistically meaningful part.
agree = int((exp.argmax(1) == got.argmax(1)).sum())
max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
pt_acc = float((exp.argmax(1) == np.array([labels.index(t) for t in truth])).mean())
tn_acc = float((got.argmax(1) == np.array([labels.index(t) for t in truth])).mean())
print(f"Over all {N} held-out students:")
print(f"  Tensora vs PyTorch: {agree}/{N} predictions agree")
print(f"  Max logit error:    {max_err:.2e}   mean {mean_err:.2e}   (tolerance {tol:g})")
print(f"  Accuracy:           PyTorch {pt_acc*100:.1f}%   Tensora {tn_acc*100:.1f}%")
ok = agree == N and max_err <= tol
print("  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
