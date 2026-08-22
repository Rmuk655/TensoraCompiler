#!/usr/bin/env bash
# Transformer/attention real-inference demo: self-attention text
# classifier.
#
# Classifies a whole batch of 30 held-out sentences
# (examples/attn_text_classifier.tns) through the full Tensora pipeline
# (--run, JIT) in one go, and checks every row against the PyTorch
# reference saved at prep time -- so the agreement is statistically
# meaningful, not just 3 examples. Only numpy is needed here (no torch);
# run tools/models/prepare_text_attn_demo.py once first.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/attn"
EVAL="$ART/eval"
TOL="1e-4"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch):"
  echo "  python3 tools/models/prepare_text_attn_demo.py"
  exit 1
fi

echo "=== Tensora Attention Real-World Demo: text topic classification ==="
echo "Model: examples/attn_text_classifier.tns (self-attention -> mean pool -> linear -> relu -> linear)"
echo

"$BUILD/tensora" --run examples/attn_text_classifier.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

python3 - "$EVAL" "$TOL" <<'PY'
import json, sys
import numpy as np
d, tol = sys.argv[1], float(sys.argv[2])
TOPICS = ["weather", "food", "sports"]
meta = json.load(open(f"{d}/meta.json"))
exp  = np.load(f"{d}/expected.npy")      # [N,3] PyTorch logits
got  = np.load(f"{d}/tensora_out.npy")   # [N,3] Tensora logits
sents = meta["sentences"]
truth = meta["true_topics"]
N = exp.shape[0]

# A few readable individual predictions: one of each topic for variety,
# then fill with evenly-spaced rows so the shown sentences are spread
# across the batch rather than adjacent.
disp, seen = [], set()
for i, t in enumerate(truth):
    if t not in seen:
        seen.add(t); disp.append(i)
for i in [round(k * (N - 1) / 4) for k in range(5)]:
    if len(disp) >= 5:
        break
    if i not in disp:
        disp.append(i)
disp = sorted(disp)[:5]
print(f"Showing {len(disp)} of {N} classified sentences (across topics):\n")
for i in disp:
    print(f'Sentence (row {i} of {N}): "{sents[i]}"')
    print(f"  True topic:        {truth[i]}")
    print(f"  PyTorch reference: {TOPICS[int(exp[i].argmax())]}")
    print(f"  Tensora result:    {TOPICS[int(got[i].argmax())]}")
    print()

# Aggregate over the whole batch -- the statistically meaningful part.
agree = int((exp.argmax(1) == got.argmax(1)).sum())
max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
truth_idx = np.array([TOPICS.index(t) for t in truth])
pt_acc = float((exp.argmax(1) == truth_idx).mean())
tn_acc = float((got.argmax(1) == truth_idx).mean())
print(f"Over all {N} held-out sentences:")
print(f"  Tensora vs PyTorch: {agree}/{N} predictions agree")
print(f"  Max logit error:    {max_err:.2e}   mean {mean_err:.2e}   (tolerance {tol:g})")
print(f"  Accuracy:           PyTorch {pt_acc*100:.1f}%   Tensora {tn_acc*100:.1f}%")
ok = agree == N and max_err <= tol
print("  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
