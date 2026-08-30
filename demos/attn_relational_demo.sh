#!/usr/bin/env bash
# Two-hop relational-reasoning real-inference demo: course eligibility.
#
# A two-hop relational-reasoning task, not a single lookup: the
# answer (which course is now eligible) is never a fact stated
# directly in the input (see examples/attn_relational_eligibility.tns's
# header for the full task and why it needs two chained
# attention/lookup steps instead of one).
# Reports the model's own accuracy against ground truth *and*
# Tensora-vs-PyTorch agreement separately -- the two are different
# questions. This task's PyTorch model is honestly imperfect
# (~55-60% held-out, see tools/models/prepare_relational_demo.py's
# docstring); what this demo actually proves is that Tensora
# reproduces that same, imperfect model exactly, not that the task is
# solved perfectly. Only numpy is needed here (no torch); run
# tools/models/prepare_relational_demo.py once first.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/attn_rel"
EVAL="$ART/eval"
TOL="1e-4"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch):"
  echo "  python3 tools/models/prepare_relational_demo.py"
  exit 1
fi

echo "=== Tensora Relational-Reasoning Real-World Demo: Course Eligibility (2-hop) ==="
echo "Model: examples/attn_relational_eligibility.tns (attend over completed courses -> per-course coverage/already checks -> combine)"
echo

"$BUILD/tensora" --run examples/attn_relational_eligibility.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

python3 - "$EVAL" "$TOL" <<'PY'
import json, sys
import numpy as np
d, tol = sys.argv[1], float(sys.argv[2])
meta = json.load(open(f"{d}/meta.json"))
exp  = np.load(f"{d}/expected.npy")      # [N,6] PyTorch logits
got  = np.load(f"{d}/tensora_out.npy")   # [N,6] Tensora logits
examples = meta["examples"]
N = exp.shape[0]

truth_idx_all = np.array([ex["candidates"].index(ex["target"]) for ex in examples])
pt_pred_all = exp.argmax(1)
correct_rows = [i for i in range(N) if pt_pred_all[i] == truth_idx_all[i]]
wrong_rows   = [i for i in range(N) if pt_pred_all[i] != truth_idx_all[i]]
# Show a mix of rows the model gets right and rows it gets wrong -- since
# accuracy is well under 100%, picking rows by position alone (e.g. every
# 50th) can land entirely on one bucket by chance and misleadingly suggest
# the model always succeeds or always fails.
disp = sorted(correct_rows[:3] + wrong_rows[:2])
print(f"Showing {len(disp)} of {N} eligibility queries (real generated sentences, not symbols; "
      f"a mix of correct and incorrect predictions, not cherry-picked):\n")
for i in disp:
    ex = examples[i]
    candidates = ex["candidates"]
    student = ex["student"]
    print(f"Query (row {i} of {N}):")
    for s in ex["completed_sentences"]:
        print(f"    {s}")
    for c in candidates:
        print(f"    {ex['prerequisite_sentences'][c]}  ({ex['identity_sentences'][c]})")
    print(f"  Question: Which of {{{', '.join(candidates)}}} can {student} now take?")
    print(f"  Ground truth:      {ex['target']}")
    print(f"  PyTorch reference: {candidates[int(exp[i].argmax())]}")
    print(f"  Tensora result:    {candidates[int(got[i].argmax())]}")
    print()

agree = int((exp.argmax(1) == got.argmax(1)).sum())
max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
pt_acc = float((exp.argmax(1) == truth_idx_all).mean())
tn_acc = float((got.argmax(1) == truth_idx_all).mean())
print(f"Over all {N} held-out queries:")
print(f"  Tensora vs PyTorch:   {agree}/{N} predictions agree")
print(f"  Max logit error:      {max_err:.2e}   mean {mean_err:.2e}   (tolerance {tol:g})")
print(f"  Eligibility accuracy: PyTorch {pt_acc*100:.1f}%   Tensora {tn_acc*100:.1f}%  (task is genuinely hard -- see script header, not a bug)")
ok = agree == N and max_err <= tol
print("  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
