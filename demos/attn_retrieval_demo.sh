#!/usr/bin/env bash
# Multi-head attention real-inference demo: Associative Recall (AR),
# single-query form (4 heads).
#
# A different task from attn_text_classifier_demo.sh's single-head
# topic classification -- chosen to give multi-head attention
# something distinctive to show. Each of 30 held-out sequences is 6
# shuffled (key, value) slots followed by a query naming one key; the
# model must retrieve the value bound to that key *in this specific
# sequence* (the binding is reshuffled every time, so it cannot be
# memorized -- see examples/attn_multihead_retrieval.tns's header for
# why this is the standard "Associative Recall" benchmark task
# (Ba et al. 2016; Arora et al.'s "Zoology", arXiv:2312.04927) and how
# it demonstrates the same "read the right earlier position"
# mechanism a real language model uses for next-token prediction).
# Runs the full Tensora pipeline (--run, JIT) in one go and checks
# every row against the PyTorch reference saved at prep time. Only
# numpy is needed here (no torch); run
# tools/models/prepare_multihead_attn_demo.py once first.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/attn_mh"
EVAL="$ART/eval"
TOL="1e-4"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch):"
  echo "  python3 tools/models/prepare_multihead_attn_demo.py"
  exit 1
fi

echo "=== Tensora Multi-Head Attention Real-World Demo: Associative Recall (single-query) ==="
echo "Model: examples/attn_multihead_retrieval.tns (4-head self-attention -> select query position -> linear -> relu -> linear)"
echo

"$BUILD/tensora" --run examples/attn_multihead_retrieval.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

python3 - "$EVAL" "$TOL" <<'PY'
import json, sys
import numpy as np
d, tol = sys.argv[1], float(sys.argv[2])
VALUES = ["one", "two", "three", "four", "five", "six"]
meta = json.load(open(f"{d}/meta.json"))
exp  = np.load(f"{d}/expected.npy")      # [N,6] PyTorch logits
got  = np.load(f"{d}/tensora_out.npy")   # [N,6] Tensora logits
seqs = meta["sequences"]
N = exp.shape[0]

disp = sorted({round(k * (N - 1) / 4) for k in range(5)})
print(f"Showing {len(disp)} of {N} retrieval queries:\n")
for i in disp:
    s = seqs[i]
    print(f"Sequence (row {i} of {N}): {' '.join(s['slots'])}  ?query={s['query_key']}")
    print(f"  True value:        {s['true_value']}")
    print(f"  PyTorch reference: {VALUES[int(exp[i].argmax())]}")
    print(f"  Tensora result:    {VALUES[int(got[i].argmax())]}")
    print()

agree = int((exp.argmax(1) == got.argmax(1)).sum())
max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
truth_idx = np.array([VALUES.index(s["true_value"]) for s in seqs])
pt_acc = float((exp.argmax(1) == truth_idx).mean())
tn_acc = float((got.argmax(1) == truth_idx).mean())
print(f"Over all {N} held-out queries:")
print(f"  Tensora vs PyTorch: {agree}/{N} predictions agree")
print(f"  Max logit error:    {max_err:.2e}   mean {mean_err:.2e}   (tolerance {tol:g})")
print(f"  Retrieval accuracy: PyTorch {pt_acc*100:.1f}%   Tensora {tn_acc*100:.1f}%")
ok = agree == N and max_err <= tol
print("  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
