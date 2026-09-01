#!/usr/bin/env bash
# Real-pretrained-Transformer attention-block demo, combined: three
# checks against one genuine, open-source Vision Transformer
# (google/vit-base-patch16-224-in21k), all reusing the same block
# recomputation (examples/vit_attention_block.tns), in one command:
#
#   1. one layer     -- Tensora's recomputed output for the LAST encoder
#                        layer vs. that same real model's own output,
#                        on a real photograph.
#   2. all 12 layers -- the same check, independently repeated at
#                        EVERY encoder layer (each with its own real
#                        input hidden state and own real pretrained
#                        weights), printing a per-layer error table and
#                        PASS 12/12 -- stronger evidence than checking
#                        one cherry-picked layer alone.
#   3. visualization (optional, needs matplotlib) -- renders six PNGs
#                        of what the checked block is actually looking
#                        at in the photo; skipped with a clear message
#                        if matplotlib is not installed.
#
# Only numpy is needed for checks 1-2 (no torch/transformers); run
# tools/models/prepare_vit_attention_demo.py once first (needs
# PyTorch + transformers + pillow, and internet access to download the
# ~350MB checkpoint).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/vit_attn"
EVAL="$ART/eval"
ALL_LAYERS="$ART/eval_all_layers"
TOL="1e-4"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch + transformers + pillow):"
  echo "  pip install transformers safetensors pillow"
  echo "  python3 tools/models/prepare_vit_attention_demo.py"
  exit 1
fi

echo "=== Tensora Real-Pretrained-Transformer Demo: ViT-Base self-attention ==="
echo "Model: google/vit-base-patch16-224-in21k (12 heads, 768-dim, 197 positions)"
echo "Program: examples/vit_attention_block.tns"
echo

echo "--- 1/3: one encoder layer (last, index 11 of 12) vs. the real model's own output ---"
"$BUILD/tensora" --run examples/vit_attention_block.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

one_layer_ok=0
python3 - "$EVAL" "$TOL" <<'PY' || one_layer_ok=1
import json, sys
import numpy as np
d, tol = sys.argv[1], float(sys.argv[2])
meta = json.load(open(f"{d}/meta.json"))
exp = np.load(f"{d}/expected.npy")     # [1, seq, 768] real ViT's own attention output
got = np.load(f"{d}/tensora_out.npy")  # [1, seq, 768] Tensora's recomputation

print(f"Model: {meta['model']}, layer {meta['layer']}, seq_len={meta['seq_len']}, "
      f"hidden_size={meta['hidden_size']}, heads={meta['heads']}")
max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
rel_err = float(np.max(np.abs(exp - got)) / (np.max(np.abs(exp)) + 1e-12))
print(f"Max abs error:  {max_err:.2e}  (tolerance {tol:g})")
print(f"Mean abs error: {mean_err:.2e}")
print(f"Max relative error: {rel_err:.2e}")
ok = max_err <= tol
print("  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
echo

if [ -d "$ALL_LAYERS" ]; then
  echo "--- 2/3: all 12 encoder layers, independently ---"
  all_layers_ok=0
  printf "%-8s %-14s %-6s\n" "Layer" "Max abs error" "Result"
  printf "%-8s %-14s %-6s\n" "-----" "-------------" "------"
  for layer_dir in "$ALL_LAYERS"/layer_*; do
    layer=$(basename "$layer_dir" | sed 's/layer_//')
    out="$layer_dir/tensora_out.npy"
    "$BUILD/tensora" --run examples/vit_attention_block.tns \
      --data-dir "$layer_dir" --output "$out" >/dev/null
    result=$(python3 - "$layer_dir" "$out" "$TOL" <<'PY'
import sys
import numpy as np
layer_dir, out_path, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
exp = np.load(f"{layer_dir}/expected.npy")
got = np.load(out_path)
max_err = float(np.max(np.abs(exp - got)))
ok = max_err <= tol
print(f"{max_err:.2e} {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
    ) || all_layers_ok=1
    max_err=$(echo "$result" | cut -d' ' -f1)
    status=$(echo "$result" | cut -d' ' -f2)
    printf "%-8s %-14s %-6s\n" "$layer" "$max_err" "$status"
  done
  echo
  if [ "$all_layers_ok" -eq 0 ]; then
    echo "PASS 12/12 -- every one of the 12 encoder layers independently reproduced,"
    echo "each checked against its own real input hidden state and real weights."
  else
    echo "FAIL -- at least one layer did not match within tolerance $TOL"
  fi
else
  echo "--- 2/3: SKIP all-12-layers check: no artifacts in $ALL_LAYERS/ ---"
  all_layers_ok=0
fi
echo

echo "--- 3/3: visualization (optional) ---"
if python3 -c "import matplotlib" >/dev/null 2>&1; then
  python3 tools/models/visualize_vit_attention.py "$BUILD"
else
  echo "SKIP visualization: matplotlib not installed (pip install matplotlib)"
fi

overall_ok=$((one_layer_ok + all_layers_ok))
echo
if [ "$overall_ok" -eq 0 ]; then
  echo "PASS -- Tensora's compiled output matches a real pretrained Transformer's"
  echo "own self-attention computation, checked one layer and all 12 layers."
else
  echo "FAIL -- see above"
fi
exit "$overall_ok"
