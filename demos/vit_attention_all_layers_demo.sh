#!/usr/bin/env bash
# Independent per-layer verification: run examples/vit_attention_block.tns
# through the real `tensora` CLI once per encoder layer of a real,
# open-source pretrained Vision Transformer (google/vit-base-patch16-
# 224-in21k), each time with that layer's own real input hidden state
# and own real Q/K/V/output-projection weights, and check Tensora's
# output against that layer's own real output.
#
# This is stronger evidence than checking a single cherry-picked layer
# (make demo-vit-attention-one-layer only checks the last one): it shows Tensora
# independently reproduces the self-attention computation of EVERY one
# of the 12 encoder layers, not just one.
#
# Only numpy is needed here (no torch/transformers); run
# tools/models/prepare_vit_attention_demo.py once first (needs
# PyTorch + transformers + pillow, and internet access to download the
# ~350MB checkpoint) -- it exports one artifact directory per layer
# under demos/artifacts/vit_attn/eval_all_layers/layer_NN/.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/vit_attn/eval_all_layers"
TOL="1e-4"

if [ ! -d "$ART" ]; then
  echo "error: per-layer demo artifacts not found in $ART/."
  echo "Run the one-time preparation first (needs PyTorch + transformers + pillow):"
  echo "  pip install transformers safetensors pillow"
  echo "  python3 tools/models/prepare_vit_attention_demo.py"
  exit 1
fi

echo "=== Tensora Real-Pretrained-Transformer Demo: ALL 12 attention layers, independently ==="
echo "Model: google/vit-base-patch16-224-in21k (12 heads, 768-dim, 197 positions)"
echo "Program: examples/vit_attention_block.tns, run once per encoder layer"
echo

overall_ok=0
printf "%-8s %-14s %-6s\n" "Layer" "Max abs error" "Result"
printf "%-8s %-14s %-6s\n" "-----" "-------------" "------"

for layer_dir in "$ART"/layer_*; do
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
  ) || overall_ok=1
  max_err=$(echo "$result" | cut -d' ' -f1)
  status=$(echo "$result" | cut -d' ' -f2)
  printf "%-8s %-14s %-6s\n" "$layer" "$max_err" "$status"
done

echo
if [ "$overall_ok" -eq 0 ]; then
  echo "PASS 12/12 -- Tensora independently reproduces the real pretrained model's"
  echo "self-attention computation at every one of its 12 encoder layers, each"
  echo "checked against that layer's own real input hidden state, own real"
  echo "pretrained weights, and own real output (not just one chosen layer)."
else
  echo "FAIL -- at least one layer did not match within tolerance $TOL"
fi
exit "$overall_ok"
