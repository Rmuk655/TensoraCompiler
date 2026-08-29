#!/usr/bin/env bash
# Real-pretrained-Transformer attention-block demo: one self-attention
# block from google/vit-base-patch16-224-in21k (a genuine, open-source
# Vision Transformer, HuggingFace/Google), run through the real
# `tensora` pipeline with the block's real ImageNet-21k-pretrained
# weights, and checked against the real model's own output for that
# block on a real photograph.
#
# Only numpy is needed here (no torch/transformers); run
# tools/models/prepare_vit_attention_demo.py once first (needs
# PyTorch + transformers + pillow, and internet access to download the
# ~350MB checkpoint).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/vit_attn"
EVAL="$ART/eval"
TOL="1e-4"

if [ ! -d "$EVAL" ]; then
  echo "error: demo artifacts not found in $EVAL/."
  echo "Run the one-time preparation first (needs PyTorch + transformers + pillow):"
  echo "  pip install transformers safetensors pillow"
  echo "  python3 tools/models/prepare_vit_attention_demo.py"
  exit 1
fi

echo "=== Tensora Real-Pretrained-Transformer Demo: one ViT-Base self-attention block ==="
echo "Model: google/vit-base-patch16-224-in21k, last encoder layer (index 11 of 12; 12 heads, 768-dim, 197 positions)"
echo "Program: examples/vit_attention_block.tns"
echo

"$BUILD/tensora" --run examples/vit_attention_block.tns --data-dir "$EVAL" --output "$EVAL/tensora_out.npy" >/dev/null

python3 - "$EVAL" "$TOL" <<'PY'
import json, sys
import numpy as np
d, tol = sys.argv[1], float(sys.argv[2])
meta = json.load(open(f"{d}/meta.json"))
exp = np.load(f"{d}/expected.npy")     # [1, seq, 768] real ViT's own attention output
got = np.load(f"{d}/tensora_out.npy")  # [1, seq, 768] Tensora's recomputation

print(f"Model: {meta['model']}, layer {meta['layer']}, seq_len={meta['seq_len']}, "
      f"hidden_size={meta['hidden_size']}, heads={meta['heads']}")
print(f"Output shape: PyTorch {exp.shape}, Tensora {got.shape}\n")

max_err = float(np.max(np.abs(exp - got)))
mean_err = float(np.mean(np.abs(exp - got)))
rel_err = float(np.max(np.abs(exp - got)) / (np.max(np.abs(exp)) + 1e-12))
print("Sample values (position 0, first 5 of 768 channels):")
print(f"  PyTorch (real ViT): {exp[0, 0, :5]}")
print(f"  Tensora:            {got[0, 0, :5]}")
print()
print(f"Max abs error:  {max_err:.2e}  (tolerance {tol:g})")
print(f"Mean abs error: {mean_err:.2e}")
print(f"Max relative error: {rel_err:.2e}")
ok = max_err <= tol
print("  " + ("PASS -- Tensora's output matches a real pretrained Transformer's own attention block"
              if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
