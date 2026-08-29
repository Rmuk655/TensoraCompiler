#!/usr/bin/env bash
# ViT attention-SUBSTITUTION demo: does the real-world answer survive
# swapping one real attention block's computation for Tensora's own?
#
# Every other ViT demo (demo-vit-attention-one-layer,
# demo-vit-attention-all-layers, demo-vit-attention-viz) proves a
# numerical claim in isolation: Tensora's output tensor for one
# attention block matches PyTorch's own output tensor for that block.
# This demo goes one step further and asks the practical question:
# splice Tensora's recomputed block back into a REAL, LIVE ImageNet-1k
# classifier (google/vit-base-patch16-224) mid-inference, run the same
# photograph through the rest of the (unmodified) 12-layer model, and
# check whether the model's real answer -- its predicted label and
# confidence -- is unchanged.
#
# Honesty boundary (see tools/models/prepare_vit_substitution_demo.py
# for the full explanation): only ONE self-attention block's math
# (Q@K/softmax/@V/output-projection) is computed by Tensora's own
# compiled binary. The patch-embedding convolution, the other 11
# attention blocks, all MLP blocks, every layernorm, and the final
# classifier head are the real, unmodified HuggingFace model -- this
# demo does not claim Tensora runs the whole ViT.
#
# Runs the substitution on FIVE visibly different real photographs
# (dog, car, elephant, airplane, cat) -- not just one -- so the result
# is not a lucky single case. If matplotlib is available, also renders
# a visual dashboard per photograph plus a one-row summary gallery to
# demos/artifacts/vit_substitution/*.png.
#
# This script needs PyTorch + transformers + pillow + a built tensora
# binary (it downloads/caches the ~350MB checkpoint plus five small
# photographs once, then re-runs fast from local caches):
#   pip install transformers safetensors pillow matplotlib
#   cmake --build build
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
ART="demos/artifacts/vit_substitution"
GALLERY_SUMMARY="$ART/gallery_summary.json"

echo "=== Tensora ViT Attention-Substitution Demo ==="
echo "Model: google/vit-base-patch16-224 (real ImageNet-1k classifier, 1000 classes)"
echo "Substituting: encoder layer 11's self-attention block (last of 12) with Tensora's own recomputation"
echo "Program: examples/vit_attention_block.tns"
echo "Photographs: dog, car, elephant, airplane, cat (5 real, visibly different images)"
echo

python3 tools/models/prepare_vit_substitution_demo.py "$BUILD"

set +e
python3 - "$GALLERY_SUMMARY" <<'PY'
import json, sys
results = json.load(open(sys.argv[1]))

print()
print("--- Result ---")
n_ok = 0
for r in results:
    ok = r["label_unchanged"]
    n_ok += ok
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {r['name']:>9s}: baseline={r['baseline_label']!r} "
          f"({r['baseline_confidence']:.4f})  substituted={r['substituted_label']!r} "
          f"({r['substituted_confidence']:.4f})  "
          f"block err={r['block_output_max_abs_error']:.2e}  "
          f"logit err={r['final_logits_max_abs_error']:.2e}")
print()
print(f"{n_ok}/{len(results)} photographs: predicted label unchanged after substitution.")
if n_ok == len(results):
    print("PASS -- swapping one real attention block for Tensora's own compiled computation, "
          "mid-inference inside a real production classifier, did not change the real-world "
          "answer on any of the five photographs.")
else:
    print("FAIL -- predicted label changed after substitution on at least one photograph.")
sys.exit(0 if n_ok == len(results) else 1)
PY
STATUS=$?
set -e

if python3 -c "import matplotlib" >/dev/null 2>&1; then
    echo
    echo "--- Rendering visual dashboard ---"
    python3 tools/models/visualize_vit_substitution.py
else
    echo
    echo "(matplotlib not installed -- skipping visual dashboard; 'pip install matplotlib' to enable)"
fi

exit "$STATUS"
