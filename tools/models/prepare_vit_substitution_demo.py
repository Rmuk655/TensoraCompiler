#!/usr/bin/env python3
"""Prepare the ViT attention-substitution demo: does swapping ONE real
attention block's computation for Tensora's own recomputation change a
real model's real answer?

Every other ViT demo in this project (make demo-vit-attention) proves a purely
NUMERICAL claim: "Tensora's output tensor for this block matches
PyTorch's output tensor for this block, to within 1e-6." That is
already strong evidence of correctness, but it stops short of
answering the practical question a user actually cares about: if you
ran real inference on a real production classifier and Tensora
computed part of it, would the real-world result still be right?

This script answers that directly, end-to-end, on a genuine
classification model (not the headless google/vit-base-patch16-224-
in21k encoder the other ViT demos use, which has no classifier at
all -- this uses google/vit-base-patch16-224, fine-tuned on
ImageNet-1k, with a real 1000-way classification head):

  1. Run a real photograph through the unmodified, real HuggingFace
     model end-to-end -> baseline top-1 ImageNet label + confidence.
  2. Capture the real input hidden state and real, pretrained Q/K/V/
     output-projection weights of ONE self-attention block (same
     extraction as prepare_vit_attention_demo.py; same
     examples/vit_attention_block.tns program computes it).
  3. Run that block through the real `tensora` compiler/runtime (a
     separate process, --run examples/vit_attention_block.tns) to get
     Tensora's own recomputation of that block's output.
  4. Monkey-patch that ONE block's forward pass, in the live model, to
     return Tensora's recomputed tensor instead of running the block's
     own PyTorch code, then run the SAME photograph through the SAME
     model end-to-end again -> substituted top-1 label + confidence.
  5. Compare: same label? nearly identical confidence? nearly
     identical logits, end to end, 11 more encoder layers plus
     layernorm/classifier head downstream of the substitution point?

This is still an honest boundary: only the one attention block's
Q@K/softmax/@V/output-projection math is computed by Tensora (as
every other ViT demo states, Tensora does not yet have patch-embedding
convolution/layernorm/GELU -- to be considered in future).
Steps 1, 2, 4 and 5 above are ordinary PyTorch/HuggingFace code; step 3
is the only place Tensora's own compiled binary runs. What's new here,
versus the other ViT demos, is stitching Tensora's block output back
into a live model and observing the real-world answer survive the
substitution -- proof that Tensora's output is not just numerically
close in isolation, but functionally interchangeable inside a real
production inference pipeline.

Needs `transformers`, `safetensors`, and `pillow` in addition to
tools/models/requirements.txt, plus a BUILT tensora binary (this script
shells out to it directly, unlike the training scripts):
    pip install transformers safetensors pillow
    cmake --build build   # if not already built
    python3 tools/models/prepare_vit_substitution_demo.py
"""
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("error: PyTorch is required. Install with: pip install -r tools/models/requirements.txt")

try:
    from transformers import ViTForImageClassification, ViTImageProcessor
except ImportError:
    raise SystemExit("error: transformers is required. Install with: pip install transformers safetensors pillow")

try:
    from PIL import Image
except ImportError:
    raise SystemExit("error: pillow is required. Install with: pip install pillow")

MODEL_ID = "google/vit-base-patch16-224"  # real ImageNet-1k classifier, not the headless -in21k encoder
HEADS = 12
DK = 64
D = 768
NUM_LAYERS = 12
LAYER = int(os.environ.get("TENSORA_VIT_LAYER", "11"))
IMAGE_URL = os.environ.get(
    "TENSORA_VIT_IMAGE",
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg")
# Five visibly different photographs (real, public-domain/CC-licensed
# Wikimedia Commons photos -- see each URL's own file-description page
# for attribution/license), one per ImageNet super-category, so the
# demo shows the substitution surviving on more than one cherry-picked
# image. Overridden entirely when TENSORA_VIT_IMAGE is set (single-
# image mode, used by prepare_vit_substitution_demo.py's original
# single-photo API for backward compatibility with other callers).
GALLERY_IMAGES = {
    "dog": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "car": ("https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Porsche%20911%2C%2070%20Years%20Porsche%20Sports%20Car%2C%20Berlin%20%281X7A3904%29.jpg"),
    "elephant": "https://commons.wikimedia.org/wiki/Special:FilePath/African_Bush_Elephant.jpg",
    "airplane": "https://commons.wikimedia.org/wiki/Special:FilePath/Airbus_A320.jpg",
    "cat": "https://commons.wikimedia.org/wiki/Special:FilePath/Tabby_cat_with_blue_eyes-3336579.jpg",
}
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_DIR = os.path.join(ROOT, "demos", "artifacts", "vit_substitution")
TNS_PROGRAM = os.path.join(ROOT, "examples", "vit_attention_block.tns")


def per_head_qkv(lin):
    W = lin.weight.detach().numpy().T.astype(np.float32)  # [768, 768] = [d, head*dk]
    b = lin.bias.detach().numpy().astype(np.float32)       # [768]
    W_h = W.reshape(D, HEADS, DK).transpose(1, 0, 2)        # [head, d, dk]
    b_h = b.reshape(HEADS, DK)                              # [head, dk]
    return W_h, b_h


def per_head_o(lin):
    W = lin.weight.detach().numpy().T.astype(np.float32)  # [in=768, out=768]
    b = lin.bias.detach().numpy().astype(np.float32)       # [768]
    W_h = W.reshape(HEADS, DK, D)                           # [head, dk, d]
    return W_h, b


def find_tensora_binary():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TENSORA_BUILD_DIR", "build")
    for candidate in (os.path.join(ROOT, build_dir, "tensora"), os.path.join(ROOT, build_dir, "bin", "tensora")):
        if os.path.exists(candidate):
            return candidate
    raise SystemExit(f"error: tensora binary not found under {build_dir}/. Build it first: cmake --build {build_dir}")


def top1(logits, id2label):
    probs = torch.softmax(logits, dim=-1)[0]
    conf, idx = probs.max(dim=0)
    return id2label[int(idx)], float(conf), logits[0].detach().numpy().astype(np.float32)


def top_k(logits, id2label, k=5):
    """Returns the top-k (label, probability) pairs, for the dashboard's
    side-by-side probability bars -- not just the single winning class."""
    probs = torch.softmax(logits, dim=-1)[0]
    top_p, top_idx = probs.topk(k)
    return [(id2label[int(i)], float(p)) for p, i in zip(top_p, top_idx)]


def process_one_image(name, img_path, model, processor, id2label, tensora_bin):
    """Runs the full capture -> Tensora recompute -> live-model
    substitution -> compare pipeline (steps 1-4 in the module
    docstring) for ONE photograph, and returns a summary dict plus the
    raw image bytes (for the dashboard)."""
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # --- Step 1: baseline, fully-real forward pass. ---
    with torch.no_grad():
        baseline_out = model(**inputs)
    base_label, base_conf, base_logits = top1(baseline_out.logits, id2label)
    base_top5 = top_k(baseline_out.logits, id2label)
    print(f"\n[{name}] Baseline (100% real PyTorch/HuggingFace): {base_label!r}  confidence={base_conf:.4f}")

    # --- Step 2: capture layer LAYER's real input + real weights. ---
    captured = {}

    def capture_hook(_module, args, output):
        captured["in"] = args[0].detach()
        captured["out"] = (output[0] if isinstance(output, tuple) else output).detach()

    target_attn = model.vit.layers[LAYER].attention
    handle = target_attn.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    X = captured["in"].numpy().astype(np.float32)  # [1, seq, 768]
    seq = X.shape[1]
    Wq, bq = per_head_qkv(target_attn.q_proj)
    Wk, bk = per_head_qkv(target_attn.k_proj)
    Wv, bv = per_head_qkv(target_attn.v_proj)
    Wo, bo = per_head_o(target_attn.o_proj)

    eval_dir = os.path.join(OUT_DIR, "eval", name)
    os.makedirs(eval_dir, exist_ok=True)
    x_h1 = X[:, None, :, :]
    np.save(os.path.join(eval_dir, "Xq.npy"), x_h1)
    np.save(os.path.join(eval_dir, "Xk.npy"), x_h1)
    np.save(os.path.join(eval_dir, "Wq.npy"), Wq.reshape(1, HEADS, D, DK))
    np.save(os.path.join(eval_dir, "bq.npy"), bq.reshape(HEADS, 1, DK))
    np.save(os.path.join(eval_dir, "Wk.npy"), Wk.reshape(1, HEADS, D, DK))
    np.save(os.path.join(eval_dir, "bk.npy"), bk.reshape(HEADS, 1, DK))
    np.save(os.path.join(eval_dir, "Wv.npy"), Wv.reshape(1, HEADS, D, DK))
    np.save(os.path.join(eval_dir, "bv.npy"), bv.reshape(HEADS, 1, DK))
    np.save(os.path.join(eval_dir, "Wo.npy"), Wo.reshape(1, HEADS, DK, D))
    np.save(os.path.join(eval_dir, "bo.npy"), bo)
    np.save(os.path.join(eval_dir, "expected.npy"), captured["out"].numpy().astype(np.float32))
    json.dump({"model": MODEL_ID, "layer": LAYER, "num_layers": NUM_LAYERS,
               "seq_len": int(seq), "hidden_size": D, "heads": HEADS},
              open(os.path.join(eval_dir, "meta.json"), "w"), indent=2)
    print(f"[{name}] Captured layer {LAYER}'s real input + real pretrained weights -> {os.path.normpath(eval_dir)}")

    # --- Step 3: run the real Tensora compiler/runtime on that block. ---
    tensora_out_path = os.path.join(eval_dir, "tensora_out.npy")
    print(f"[{name}] Running the real tensora binary on {os.path.relpath(TNS_PROGRAM, ROOT)} ...")
    subprocess.run([tensora_bin, "--run", TNS_PROGRAM, "--data-dir", eval_dir,
                    "--output", tensora_out_path], check=True, capture_output=True)
    tensora_block_out = np.load(tensora_out_path)  # [1, seq, 768], Tensora's own recomputation
    block_err = float(np.max(np.abs(tensora_block_out - captured["out"].numpy())))
    print(f"[{name}] Tensora's block output vs PyTorch's own block output: max abs error {block_err:.2e}")

    # --- Step 4: splice Tensora's block output into the LIVE model and
    # re-run the SAME photograph end-to-end. ---
    tensora_tensor = torch.from_numpy(tensora_block_out.astype(np.float32))

    class SubstitutedAttention(nn.Module):
        """Drop-in replacement for one ViTAttention block: returns
        Tensora's precomputed tensor instead of executing any of this
        block's own Q/K/V/softmax/output-projection code. Every other
        layer of the real model (patch embedding, the other 11
        attention blocks, MLP blocks, layernorms, the classifier head)
        runs completely unmodified."""

        def __init__(self, precomputed):
            super().__init__()
            self.precomputed = precomputed

        def forward(self, hidden_states, head_mask=None, output_attentions=False):
            return (self.precomputed, None)

    original_attention = model.vit.layers[LAYER].attention
    model.vit.layers[LAYER].attention = SubstitutedAttention(tensora_tensor)
    with torch.no_grad():
        substituted_out = model(**inputs)
    model.vit.layers[LAYER].attention = original_attention  # restore for the next image in the gallery

    sub_label, sub_conf, sub_logits = top1(substituted_out.logits, id2label)
    sub_top5 = top_k(substituted_out.logits, id2label)
    print(f"[{name}] Substituted (layer {LAYER}'s attention computed by Tensora): "
          f"{sub_label!r}  confidence={sub_conf:.4f}")

    logit_err = float(np.max(np.abs(base_logits - sub_logits)))
    print(f"[{name}] Final-logit max abs error (baseline vs substituted, end-to-end): {logit_err:.2e}")
    return {
        "name": name, "image_path": os.path.relpath(img_path, ROOT),
        "model": MODEL_ID, "substituted_layer": LAYER, "num_layers": NUM_LAYERS,
        "block_output_max_abs_error": block_err,
        "baseline_label": base_label, "baseline_confidence": base_conf, "baseline_top5": base_top5,
        "substituted_label": sub_label, "substituted_confidence": sub_conf, "substituted_top5": sub_top5,
        "final_logits_max_abs_error": logit_err,
        "label_unchanged": base_label == sub_label,
    }


def main():
    torch.manual_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Single-photo mode (TENSORA_VIT_IMAGE set): only that one image is
    # processed, matching this script's original API. Default mode: the
    # five-photo gallery (dog/car/elephant/airplane/cat), so the demo
    # shows the substitution surviving across visibly different real
    # photographs, not one cherry-picked image.
    single_override = os.environ.get("TENSORA_VIT_IMAGE")
    images = {"custom": single_override} if single_override else GALLERY_IMAGES

    downloaded = {}
    for name, src in images.items():
        if src.startswith("http://") or src.startswith("https://"):
            path = os.path.join(OUT_DIR, "images", f"{name}.jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                print(f"Downloading a real sample photograph ({name}) from {src} ...")
                # Wikimedia (and some other hosts) reject the default
                # urllib User-Agent with a 403; identify honestly instead.
                req = urllib.request.Request(src, headers={"User-Agent": "TensoraCompilerDemo/1.0"})
                with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
                    f.write(resp.read())
        else:
            path = src
        downloaded[name] = path

    print(f"Loading real pretrained ImageNet-1k classifier: {MODEL_ID} (one-time download, ~350MB) ...")
    model = ViTForImageClassification.from_pretrained(MODEL_ID, attn_implementation="eager")
    model.eval()
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    id2label = model.config.id2label
    tensora_bin = find_tensora_binary()

    results = [process_one_image(name, path, model, processor, id2label, tensora_bin)
               for name, path in downloaded.items()]

    json.dump(results, open(os.path.join(OUT_DIR, "gallery_summary.json"), "w"), indent=2)
    # Backward-compatible single-image summary.json: the first (or only)
    # image's result, for callers/scripts written against the original API.
    json.dump(results[0], open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)

    n_ok = sum(1 for r in results if r["label_unchanged"])
    print(f"\n{n_ok}/{len(results)} images: predicted label unchanged after substitution.")
    print(f"Wrote per-image gallery summary to {os.path.normpath(os.path.join(OUT_DIR, 'gallery_summary.json'))}")


if __name__ == "__main__":
    main()
