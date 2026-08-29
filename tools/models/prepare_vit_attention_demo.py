#!/usr/bin/env python3
"""Prepare the real-pretrained-ViT attention-block validation demo.

Downloads google/vit-base-patch16-224-in21k (a genuine, open-source
Vision Transformer pretrained by Google on ~14M ImageNet-21k images --
no retraining here, no synthetic weights), runs one real photograph
through it up to encoder layer LAYER's self-attention block, and
exports:

  demos/artifacts/vit_attn/eval/
      Xq.npy Xk.npy                    -> the block's real input hidden
                                           state (identical data under
                                           two axis names -- see the
                                           .tns file's header)
      Wq.npy bq.npy Wk.npy bk.npy
      Wv.npy bv.npy Wo.npy bo.npy      -> the block's real, pretrained
                                           Q/K/V/output-projection
                                           weights and biases
      expected.npy                     -> the real ViT model's own
                                           output for this block (not
                                           a hand re-derivation: it is
                                           captured directly via a
                                           forward hook on the live
                                           HuggingFace module)
      expected_attn_weights.npy        -> the real post-softmax
                                           attention matrix
                                           [1, heads, seq, seq] for this
                                           block, also from the hook
                                           (attn_implementation="eager"
                                           so PyTorch actually returns
                                           it) -- used by
                                           examples/vit_attention_weights.tns
                                           and
                                           tools/models/visualize_vit_attention.py
                                           for an attention-heatmap-over-
                                           the-real-photo visualization
      all_layers_attn_weights.npy      -> the real post-softmax
                                           attention matrix for EVERY
                                           encoder layer,
                                           [num_layers, heads, seq, seq]
                                           -- computed directly by the
                                           real model (not Tensora; only
                                           layer LAYER is independently
                                           recomputed and checked by
                                           Tensora), used only for the
                                           multi-layer attention-rollout
                                           visualization (an established
                                           interpretability technique,
                                           Abnar & Zuidema 2020), which
                                           needs every layer's attention
                                           to trace how the [CLS] token's
                                           attention composes across
                                           depth

  demos/artifacts/vit_attn/eval_all_layers/layer_00/ .. layer_11/
      The SAME set of files as eval/ above (Xq.npy, weights, biases,
      expected.npy, meta.json) but captured independently for EVERY
      one of the 12 encoder layers, not only LAYER. Each layer gets
      its own real input hidden state and its own real, pretrained
      Q/K/V/output-projection weights -- used by
      demos/vit_attention_all_layers_demo.sh to run
      examples/vit_attention_block.tns through Tensora once per layer
      and report a per-layer PASS/FAIL table, so the claim is "Tensora
      independently reproduces the self-attention computation of every
      encoder layer of this real pretrained model", not just one
      cherry-picked layer.

Layer LAYER is chosen deliberately, not layer 0: early ViT layers tend
to attend to low-level, local texture (see docs/testing/TEST_MATRIX.md
E2E-008 note); a late layer (this project uses the LAST encoder layer)
produces the much more legible, semantically-focused-on-the-subject
attention pattern that makes a heatmap demo actually readable. This is
purely a choice of *which* real block to extract -- Tensora recomputes
whichever layer is configured exactly the same way, and the numerical
check does not depend on which layer is chosen.

The eval dir is directly usable as `tensora --run
examples/vit_attention_block.tns --data-dir <dir>`: Tensora recomputes
this one block from the real input and real weights, and the demo
checks its output against the real model's own output for the exact
same block. This tests a full-size (768-dim, 12-head, 197-position)
production Transformer's real attention block, not a toy-sized one --
the same math this project already validates at 4-head/32-dim scale
in examples/attn_multihead_retrieval.tns, now checked at
production scale against genuine pretrained parameters.

Only the attention block is compiled; the rest of ViT (patch-embedding
convolution, layernorm, GELU) is future scope for Tensora (see
docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md Section 9) --
this demo does not claim to run the whole model.

Needs `transformers`, `safetensors`, and `pillow` in addition to
tools/models/requirements.txt (PyTorch download of a ~350MB checkpoint,
one time, requires internet access):
    pip install transformers safetensors pillow
    python3 tools/models/prepare_vit_attention_demo.py
"""
import json
import os
import urllib.request

import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit("error: PyTorch is required. Install with: pip install -r tools/models/requirements.txt")

try:
    from transformers import ViTImageProcessor, ViTModel
except ImportError:
    raise SystemExit("error: transformers is required. Install with: pip install transformers safetensors pillow")

try:
    from PIL import Image
except ImportError:
    raise SystemExit("error: pillow is required. Install with: pip install pillow")

MODEL_ID = "google/vit-base-patch16-224-in21k"
HEADS = 12
DK = 64          # hidden_size / num_attention_heads = 768 / 12
D = 768
NUM_LAYERS = 12
# Both knobs below can be overridden without editing this file, e.g.
#   make prep-vit-attention IMAGE=my-cat.jpg
#   make prep-vit-attention LAYER=8
# so a user can ask "what if I try a different layer/photo?" without
# reading the source. LAYER defaults to the last layer (11): later
# layers give more subject-focused, legible attention patterns than
# layer 0 -- see the module docstring.
LAYER = int(os.environ.get("TENSORA_VIT_LAYER", "11"))
# A single foreground subject (a fluffy white dog) against a simpler,
# less textured background than the earlier snowy-forest photo -- also
# makes the resulting heatmap easier to read. TENSORA_VIT_IMAGE may be
# an http(s) URL (downloaded) or a local file path (used as-is).
IMAGE_URL = os.environ.get(
    "TENSORA_VIT_IMAGE",
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "vit_attn")


def main():
    if not (0 <= LAYER < NUM_LAYERS):
        raise SystemExit(f"TENSORA_VIT_LAYER must be 0..{NUM_LAYERS - 1}, got {LAYER}")
    torch.manual_seed(0)
    img_path = os.path.join(OUT_DIR, "input_image.jpg")
    os.makedirs(OUT_DIR, exist_ok=True)
    # Re-download only when the requested source actually changed (a
    # sidecar file records which URL/path last produced input_image.jpg)
    # -- otherwise switching TENSORA_VIT_IMAGE would silently keep using
    # a stale cached photo from a previous run.
    source_marker = os.path.join(OUT_DIR, ".image_source")
    last_source = open(source_marker).read().strip() if os.path.exists(source_marker) else None
    if IMAGE_URL.startswith("http://") or IMAGE_URL.startswith("https://"):
        if not os.path.exists(img_path) or last_source != IMAGE_URL:
            print(f"Downloading a real sample photograph from {IMAGE_URL} ...")
            urllib.request.urlretrieve(IMAGE_URL, img_path)
            open(source_marker, "w").write(IMAGE_URL)
    else:
        print(f"Using local photograph: {IMAGE_URL}")
        img_path = IMAGE_URL
        open(source_marker, "w").write(IMAGE_URL)

    print(f"Loading real pretrained weights: {MODEL_ID} (one-time download, ~350MB) ...")
    model = ViTModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    model.eval()
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Capture the real input to, and real output of, EVERY encoder
    # layer's self-attention block via forward hooks -- not a hand
    # re-derivation, the model's own computation. Layer LAYER's capture
    # feeds the single-layer demo/visualization above; all 12 layers'
    # captures feed the independent per-layer verification below.
    captured = {}
    per_layer_io = [None] * NUM_LAYERS
    all_layer_weights = [None] * NUM_LAYERS

    def make_hook(layer_idx):
        def hook(_module, args, output):
            layer_out = output[0] if isinstance(output, tuple) else output
            per_layer_io[layer_idx] = (args[0].detach(), layer_out.detach())
            if layer_idx == LAYER:
                captured["in"] = args[0].detach()
                captured["out"] = layer_out.detach()
            # ViTAttention.forward returns (attn_output, attn_weights); the
            # "eager" implementation above is what actually populates
            # attn_weights (some fused/SDPA backends return None here).
            if isinstance(output, tuple) and output[1] is not None:
                all_layer_weights[layer_idx] = output[1].detach()
        return hook

    handles = [model.layers[i].attention.register_forward_hook(make_hook(i)) for i in range(NUM_LAYERS)]
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()

    X = captured["in"].numpy().astype(np.float32)          # [1, seq, 768]
    expected = captured["out"].numpy().astype(np.float32)  # [1, seq, 768]
    attn_weights = all_layer_weights[LAYER].numpy().astype(np.float32)   # [1, 12, seq, seq]
    all_layers_attn = np.concatenate(
        [w.numpy().astype(np.float32) for w in all_layer_weights], axis=0)  # [12, 12, seq, seq]
    seq = X.shape[1]
    print(f"Captured real hidden state, real attention output, and real attention weights "
          f"for all {NUM_LAYERS} layers (layer {LAYER} recomputed by Tensora): seq_len={seq}")

    attn = model.layers[LAYER].attention

    def per_head_qkv(lin):
        # nn.Linear stores [out, in]; Tensora computes x @ W with
        # W: [in, out], so transpose, then split the 768-wide `out`
        # axis into (head=12, dk=64) -- the concat order PyTorch's
        # own o_proj expects on the way back in.
        W = lin.weight.detach().numpy().T.astype(np.float32)  # [768, 768] = [d, head*dk]
        b = lin.bias.detach().numpy().astype(np.float32)       # [768]
        W_h = W.reshape(D, HEADS, DK).transpose(1, 0, 2)        # [head, d, dk]
        b_h = b.reshape(HEADS, DK)                              # [head, dk]
        return W_h, b_h

    def per_head_o(lin):
        # o_proj: [out=768, in=768]; Tensora needs it as per-head
        # slabs of the *input* axis (see the .tns file's header: this
        # is the block-matmul identity for "concat heads, then
        # project").
        W = lin.weight.detach().numpy().T.astype(np.float32)  # [in=768, out=768]
        b = lin.bias.detach().numpy().astype(np.float32)       # [768]
        W_h = W.reshape(HEADS, DK, D)                           # [head, dk, d]
        return W_h, b

    Wq, bq = per_head_qkv(attn.q_proj)
    Wk, bk = per_head_qkv(attn.k_proj)
    Wv, bv = per_head_qkv(attn.v_proj)
    Wo, bo = per_head_o(attn.o_proj)

    d = os.path.join(OUT_DIR, "eval")
    os.makedirs(d, exist_ok=True)

    x_h1 = X[:, None, :, :]  # add the head=1 broadcast axis: [1, 1, seq, 768]
    np.save(os.path.join(d, "Xq.npy"), x_h1)
    np.save(os.path.join(d, "Xk.npy"), x_h1)
    np.save(os.path.join(d, "Wq.npy"), Wq.reshape(1, HEADS, D, DK))
    np.save(os.path.join(d, "bq.npy"), bq.reshape(HEADS, 1, DK))
    np.save(os.path.join(d, "Wk.npy"), Wk.reshape(1, HEADS, D, DK))
    np.save(os.path.join(d, "bk.npy"), bk.reshape(HEADS, 1, DK))
    np.save(os.path.join(d, "Wv.npy"), Wv.reshape(1, HEADS, D, DK))
    np.save(os.path.join(d, "bv.npy"), bv.reshape(HEADS, 1, DK))
    np.save(os.path.join(d, "Wo.npy"), Wo.reshape(1, HEADS, DK, D))
    np.save(os.path.join(d, "bo.npy"), bo)
    np.save(os.path.join(d, "expected.npy"), expected)  # [1, seq, 768], matches Tensora's [b=1, qseq, d] output
    np.save(os.path.join(d, "expected_attn_weights.npy"), attn_weights)  # [1, 12, seq, seq], real softmax weights
    np.save(os.path.join(d, "all_layers_attn_weights.npy"), all_layers_attn)  # [12, 12, seq, seq], real, all layers

    meta = {"model": MODEL_ID, "layer": LAYER, "num_layers": NUM_LAYERS, "seq_len": int(seq),
            "hidden_size": D, "heads": HEADS}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    print(f"Wrote real ViT attention-block artifacts to {os.path.normpath(OUT_DIR)}")

    # --- Independent per-layer export: every one of the 12 encoder
    # layers gets its own real input hidden state and real, pretrained
    # Q/K/V/output-projection weights, so each layer can be recomputed
    # and checked by Tensora independently (not just layer LAYER).
    all_layers_dir = os.path.join(OUT_DIR, "eval_all_layers")
    for i in range(NUM_LAYERS):
        Xi, Yi = per_layer_io[i]
        Xi = Xi.numpy().astype(np.float32)  # [1, seq, 768]
        Yi = Yi.numpy().astype(np.float32)  # [1, seq, 768]
        attn_i = model.layers[i].attention
        Wq_i, bq_i = per_head_qkv(attn_i.q_proj)
        Wk_i, bk_i = per_head_qkv(attn_i.k_proj)
        Wv_i, bv_i = per_head_qkv(attn_i.v_proj)
        Wo_i, bo_i = per_head_o(attn_i.o_proj)

        li = os.path.join(all_layers_dir, f"layer_{i:02d}")
        os.makedirs(li, exist_ok=True)
        xi_h1 = Xi[:, None, :, :]
        np.save(os.path.join(li, "Xq.npy"), xi_h1)
        np.save(os.path.join(li, "Xk.npy"), xi_h1)
        np.save(os.path.join(li, "Wq.npy"), Wq_i.reshape(1, HEADS, D, DK))
        np.save(os.path.join(li, "bq.npy"), bq_i.reshape(HEADS, 1, DK))
        np.save(os.path.join(li, "Wk.npy"), Wk_i.reshape(1, HEADS, D, DK))
        np.save(os.path.join(li, "bk.npy"), bk_i.reshape(HEADS, 1, DK))
        np.save(os.path.join(li, "Wv.npy"), Wv_i.reshape(1, HEADS, D, DK))
        np.save(os.path.join(li, "bv.npy"), bv_i.reshape(HEADS, 1, DK))
        np.save(os.path.join(li, "Wo.npy"), Wo_i.reshape(1, HEADS, DK, D))
        np.save(os.path.join(li, "bo.npy"), bo_i)
        np.save(os.path.join(li, "expected.npy"), Yi)
        json.dump({"model": MODEL_ID, "layer": i, "num_layers": NUM_LAYERS,
                   "seq_len": int(seq), "hidden_size": D, "heads": HEADS},
                  open(os.path.join(li, "meta.json"), "w"), indent=2)
    print(f"Wrote independent per-layer artifacts for all {NUM_LAYERS} layers to "
          f"{os.path.normpath(all_layers_dir)}")


if __name__ == "__main__":
    main()
