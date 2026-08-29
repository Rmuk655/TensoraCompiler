#!/usr/bin/env python3
"""Attention-heatmap visualization: inspecting learned ViT attention.

Runs examples/vit_attention_weights.tns through the real `tensora`
CLI (not a numpy re-derivation) to get the FULL [197, 197] attention
matrix (every query token, not only [CLS]) for all 12 heads, using the
SAME real hidden state and SAME real pretrained Q/K weights already
extracted by prepare_vit_attention_demo.py. Checks Tensora's output
against the real model's own attention weights (captured directly from
PyTorch via a forward hook, see that script), then renders three views:

  attention_heatmap.png   -- one panel per head (12 panels), [CLS]'s
                              attention over the 196 image patches: the
                              whole point of multi-head attention is
                              that different heads attend to different
                              parts of the same input, which no single
                              panel can show by itself. Every panel
                              here is Tensora-verified.
  attention_rollout.png    -- two more legible summaries: [CLS]'s
                              mean-over-heads attention at the
                              Tensora-verified layer, and a full
                              attention-rollout trace across all 12
                              encoder layers (Abnar & Zuidema, 2020).
                              Rollout needs every layer's attention,
                              and Tensora compiles only one block, so
                              it is computed directly from the real
                              model's own weights, NOT from Tensora --
                              both panels are labeled and bordered to
                              make that boundary visually explicit, not
                              just stated in a caption.
  attention_microscope.png -- attention is a full [197, 197] matrix,
                              not just what [CLS] does; this figure
                              lets you inspect any single *patch*
                              token's row ("what does this patch pay
                              attention to?") and column ("which other
                              patches pay attention back to this one?")
                              in the SAME Tensora-verified layer used
                              above -- both panels are Tensora-verified,
                              since the full attention matrix (not only
                              its [CLS] row) is checked against the real
                              model. By default this inspects THREE
                              patches at once -- one on the dog's face,
                              one on its body, one on the background --
                              so the figure is a small experiment ("do
                              face/body/background patches behave
                              differently?"), not just one cherry-picked
                              example; pass "row,col" (0-13 each) to
                              inspect one specific patch instead.
  attention_asymmetry.png  -- attention is directional: patch A
                              attending to patch B is a different
                              number from patch B attending to patch A.
                              This figure finds the most asymmetric
                              patch pairs in the SAME Tensora-verified
                              layer and draws them as arrows (thicker
                              in the stronger direction), plus a table
                              of the underlying weights.
  attention_diagnostics.png -- two quantitative, non-visual-cherry-
                              picking checks: (a) per-head attention
                              entropy (a head that spreads attention
                              over many patches has high entropy; one
                              that focuses on a few has low entropy --
                              both are Tensora-verified numbers, not
                              qualitative "it looks focused"), and (b)
                              a real-model vs Tensora attention error
                              map at a dramatically enlarged scale, so
                              the "these match" claim is a picture, not
                              only the printed max-error number.

This turns the numerical-fidelity demo (demo-vit-attention-one-layer) into a
visual one: it shows, using Tensora's own compiled computation, which
regions of a real photograph a real pretrained Transformer attends to.

Usage (after tools/models/prepare_vit_attention_demo.py has been run
once):
    python3 tools/models/visualize_vit_attention.py [build_dir] [row,col]

Needs matplotlib and pillow in addition to numpy (see
tools/models/requirements.txt); does not need PyTorch/transformers at
this stage -- all real data was already exported to .npy by the prep
script.
"""
import json
import os
import subprocess
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("error: matplotlib is required. Install with: pip install matplotlib")

try:
    from PIL import Image
except ImportError:
    raise SystemExit("error: pillow is required. Install with: pip install pillow")

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ART = os.path.join(ROOT, "demos", "artifacts", "vit_attn")
EVAL = os.path.join(ART, "eval")
PATCH_GRID = 14   # 224 / 16 = 14 patches per side (196 patches + 1 [CLS] = 197)
TOL = 1e-4

VERIFIED_COLOR = "#1a7f37"    # green: Tensora's own compiled output, checked against the real model
REFERENCE_COLOR = "#888888"   # gray: real-model-only data Tensora did not (and, for whole-model
                               # chains, currently cannot) compute


def frame(ax, verified, note="", clear_ticks=True):
    """Border + label an axes panel so a viewer can tell, at a glance and
    without reading the caption, whether a panel's data came from
    Tensora's own verified computation or is reference-only context.
    clear_ticks=False keeps an axes' existing tick labels (e.g. a bar
    chart's head-index labels) instead of blanking them."""
    if clear_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    color = VERIFIED_COLOR if verified else REFERENCE_COLOR
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(3.5)
        spine.set_linestyle("solid" if verified else (0, (5, 3)))
    label = ("\u2713 TENSORA-VERIFIED" + (f" ({note})" if note else "")) if verified else \
            ("reference only -- PyTorch, not Tensora" + (f" ({note})" if note else ""))
    ax.set_xlabel(label, color=color, fontsize=9,
                  fontweight="bold" if verified else "normal")


def overlay(ax, image, heatmap):
    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=0.5,
              extent=(0, image.width, image.height, 0), interpolation="bilinear")


def main():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else "build"
    patch_arg = sys.argv[2] if len(sys.argv) > 2 else None
    tensora = os.path.join(ROOT, build_dir, "tensora")
    img_path = os.path.join(ART, "input_image.jpg")
    exp_path = os.path.join(EVAL, "expected_attn_weights.npy")

    if not (os.path.exists(EVAL) and os.path.exists(exp_path)):
        print(f"error: {EVAL} (or expected_attn_weights.npy) not found.")
        print("Run first: python3 tools/models/prepare_vit_attention_demo.py")
        sys.exit(1)

    out_path = os.path.join(EVAL, "tensora_attn_weights.npy")
    print("Running examples/vit_attention_weights.tns through the real tensora CLI ...")
    subprocess.run(
        [tensora, "--run", "examples/vit_attention_weights.tns",
         "--data-dir", EVAL, "--output", out_path],
        cwd=ROOT, check=True,
    )

    got = np.load(out_path)              # [1, 12, 197, 197] -- Tensora's own computation, EVERY query
    expected = np.load(exp_path)         # [1, 12, 197, 197] -- real ViT's own attention weights
    max_err = float(np.max(np.abs(got - expected)))
    print(f"Tensora vs real ViT attention weights: max abs error {max_err:.2e} (tolerance {TOL:g})")
    if max_err > TOL:
        print("FAIL: Tensora's attention weights do not match the real model closely enough")
        sys.exit(1)
    print("PASS: Tensora's own compiled attention weights match the real pretrained model "
          "-- for the FULL [197,197] matrix, not just the [CLS] row")

    meta = json.load(open(os.path.join(EVAL, "meta.json")))

    # Per-head entropy of [CLS]'s attention (low = focused on a few
    # patches, high = spread over many) -- computed once, straight from
    # Tensora's own verified output, and reused both by the diagnostics
    # figure below and to pick three representative heads (most
    # focused / middle / most diffuse) for the per-head microscope
    # panels, so the qualitative heatmaps and the quantitative entropy
    # numbers are talking about the same heads.
    eps = 1e-12
    entropies = []
    for head in range(meta["heads"]):
        p = got[0, head, 0, :]
        p = p / p.sum()
        entropies.append(float(-np.sum(p * np.log(p + eps))))
    order = sorted(range(meta["heads"]), key=lambda h: entropies[h])
    head_lo = order[0]                     # most focused (lowest entropy)
    head_hi = order[-1]                    # most diffuse (highest entropy)
    head_mid = order[len(order) // 2]      # a middle-of-the-pack head
    print(f"Head entropy range: head {head_lo} lowest ({entropies[head_lo]:.2f} nats, most "
          f"focused), head {head_mid} middle ({entropies[head_mid]:.2f}), head {head_hi} "
          f"highest ({entropies[head_hi]:.2f} nats, most diffuse)")

    # [CLS] token is sequence position 0; positions 1..196 are the 14x14
    # real image patches, in row-major raster order (ViT's own patch
    # embedding order).
    image = Image.open(img_path).convert("RGB")
    mean_heads = got[0].mean(axis=0)  # Tensora's own layer, averaged over heads: [seq, seq]

    # --- Figure 1: [CLS] attention, one panel per head (all Tensora-verified) ---
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Real input photograph")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for head in range(meta["heads"]):
        ax = axes[head + 1]
        cls_attn = got[0, head, 0, 1:]                      # [196]
        heatmap = cls_attn.reshape(PATCH_GRID, PATCH_GRID)   # [14, 14]
        overlay(ax, image, heatmap)
        ax.set_title(f"head {head}")
        frame(ax, verified=True)

    for ax in axes[meta["heads"] + 1:]:
        ax.axis("off")

    fig.suptitle(f"[CLS] token attention, all {meta['heads']} heads (layer {meta['layer']})\n"
                 f"{meta['model']} (Tensora vs real model max error {max_err:.1e})")

    out_png = os.path.join(ART, "attention_heatmap.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Wrote attention heatmap to {os.path.normpath(out_png)}")

    # --- Figure 2: mean-over-heads (Tensora-verified) vs multi-layer rollout (reference only) ---
    # Rollout needs every layer's attention, and Tensora compiles only one
    # block -- it is computed directly from the real model's own weights
    # at every layer (all_layers_attn_weights.npy), NOT from Tensora. The
    # green solid border above marks what Tensora itself computed and
    # this script independently checked; the gray dashed border marks
    # real-model-only reference context, so the boundary between "Tensora
    # proved this" and "the real model does this" is visible, not just
    # stated in a caption.
    all_layers_path = os.path.join(EVAL, "all_layers_attn_weights.npy")
    if not os.path.exists(all_layers_path):
        print(f"note: {all_layers_path} not found (re-run prepare_vit_attention_demo.py "
              "to get it); skipping rollout/microscope panels")
        return
    all_layers = np.load(all_layers_path)  # [num_layers, heads, seq, seq], real model, all layers

    mean_heatmap = mean_heads[0, 1:].reshape(PATCH_GRID, PATCH_GRID)

    # Attention rollout (Abnar & Zuidema, 2020): trace how attention
    # composes across depth by chaining each layer's head-averaged
    # attention (with an added identity term for the residual
    # connection every Transformer block has) through matrix
    # multiplication, then read off the [CLS] row.
    seq = all_layers.shape[-1]
    identity = np.eye(seq, dtype=np.float32)
    rollout = None
    for layer_attn in all_layers:                    # layer_attn: [heads, seq, seq]
        layer_mean = layer_attn.mean(axis=0)          # [seq, seq]
        layer_hat = 0.5 * layer_mean + 0.5 * identity
        rollout = layer_hat if rollout is None else layer_hat @ rollout
    rollout_heatmap = rollout[0, 1:].reshape(PATCH_GRID, PATCH_GRID)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6))
    axes2[0].imshow(image)
    axes2[0].set_title("Real input photograph")
    axes2[0].set_xticks([]); axes2[0].set_yticks([])
    for spine in axes2[0].spines.values():
        spine.set_visible(False)

    overlay(axes2[1], image, mean_heatmap)
    axes2[1].set_title(f"Mean over {meta['heads']} heads\n(layer {meta['layer']})")
    frame(axes2[1], verified=True, note=f"max err {max_err:.1e}")

    overlay(axes2[2], image, rollout_heatmap)
    axes2[2].set_title(f"Attention rollout\n(all {meta['num_layers']} layers)")
    frame(axes2[2], verified=False, note="needs LayerNorm/GELU")

    fig2.suptitle(f"{meta['model']}: inspecting learned ViT attention\n"
                  f"(where does [CLS] attend?)")
    out_png2 = os.path.join(ART, "attention_rollout.png")
    fig2.tight_layout()
    fig2.savefig(out_png2, dpi=150)
    print(f"Wrote mean-head/rollout visualization to {os.path.normpath(out_png2)}")

    # --- Figure 3: attention microscope -- any patch, not just [CLS] ---
    # Attention is the full [197, 197] matrix Tensora just proved it
    # reproduces; [CLS] is only row 0 of it. Pick any patch token's row
    # ("what does this patch attend to?") and column ("what attends back
    # to this patch?") -- both are Tensora-verified, exactly like the
    # [CLS] row above, since the whole matrix was checked, not only row 0.
    #
    # Mean-over-heads alone can hide structure Figure 1 already proved
    # exists (different heads attend to different things): three
    # per-head "ATTENDS TO" columns are shown alongside the mean, using
    # the SAME three heads (most focused / middle / most diffuse by
    # [CLS] entropy) picked above, so the qualitative picture and the
    # quantitative entropy numbers refer to the same heads.
    #
    # With no CLI arg, inspect THREE patches at once -- one on the dog's
    # face, one on its body, one on the background -- picked once, ahead
    # of time, from the actual patch-grid colors (not cherry-picked after
    # seeing the results): this turns the figure into a small experiment
    # ("do face / body / background patches behave differently?") rather
    # than one arbitrary example.
    show_heads = [head_lo, head_mid, head_hi]

    def microscope_panel(row_col_list):
        n = len(row_col_list)
        ncols = 2 + len(show_heads) + 1  # marker, mean-to, per-head-to..., mean-back
        fig3, axes3 = plt.subplots(n, ncols, figsize=(4.6 * ncols, 5.0 * n), squeeze=False)
        for i, (row, col, label) in enumerate(row_col_list):
            query_idx = 1 + row * PATCH_GRID + col  # +1: position 0 is [CLS]
            attends_to_mean = mean_heads[query_idx, 1:].reshape(PATCH_GRID, PATCH_GRID)
            attended_by_mean = mean_heads[1:, query_idx].reshape(PATCH_GRID, PATCH_GRID)

            ax0 = axes3[i][0]
            ax0.imshow(image)
            ax0.set_title(f"{label}\npatch (row={row}, col={col}) marked")
            ax0.set_xticks([]); ax0.set_yticks([])
            for spine in ax0.spines.values():
                spine.set_visible(False)
            px = col / PATCH_GRID * image.width
            py = row / PATCH_GRID * image.height
            pw, ph = image.width / PATCH_GRID, image.height / PATCH_GRID
            ax0.add_patch(mpatches.Rectangle((px, py), pw, ph, fill=False,
                                              edgecolor="white", linewidth=2.5))
            ax0.add_patch(mpatches.Rectangle((px, py), pw, ph, fill=False,
                                              edgecolor="black", linewidth=1))

            overlay(axes3[i][1], image, attends_to_mean)
            axes3[i][1].set_title("ATTENDS TO ...\n(mean over heads)")
            frame(axes3[i][1], verified=True, note=f"layer {meta['layer']}")

            for j, head in enumerate(show_heads):
                head_map = got[0, head, query_idx, 1:].reshape(PATCH_GRID, PATCH_GRID)
                ax = axes3[i][2 + j]
                overlay(ax, image, head_map)
                role = "most focused" if head == head_lo else \
                       "most diffuse" if head == head_hi else "middle"
                ax.set_title(f"ATTENDS TO ...\nhead {head} ({role})")
                frame(ax, verified=True, note=f"layer {meta['layer']}")

            overlay(axes3[i][-1], image, attended_by_mean)
            axes3[i][-1].set_title("... ATTENDED BACK\n(mean over heads)")
            frame(axes3[i][-1], verified=True, note=f"layer {meta['layer']}")
        return fig3

    if patch_arg:
        row_s, col_s = patch_arg.split(",")
        row, col = int(row_s), int(col_s)
        fig3 = microscope_panel([(row, col, "Real input photograph")])
        fig3.suptitle(f"Attention microscope: one patch's full row & column of the "
                      f"real [197x197] matrix\n{meta['model']} "
                      f"(pass a different \"row,col\" to inspect another patch)")
        picked_desc = f"(patch row={row}, col={col})"
    else:
        # Picked from the actual patch-grid colors ahead of time: white/
        # bright patches on the dog (face near the top of the dog cluster,
        # body in its center), green patches off the dog (background).
        # See tools/models/README or git history for how these were
        # chosen -- not adjusted after looking at the attention output.
        picks = [
            (2, 3, "Dog's FACE patch"),
            (8, 5, "Dog's BODY patch"),
            (6, 11, "BACKGROUND patch"),
        ]
        fig3 = microscope_panel(picks)
        fig3.suptitle(f"Attention microscope: face vs. body vs. background\n"
                      f"{meta['model']} (pass \"row,col\" to inspect one specific patch instead)")
        picked_desc = "(face/body/background; pass row,col for one specific patch)"

    out_png3 = os.path.join(ART, "attention_microscope.png")
    fig3.tight_layout()
    fig3.savefig(out_png3, dpi=110)
    print(f"Wrote attention microscope to {os.path.normpath(out_png3)} {picked_desc}")

    # --- Figure 4: attention asymmetry -- attention is directional ---
    # Attention(A -> B) != Attention(B -> A) in general. Find the patch
    # pairs (excluding [CLS]) where the two directions differ the most
    # in the SAME Tensora-verified mean-over-heads matrix, and draw them
    # as arrows: a thick solid arrow for the stronger direction, a thin
    # dashed one for the weaker. Only the top 3 pairs are shown (not 8):
    # more than that turns into a hard-to-read tangle of crossing lines,
    # and each pair is labeled with its own two numbers directly on the
    # image, so a viewer does not have to infer direction from arrow
    # style alone.
    n_patches = PATCH_GRID * PATCH_GRID
    patch_attn = mean_heads[1:, 1:]  # [196, 196], patch-to-patch only
    asym = patch_attn - patch_attn.T
    iu = np.triu_indices(n_patches, k=1)
    diffs = asym[iu]
    top_k = 3
    top_idx = np.argsort(-np.abs(diffs))[:top_k]
    pair_colors = ["#ffeb3b", "#00e5ff", "#ff4dff"]  # yellow, cyan, magenta -- one per pair

    fig4, ax4 = plt.subplots(figsize=(9, 9))
    ax4.imshow(image)
    ax4.set_xticks([]); ax4.set_yticks([])

    def patch_center(idx):
        r, c = divmod(idx, PATCH_GRID)
        return ((c + 0.5) / PATCH_GRID * image.width,
                (r + 0.5) / PATCH_GRID * image.height)

    table_rows = []
    for rank, k in enumerate(top_idx):
        i, j = iu[0][k], iu[1][k]
        a_ij = patch_attn[i, j]  # i -> j
        a_ji = patch_attn[j, i]  # j -> i
        xi, yi = patch_center(i)
        xj, yj = patch_center(j)
        color = pair_colors[rank % len(pair_colors)]
        if a_ij >= a_ji:
            strong, weak = (xi, yi), (xj, yj)
            strong_val, weak_val = a_ij, a_ji
        else:
            strong, weak = (xj, yj), (xi, yi)
            strong_val, weak_val = a_ji, a_ij
        ax4.annotate("", xy=strong, xytext=weak,
                     arrowprops=dict(arrowstyle="-|>", color=color, linewidth=3.0, alpha=0.95))
        ax4.annotate("", xy=weak, xytext=strong,
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     linewidth=0.8, linestyle=(0, (2, 2)), alpha=0.7))
        mx, my = (strong[0] + weak[0]) / 2, (strong[1] + weak[1]) / 2
        ax4.text(mx, my, f"#{rank + 1}: {strong_val:.3f} \u2192\n\u2190 {weak_val:.3f}",
                 color="black", fontsize=9, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=color, alpha=0.85, edgecolor="none"))
        table_rows.append(((int(i // PATCH_GRID), int(i % PATCH_GRID)),
                            (int(j // PATCH_GRID), int(j % PATCH_GRID)),
                            float(a_ij), float(a_ji)))

    frame(ax4, verified=True, note=f"mean over heads, top-{top_k} most directional patch pairs")
    ax4.set_title(f"Attention is directional: patch A attending to patch B\n"
                  f"is a different number from B attending to A ({meta['model']}, layer {meta['layer']})")

    out_png4 = os.path.join(ART, "attention_asymmetry.png")
    fig4.tight_layout()
    fig4.savefig(out_png4, dpi=150)
    print(f"Wrote attention asymmetry figure to {os.path.normpath(out_png4)}")
    print(f"Top {top_k} asymmetric patch pairs (patch A row,col -> patch B row,col : A->B, B->A):")
    for a, b, a_to_b, b_to_a in table_rows:
        print(f"  {a} -> {b} : {a_to_b:.4f}  |  {b} -> {a} : {b_to_a:.4f}")

    # --- Figure 5: quantitative diagnostics -- entropy + error map ---
    # (a) Per-head attention entropy (computed earlier, above): how
    #     spread out (high entropy) or focused (low entropy) each
    #     head's [CLS] attention is, straight from Tensora's own
    #     verified output.
    # (b) Real-model vs Tensora attention error, shown as a picture at a
    #     dramatically enlarged scale rather than only a printed number
    #     -- "here is what the real model computes, here is what Tensora
    #     computes, here is every place they differ".
    expected_mean = expected[0].mean(axis=0)  # real model, mean over heads: [197, 197]
    got_mean = got[0].mean(axis=0)            # Tensora, mean over heads: [197, 197]
    diff = np.abs(got_mean - expected_mean)
    amplify = 1.0 / max(diff.max(), 1e-12)

    fig5, axes5 = plt.subplots(1, 4, figsize=(20, 5))

    bars = axes5[0].barh(range(meta["heads"]), entropies, color="#1a7f37")
    axes5[0].set_yticks(range(meta["heads"]))
    axes5[0].set_yticklabels([f"head {h}" for h in range(meta["heads"])])
    axes5[0].invert_yaxis()
    for h, bar in enumerate(bars):
        axes5[0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                       f"{entropies[h]:.2f}", va="center", fontsize=8)
    axes5[0].set_xlim(0, max(entropies) * 1.15)
    axes5[0].set_title("Per-head attention entropy\n"
                        f"(head {head_lo} most focused, head {head_hi} most diffuse)")
    # clear_ticks=False: keep the "head 0".."head 11" y-tick labels that
    # frame() would otherwise blank out along with the panel's axes.
    frame(axes5[0], verified=True,
          note="entropy in nats, low=focused/high=diffuse", clear_ticks=False)

    im1 = axes5[1].imshow(expected_mean, cmap="viridis")
    axes5[1].set_title("Real ViT attention\n(mean over heads)")
    axes5[1].set_xticks([]); axes5[1].set_yticks([])
    frame(axes5[1], verified=False, note="real model, PyTorch")

    im2 = axes5[2].imshow(got_mean, cmap="viridis", vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
    axes5[2].set_title("Tensora attention\n(mean over heads)")
    axes5[2].set_xticks([]); axes5[2].set_yticks([])
    frame(axes5[2], verified=True)

    im3 = axes5[3].imshow(diff * amplify, cmap="inferno", vmin=0, vmax=1)
    axes5[3].set_title(f"|difference| x {amplify:.0f}\n(max raw error {diff.max():.1e})")
    axes5[3].set_xticks([]); axes5[3].set_yticks([])
    frame(axes5[3], verified=True, note="enlarged error scale")

    fig5.suptitle(f"{meta['model']}: quantitative checks, not just a max-error number "
                  f"(layer {meta['layer']})")
    out_png5 = os.path.join(ART, "attention_diagnostics.png")
    fig5.tight_layout()
    fig5.savefig(out_png5, dpi=150)
    print(f"Wrote attention diagnostics to {os.path.normpath(out_png5)}")

    # --- Figure 6: top-K attended patches as labeled boxes (no halo) ---
    # A continuous, interpolated heatmap overlay (Figures 1-3) is
    # attractive but not a literal picture of what the matrix contains:
    # bilinear interpolation between 14x14 cells and a "jet" colormap
    # both add visual information the model didn't produce. This figure
    # instead draws a box around exactly the top-K patches by attention
    # weight -- the same numbers Figure 3 visualizes as a heatmap, shown
    # here with no interpolation and a printed value per box.
    TOP_K = 5

    def topk_boxes_panel(row, col, label):
        query_idx = 1 + row * PATCH_GRID + col
        row_weights = mean_heads[query_idx, 1:]  # Tensora's own, verified: [196]
        order = np.argsort(-row_weights)[:TOP_K]
        pw, ph = image.width / PATCH_GRID, image.height / PATCH_GRID

        fig6, ax6 = plt.subplots(1, 1, figsize=(7, 7))
        ax6.imshow(image)
        ax6.set_xticks([]); ax6.set_yticks([])

        qpx, qpy = col * pw, row * ph
        ax6.add_patch(mpatches.Rectangle((qpx, qpy), pw, ph, fill=False,
                                          edgecolor="lime", linewidth=3))
        ax6.text(qpx + pw / 2, qpy - 6, "QUERY", color="lime", fontsize=10,
                  fontweight="bold", ha="center", va="bottom")

        colors = plt.cm.autumn(np.linspace(0, 0.8, TOP_K))
        for rank, patch in enumerate(order):
            r, c = int(patch) // PATCH_GRID, int(patch) % PATCH_GRID
            weight = float(row_weights[patch])
            px, py = c * pw, r * ph
            ax6.add_patch(mpatches.Rectangle((px, py), pw, ph, fill=False,
                                              edgecolor=colors[rank], linewidth=2.5))
            ax6.text(px + pw / 2, py + ph + 10, f"#{rank + 1}\n{weight:.3f}",
                      color=colors[rank], fontsize=8, fontweight="bold",
                      ha="center", va="top",
                      bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55,
                                edgecolor="none"))

        ax6.set_title(f"Top-{TOP_K} attended patches for {label} (row={row}, col={col})\n"
                      f"exact patches by weight -- no interpolation, no colormap halo")
        frame(ax6, verified=True, note=f"layer {meta['layer']}, mean over heads")
        return fig6

    if patch_arg:
        row_s, col_s = patch_arg.split(",")
        fig6 = topk_boxes_panel(int(row_s), int(col_s), "the requested patch")
    else:
        fig6 = topk_boxes_panel(2, 3, "the dog's FACE patch")
    out_png6 = os.path.join(ART, "attention_topk_boxes.png")
    fig6.tight_layout()
    fig6.savefig(out_png6, dpi=150)
    print(f"Wrote top-K attended-patch boxes to {os.path.normpath(out_png6)}")


if __name__ == "__main__":
    main()
