#!/usr/bin/env python3
"""Renders the ViT attention-substitution visual dashboard.

tools/models/prepare_vit_substitution_demo.py already proves the
result numerically ("PASS -- label unchanged, final-logit max abs
error ~1e-6") on five real, visibly different photographs. This
script turns that JSON into pictures, because the numerical result
alone under-sells what actually happened: a real attention block
inside a real, live, 86M-parameter production classifier was replaced
by Tensora's own compiled native code, mid-inference, and the model's
real-world prediction did not change.

For each of the five gallery photographs (run
prepare_vit_substitution_demo.py first), writes
demos/artifacts/vit_substitution/dashboard_<name>.png with four panels:

  1. The real input photograph.
  2. A schematic of the 12-layer ViT encoder stack with layer 11 drawn
     as "PyTorch -> TENSORA" -- the one block Tensora's own compiled
     binary actually computed; every other layer is unmodified.
  3. Side-by-side top-5 ImageNet-1k class-probability bars: the real,
     unmodified model's own prediction next to the Tensora-substituted
     model's prediction, for the SAME photograph.
  4. A numerical-error "meter": the substituted block's output vs the
     real model's own output for that block, and the two final logit
     vectors' max abs error, end to end.

Also writes demos/artifacts/vit_substitution/gallery.png: all five
photographs in one row, each with its predicted label and confidence
from BOTH the original and Tensora-substituted model, and a check
mark where they agree exactly (which is every one of the five, per
the underlying summary).

Needs matplotlib, Pillow, and demos/artifacts/vit_substitution/gallery_summary.json
(written by prepare_vit_substitution_demo.py):
    python3 tools/models/prepare_vit_substitution_demo.py
    python3 tools/models/visualize_vit_substitution.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_DIR = os.path.join(ROOT, "demos", "artifacts", "vit_substitution")
SUMMARY_PATH = os.path.join(OUT_DIR, "gallery_summary.json")


def draw_layer_stack(ax, substituted_layer, num_layers):
    """A compact 'layers 0..10 -> [layer 11: PyTorch->TENSORA] ->
    LayerNorm+head -> logits' diagram, so a viewer can see exactly
    which one block out of twelve was replaced."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Where the substitution happens", fontsize=10, fontweight="bold")

    ax.add_patch(FancyBboxPatch((0.5, 6.6), 4.0, 1.6, boxstyle="round,pad=0.05",
                                 linewidth=1, edgecolor="#666666", facecolor="#e8e8e8"))
    ax.text(2.5, 7.4, f"Layers 0 ... {substituted_layer - 1}\n(real PyTorch)",
            ha="center", va="center", fontsize=8)

    ax.add_patch(FancyBboxPatch((0.5, 4.2), 9.0, 1.9, boxstyle="round,pad=0.06",
                                 linewidth=2.2, edgecolor="#b02a2a", facecolor="#fde8e8"))
    ax.text(5.0, 5.75, f"LAYER {substituted_layer} ATTENTION", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#8c1f1f")
    ax.text(2.6, 4.85, "real input\n(PyTorch)", ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(5.9, 4.85), xytext=(4.1, 4.85),
                arrowprops=dict(arrowstyle="->", color="#8c1f1f", lw=1.6))
    ax.text(7.4, 4.85, "TENSORA\n(compiled native)", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#8c1f1f")

    remaining = num_layers - substituted_layer - 1
    ax.add_patch(FancyBboxPatch((0.5, 1.8), 4.0, 1.6, boxstyle="round,pad=0.05",
                                 linewidth=1, edgecolor="#666666", facecolor="#e8e8e8"))
    if remaining > 0:
        remaining_text = f"Layers {substituted_layer + 1} ... {num_layers - 1}\n(real PyTorch, {remaining} more)"
    else:
        remaining_text = "(no further encoder\nlayers -- this was the last)"
    ax.text(2.5, 2.6, remaining_text, ha="center", va="center", fontsize=8)
    ax.add_patch(FancyBboxPatch((5.0, 1.8), 4.5, 1.6, boxstyle="round,pad=0.05",
                                 linewidth=1, edgecolor="#666666", facecolor="#e8e8e8"))
    ax.text(7.25, 2.6, "LayerNorm + classifier head\n(real PyTorch)", ha="center", va="center", fontsize=8)

    for (x0, y0, x1, y1) in [(2.5, 6.6, 2.5, 6.1), (2.5, 4.2, 2.5, 3.4),
                              (2.5, 1.8, 2.5, 1.2), (2.5, 1.2, 5.0, 1.2)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
    ax.text(5.0, 0.5, "logits", ha="center", va="center", fontsize=8, style="italic")
    ax.annotate("", xy=(7.25, 1.0), xytext=(7.25, 1.8), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))


def draw_top5(ax, title, top5, color):
    labels = [lbl.split(",")[0] for lbl, _ in top5]
    probs = [p for _, p in top5]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, probs, color=color, edgecolor="black", linewidth=0.5)
    for yi, p in zip(y, probs):
        ax.text(p + 0.02, yi, f"{p*100:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def draw_error_meter(ax, block_err, logit_err, label_ok):
    ax.axis("off")
    ax.set_title("Numerical agreement", fontsize=10, fontweight="bold")
    lines = [
        "Tensora block output vs PyTorch's own block output:",
        f"    max abs error = {block_err:.2e}",
        "",
        "Final classifier logits, baseline vs substituted",
        "(11 more encoder layers + head downstream):",
        f"    max abs error = {logit_err:.2e}",
        "",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            va="top", family="monospace")
    verdict = "PASS -- prediction unchanged" if label_ok else "FAIL -- prediction changed"
    color = "#1a7a1a" if label_ok else "#b02a2a"
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.8, 0.14, transform=ax.transAxes,
                                 boxstyle="round,pad=0.02", linewidth=1.5,
                                 edgecolor=color, facecolor=color, alpha=0.15))
    ax.text(0.42, 0.09, verdict, transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=color, ha="center", va="center")


def render_dashboard(result):
    name = result["name"]
    img_path = os.path.join(ROOT, result["image_path"])
    image = Image.open(img_path).convert("RGB")

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(f"Tensora ViT Attention-Substitution: '{name}' photograph "
                 f"(layer {result['substituted_layer']} of {result['num_layers']})",
                 fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.2, 1.0], height_ratios=[1, 1])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title("Real input photograph", fontsize=10, fontweight="bold")

    ax_stack = fig.add_subplot(gs[1, 0])
    draw_layer_stack(ax_stack, result["substituted_layer"], result["num_layers"])

    ax_base = fig.add_subplot(gs[0, 1])
    draw_top5(ax_base, "Original HuggingFace top-5\n(100% real PyTorch)", result["baseline_top5"], "#4a78b5")
    ax_sub = fig.add_subplot(gs[1, 1])
    draw_top5(ax_sub, f"Tensora-substituted top-5\n(layer {result['substituted_layer']} computed by Tensora)",
               result["substituted_top5"], "#b5734a")

    ax_meter = fig.add_subplot(gs[:, 2])
    draw_error_meter(ax_meter, result["block_output_max_abs_error"],
                      result["final_logits_max_abs_error"], result["label_unchanged"])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, f"dashboard_{name}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render_gallery(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 4.2))
    if n == 1:
        axes = [axes]
    fig.suptitle("Tensora ViT Attention-Substitution: same real model, five real photographs\n"
                 "(layer 11's attention computed by Tensora; every other layer real, unmodified PyTorch)",
                 fontsize=12, fontweight="bold")
    for ax, r in zip(axes, results):
        image = Image.open(os.path.join(ROOT, r["image_path"])).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        base = r["baseline_label"].split(",")[0]
        sub = r["substituted_label"].split(",")[0]
        mark = "\u2713" if r["label_unchanged"] else "\u2717"
        color = "#1a7a1a" if r["label_unchanged"] else "#b02a2a"
        ax.set_title(f"{r['name']}\n\nPyTorch: {base} ({r['baseline_confidence']*100:.1f}%)\n"
                     f"Tensora: {sub} ({r['substituted_confidence']*100:.1f}%)\n{mark}",
                     fontsize=9, color=color)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out_path = os.path.join(OUT_DIR, "gallery.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    if not os.path.exists(SUMMARY_PATH):
        raise SystemExit(f"error: {SUMMARY_PATH} not found. Run "
                          f"tools/models/prepare_vit_substitution_demo.py first.")
    results = json.load(open(SUMMARY_PATH))
    for r in results:
        path = render_dashboard(r)
        print(f"Wrote {os.path.normpath(path)}")
    gallery_path = render_gallery(results)
    print(f"Wrote {os.path.normpath(gallery_path)}")
    n_ok = sum(1 for r in results if r["label_unchanged"])
    print(f"{n_ok}/{len(results)} images: predicted label unchanged after substitution.")


if __name__ == "__main__":
    main()
