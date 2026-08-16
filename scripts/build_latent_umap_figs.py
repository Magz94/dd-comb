#!/usr/bin/env python3
"""UMAP latent-space figures from cached test embeddings (kshot/backbone_study/emb_*.npz):
one multi-panel figure for the five original backbones, one for DINOv2 + ConvNeXt-V2-Tiny.
Each panel is a 2D UMAP of that backbone's frozen DeepWeeds test embeddings, colored by class.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "kshot" / "backbone_study"
PRETTY = {"resnet18": "ResNet-18", "vgg16": "VGG-16", "densenet121": "DenseNet-121",
          "efficientnet_b0": "EfficientNet-B0", "vit_b16": "ViT-B/16",
          "dinov2_vitb14": "DINOv2 ViT-B/14", "convnextv2_tiny": "ConvNeXt-V2-Tiny"}


def panel(ax, name):
    d = np.load(STUDY / f"emb_{name}.npz")
    emb, y = d["emb"].astype(np.float32), d["y_true"]
    z = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(emb)
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=6, alpha=0.7, linewidths=0)
    ax.set_title(PRETTY[name], fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def build(names, ncols, out, suptitle):
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    sc = None
    for i, n in enumerate(names):
        sc = panel(axes[i], n)
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cb.set_label("DeepWeeds class")
    # constrained_layout reserves space for the suptitle, avoiding title overlap
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out.with_suffix('.png')}")


out_dir = REPO / "docs" / "figs_latent"
out_dir.mkdir(parents=True, exist_ok=True)
build(["resnet18", "vgg16", "densenet121", "efficientnet_b0", "vit_b16"], 3,
      out_dir / "latent_umap_original5",
      "Latent spaces (UMAP) — five original backbones")
build(["dinov2_vitb14", "convnextv2_tiny"], 2,
      out_dir / "latent_umap_dino_cnx",
      "Latent spaces (UMAP) — DINOv2 vs ConvNeXt-V2-Tiny")
