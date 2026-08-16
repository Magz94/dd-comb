#!/usr/bin/env python3
"""Generate publication-quality robustness heatmaps for the thesis.

Metrics: Accuracy, Macro-F1, AUROC-H (entropy-based misclassification detection),
and vacuity-based AUROC-H for the evidential models. Each figure has a 5-shot and a
20-shot panel; rows = models, columns = the four datasets + a worst-case column.
Sensoy is shown with the MLP head only (linear removed).
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                     "axes.linewidth": 0.8, "savefig.dpi": 220})

ROOT = "kshot"
DATASETS = [("deepweeds_no_bg_mergeval", "DeepWeeds"),
            ("plant_seedlings_v2_split", "Plant Seedlings"),
            ("bloodmnist_224_imagefolder_mergeval", "BloodMNIST"),
            ("ssl_patches", "SSL-Patches")]

# model: (label, path-template, type, is_ours)
MODELS = [
    ("DINOv2", "dinov2_standard/{DS}", "main", False),
    ("ConvNeXt-V2", "convnextv2_standard/{DS}", "main", False),
    ("DINOv2 · MC-Dropout", "baselines/mc_dropout/dinov2_vitb14/{DS}", "main", False),
    ("DINOv2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}", "mlpsensoy", False),
    ("ConvNeXt-V2 · MC-Dropout", "baselines/mc_dropout/convnextv2_tiny/{DS}", "main", False),
    ("ConvNeXt-V2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}", "mlpsensoy", False),
    ("Vanilla Deep Ensemble", "vanilla_ens/{DS}", "main", False),
    ("DD-CoMB (ours)", "dino_cnx_dempster/{DS}", "dempster", True),
]
VAC_MODELS = [
    ("DINOv2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}", "mlpsensoy", False),
    ("ConvNeXt-V2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}", "mlpsensoy", False),
    ("DD-CoMB (ours)", "dino_cnx_dempster/{DS}", "dempster", True),
]

def root_json(mtype, dskey):
    if mtype == "mlpsensoy":
        return "results_calib" if dskey == "deepweeds_no_bg_mergeval" else "results_calib_fix_exp"
    return "results_calib_fix" if dskey in ("plant_seedlings_v2_split", "ssl_patches") else "results_calib"

def root_vac(mtype, dskey):
    if mtype == "mlpsensoy":
        return "results_calib" if dskey == "deepweeds_no_bg_mergeval" else "results_calib_fix_exp"
    return "results_calib_fix_exp" if dskey in ("plant_seedlings_v2_split", "ssl_patches") else "results_calib"

def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def auroc(sc, lab):
    order = np.argsort(sc, kind="mergesort"); s = sc[order]; l = lab[order].astype(float)
    N = len(s); ranks = np.empty(N); i = 0; rank = 1.0
    while i < N:
        j = i
        while j + 1 < N and s[j + 1] == s[i]: j += 1
        ranks[i:j + 1] = (rank + rank + (j - i)) / 2.0; rank += (j - i + 1); i = j + 1
    npos = l.sum(); nneg = (l == 0).sum()
    return (ranks[l == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg) if npos and nneg else np.nan

def val(metric, tmpl, mtype, dskey, s):
    if metric == "vacuity_auroc":
        base = os.path.join(ROOT, root_vac(mtype, dskey), tmpl.format(DS=dskey)); vs = []
        for r in range(1, 21):
            f = os.path.join(base, f"{s}_shot_run_{r}_logits.npz")
            if not os.path.isfile(f): continue
            z = np.load(f)
            if "test_alpha" not in z: continue
            al = z["test_alpha"]; y = z["test_y"]; C = al.shape[1]
            u = C / np.clip(al.sum(1), 1e-9, None); inc = (al.argmax(1) != y).astype(int)
            vs.append(auroc(u, inc))
        return np.nanmean(vs) if vs else np.nan
    base = os.path.join(ROOT, root_json(mtype, dskey), tmpl.format(DS=dskey)); vs = []
    for r in range(1, 21):
        f = os.path.join(base, f"{s}_shot_run_{r}.json")
        if not os.path.isfile(f): continue
        d = json.load(open(f))
        if metric == "acc": vs.append(d["test_accuracy"])
        elif metric == "f1":
            p, rc = d.get("test_precision"), d.get("test_recall")
            vs.append(2 * p * rc / (p + rc) if (p and rc and p + rc) else 0.0)
        elif metric == "auroc": vs.append(d.get("auroc_err_pred_entropy", np.nan))
    return np.nanmean(vs) if vs else np.nan

def matrix(models, metric, s):
    M = np.full((len(models), len(DATASETS) + 1), np.nan)
    for i, (lab, tmpl, mtype, ours) in enumerate(models):
        for j, (dskey, _) in enumerate(DATASETS):
            M[i, j] = val(metric, tmpl, mtype, dskey, s)
        M[i, -1] = np.nanmin(M[i, :-1])
    return M

def draw(ax, M, models, vmin, vmax, title):
    im = ax.imshow(M, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DATASETS) + 1))
    ax.set_xticklabels([d[1] for d in DATASETS] + ["Worst-case"], fontsize=8.5,
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[0] for m in models], fontsize=9)
    for k, (lab, _, _, ours) in enumerate(models):
        if ours: ax.get_yticklabels()[k].set_fontweight("bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.3, color="#111")
    ax.axvline(len(DATASETS) - 0.5, color="black", lw=2.4)          # separate worst-case col
    for k, (lab, _, _, ours) in enumerate(models):                  # box the 'ours' row
        if ours:
            ax.add_patch(Rectangle((-0.5, k - 0.5), M.shape[1], 1, fill=False, edgecolor="#1a5cff", lw=2.2))
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=11, pad=6)
    return im

def figure(models, metric, vmin, vmax, metric_label, fname, arrow):
    fig, axes = plt.subplots(1, 2, figsize=(3.1 + 0.9 * len(DATASETS), 0.62 * len(models) + 1.6))
    for ax, s in zip(axes, [5, 20]):
        im = draw(ax, matrix(models, metric, s), models, vmin, vmax, f"{s}-shot")
    axes[1].set_yticklabels([])
    fig.suptitle(f"{metric_label}  ({arrow})", fontsize=12.5, y=1.02, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(f"docs/{fname}", bbox_inches="tight")
    plt.close(fig); print("wrote docs/" + fname)

figure(MODELS, "acc", 0.40, 0.95, "Accuracy across domains", "fig_heatmap_accuracy.png", "higher better")
figure(MODELS, "f1",  0.30, 0.95, "Macro-F1 across domains", "fig_heatmap_macrof1.png", "higher better")
figure(MODELS, "auroc", 0.50, 0.95, "Misclassification detection (AUROC-H, entropy)", "fig_heatmap_auroc.png", "higher better")
figure(VAC_MODELS, "vacuity_auroc", 0.50, 0.95, "Vacuity-based detection (AUROC-H)", "fig_heatmap_vacuity_auroc.png", "higher better")
print("done")
