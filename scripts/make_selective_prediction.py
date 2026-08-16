#!/usr/bin/env python3
"""Selective-prediction figures for the thesis (Geifman & El-Yaniv framework).

AURC (Area Under Risk-Coverage curve, lower better), selective accuracy @80% coverage
(higher better), and risk-coverage curves. Computed with the ENTROPY score for all models,
and with the evidential VACUITY score (u = C/S) for the evidential subset (DD-CoMB, Sensoy).
These are the decision-relevant metrics: unlike AUROC they reward being accurate AND
well-ranked, so a weak-but-self-aware classifier (Sensoy) no longer looks good.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "savefig.dpi": 220})
DATASETS = [("deepweeds_no_bg_mergeval", "DeepWeeds"),
            ("plant_seedlings_v2_split", "Plant Seedlings V2"),
            ("bloodmnist_224_imagefolder_mergeval", "BloodMNIST"),
            ("ssl_patches", "SSL-Patches")]
# (label, template, is_ours, color, ls)
MODELS_ALL = [
    ("DINOv2", "dinov2_standard/{DS}", False, "#1f77b4", "-"),
    ("ConvNeXt-V2", "convnextv2_standard/{DS}", False, "#ff7f0e", "-"),
    ("DINOv2 · MC-Dropout", "baselines/mc_dropout/dinov2_vitb14/{DS}", False, "#2ca02c", "--"),
    ("DINOv2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}", False, "#8c564b", ":"),
    ("ConvNeXt-V2 · MC-Dropout", "baselines/mc_dropout/convnextv2_tiny/{DS}", False, "#9467bd", "--"),
    ("ConvNeXt-V2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}", False, "#e377c2", ":"),
    ("Vanilla Deep Ensemble", "vanilla_ens/{DS}", False, "#7f7f7f", "-."),
    ("DD-CoMB (ours)", "dino_cnx_dempster/{DS}", True, "#000000", "-"),
]
MODELS_EVID = [
    ("DINOv2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}", False, "#8c564b", ":"),
    ("ConvNeXt-V2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}", False, "#e377c2", ":"),
    ("DD-CoMB (ours)", "dino_cnx_dempster/{DS}", True, "#000000", "-"),
]
LOGIT_ROOTS = {"deepweeds_no_bg_mergeval": ["results_calib"],
               "plant_seedlings_v2_split": ["results_calib_fix_exp", "results_calib"],
               "ssl_patches": ["results_calib_fix_exp", "results_calib"],
               "bloodmnist_224_imagefolder_mergeval": ["results_calib", "results_calib_fix_exp"]}
GRID = np.linspace(0.05, 1.0, 40)
SHOTS = [1, 5, 10, 20]
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def find_npz(tmpl, dskey, s, r):
    for root in LOGIT_ROOTS[dskey]:
        f = os.path.join("kshot", root, tmpl.format(DS=dskey), f"{s}_shot_run_{r}_logits.npz")
        if os.path.isfile(f): return np.load(f)
    return None
def aurc(sc, inc):
    o = np.argsort(sc, kind="mergesort"); e = inc[o].astype(float)
    return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())
def rc_curve(sc, inc):
    o = np.argsort(sc, kind="mergesort"); e = inc[o].astype(float); N = len(e)
    return np.array([e[:max(1, int(round(c * N)))].mean() for c in GRID])
def selacc(sc, inc, cov=0.8):
    o = np.argsort(sc, kind="mergesort"); k = max(1, int(round(cov * len(o))))
    return 1.0 - inc[o][:k].mean()

_cache = {}
def gather(tmpl, dskey, s, mode):
    key = (tmpl, dskey, s, mode)
    if key in _cache: return _cache[key]
    A, S, R = [], [], []
    for r in range(1, 21):
        z = find_npz(tmpl, dskey, s, r)
        if z is None: continue
        y = z["test_y"]
        if mode == "entropy":
            p = softmax(z["test_logits"]); sc = -(p * np.log(np.clip(p, 1e-12, None))).sum(1); pred = p.argmax(1)
        else:
            if "test_alpha" not in z: continue
            al = z["test_alpha"]; sc = al.shape[1] / np.clip(al.sum(1), 1e-9, None); pred = al.argmax(1)
        inc = (pred != y).astype(int)
        A.append(aurc(sc, inc)); S.append(selacc(sc, inc)); R.append(rc_curve(sc, inc))
    res = None if not A else (float(np.mean(A)), float(np.mean(S)), np.mean(R, axis=0),
                              float(np.std(A)), float(np.std(S)))
    _cache[key] = res; return res

# ---------------- heatmaps ----------------
def heat_matrix(models, s, mode, which, worst):
    M = np.full((len(models), len(DATASETS) + 1), np.nan)
    for i, (lab, tmpl, ours, *_ ) in enumerate(models):
        for j, (dk, _) in enumerate(DATASETS):
            g = gather(tmpl, dk, s, mode); M[i, j] = np.nan if g is None else (g[0] if which == "aurc" else g[1])
        M[i, -1] = np.nanmax(M[i, :-1]) if worst == "max" else np.nanmin(M[i, :-1])
    return M
def draw_heat(ax, M, models, vmin, vmax, cmap, title):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DATASETS) + 1)); ax.set_xticklabels([d[1] for d in DATASETS] + ["Worst-case"],
        fontsize=8.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m[0] for m in models], fontsize=9)
    for k, m in enumerate(models):
        if m[2]: ax.get_yticklabels()[k].set_fontweight("bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8.3)
    ax.axvline(len(DATASETS) - 0.5, color="black", lw=2.4)
    for k, m in enumerate(models):
        if m[2]: ax.add_patch(Rectangle((-0.5, k - 0.5), M.shape[1], 1, fill=False, edgecolor="#1a5cff", lw=2.2))
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True); ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0); ax.set_title(title, fontsize=11)
    return im
def heat_fig(models, mode, which, worst, vmin, vmax, cmap, mlabel, fname):
    fig, axes = plt.subplots(1, 2, figsize=(3.1 + 0.9 * len(DATASETS), 0.62 * len(models) + 1.6))
    for ax, s in zip(axes, [5, 20]):
        im = draw_heat(ax, heat_matrix(models, s, mode, which, worst), models, vmin, vmax, cmap, f"{s}-shot")
    axes[1].set_yticklabels([])
    fig.suptitle(mlabel, fontsize=12.5, fontweight="bold", y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02).ax.tick_params(labelsize=8)
    fig.savefig(f"docs/{fname}", bbox_inches="tight"); plt.close(fig); print("wrote docs/" + fname)

# ---------------- risk-coverage curves ----------------
def rc_fig(models, mode, s, fname):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True, sharey=True)
    for ax, (dk, dn) in zip(axes.flat, DATASETS):
        for lab, tmpl, ours, col, ls in models:
            g = gather(tmpl, dk, s, mode)
            if g is None: continue
            ax.plot(GRID, g[2], ls, color=col, lw=2.6 if ours else 1.5, zorder=5 if ours else 2, label=lab)
        ax.set_title(dn, fontsize=11); ax.grid(alpha=0.3, lw=0.6); ax.set_ylim(0, 1.0); ax.set_xlim(0.05, 1.0)
    for ax in axes[:, 0]: ax.set_ylabel("Risk (error rate on kept set)")
    for ax in axes[1, :]: ax.set_xlabel("Coverage (fraction of most-confident kept)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Risk-Coverage curves ({mode}, {s}-shot) — lower is better", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97]); fig.savefig(f"docs/{fname}", bbox_inches="tight"); plt.close(fig)
    print("wrote docs/" + fname)

# ---------------- ROC curves (misclassification detection, AUROC-H) ----------------
FPRG = np.linspace(0, 1, 101)
def roc_mean(tmpl, dskey, s, mode):
    T, A = [], []
    for r in range(1, 21):
        z = find_npz(tmpl, dskey, s, r)
        if z is None: continue
        y = z["test_y"]
        if mode == "entropy":
            p = softmax(z["test_logits"]); sc = -(p * np.log(np.clip(p, 1e-12, None))).sum(1); pred = p.argmax(1)
        else:
            if "test_alpha" not in z: continue
            al = z["test_alpha"]; sc = al.shape[1] / np.clip(al.sum(1), 1e-9, None); pred = al.argmax(1)
        inc = (pred != y).astype(int); P = int(inc.sum()); N = len(inc) - P
        if P == 0 or N == 0: continue
        o = np.argsort(-sc, kind="mergesort"); ins = inc[o]
        tpr = np.concatenate([[0.0], np.cumsum(ins) / P])
        fpr = np.concatenate([[0.0], np.cumsum(1 - ins) / N])
        T.append(np.interp(FPRG, fpr, tpr)); A.append(float(np.trapz(tpr, fpr)))
    return (np.mean(T, axis=0), float(np.mean(A))) if T else (None, None)

def roc_fig(models, mode, s, fname):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), sharex=True, sharey=True)
    for ax, (dk, dn) in zip(axes.flat, DATASETS):
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.0, ls=(0, (4, 4)), zorder=1)
        for lab, tmpl, ours, col, ls in models:
            mt, au = roc_mean(tmpl, dk, s, mode)
            if mt is None: continue
            ax.plot(FPRG, mt, ls, color=col, lw=2.8 if ours else 1.5,
                    zorder=5 if ours else 2, label=f"{lab} ({au:.2f})")
        ax.set_title(dn, fontsize=11); ax.grid(alpha=0.3, lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False, handlelength=1.4, labelspacing=0.25)
    for ax in axes[:, 0]: ax.set_ylabel("True positive rate (errors caught)")
    for ax in axes[1, :]: ax.set_xlabel("False positive rate (correct predictions flagged)")
    fig.suptitle(f"ROC curves for misclassification detection ({mode}, {s}-shot) — AUROC-H in legend",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"docs/{fname}", bbox_inches="tight"); plt.close(fig)
    print("wrote docs/" + fname)

# ---------------- few-shot curves (metric vs support size) ----------------
def curve_fig(models, mode, which, ylabel, title, fname, ylim=(0.0, 1.0)):
    idx = 0 if which == "aurc" else 1
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True)
    for ax, (dk, dn) in zip(axes.flat, DATASETS):
        for lab, tmpl, ours, col, ls in models:
            ys = [(lambda g: np.nan if g is None else g[idx])(gather(tmpl, dk, s, mode)) for s in SHOTS]
            ax.plot(SHOTS, ys, ls, color=col, marker="o",
                    lw=2.8 if ours else 1.6, ms=7 if ours else 4.5,
                    zorder=5 if ours else 2, label=lab)
        ax.set_xscale("log"); ax.set_xticks(SHOTS); ax.set_xticklabels(SHOTS)
        ax.set_title(dn, fontsize=11); ax.grid(alpha=0.3, lw=0.6); ax.set_ylim(*ylim)
    for ax in axes[:, 0]: ax.set_ylabel(ylabel)
    for ax in axes[1, :]: ax.set_xlabel("support size $k$ (shots per class)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97]); fig.savefig(f"docs/{fname}", bbox_inches="tight"); plt.close(fig)
    print("wrote docs/" + fname)

# entropy (all models)
curve_fig(MODELS_ALL, "entropy", "aurc", "AURC (lower better)",
          "AURC vs. support size — per dataset (entropy)", "fig_curves_aurc.png")
curve_fig(MODELS_ALL, "entropy", "selacc", "Selective accuracy @80% coverage (higher better)",
          "Selective accuracy @80% vs. support size — per dataset (entropy)", "fig_curves_selacc80.png")
heat_fig(MODELS_ALL, "entropy", "aurc", "max", 0.0, 0.35, "RdYlGn_r",
         "AURC (entropy) — lower better", "fig_heatmap_aurc.png")
heat_fig(MODELS_ALL, "entropy", "selacc", "min", 0.2, 1.0, "RdYlGn",
         "Selective accuracy @80% coverage (entropy) — higher better", "fig_heatmap_selacc80.png")
rc_fig(MODELS_ALL, "entropy", 5, "fig_curves_riskcoverage_5shot.png")
rc_fig(MODELS_ALL, "entropy", 20, "fig_curves_riskcoverage_20shot.png")
roc_fig(MODELS_ALL, "entropy", 5, "fig_curves_roc_5shot.png")
roc_fig(MODELS_ALL, "entropy", 20, "fig_curves_roc_20shot.png")
# vacuity (evidential subset)
heat_fig(MODELS_EVID, "vacuity", "aurc", "max", 0.0, 0.35, "RdYlGn_r",
         "AURC (vacuity) — evidential models, lower better", "fig_heatmap_aurc_vacuity.png")
heat_fig(MODELS_EVID, "vacuity", "selacc", "min", 0.2, 1.0, "RdYlGn",
         "Selective accuracy @80% (vacuity) — evidential models, higher better", "fig_heatmap_selacc80_vacuity.png")
rc_fig(MODELS_EVID, "vacuity", 5, "fig_curves_riskcoverage_vacuity_5shot.png")
rc_fig(MODELS_EVID, "vacuity", 20, "fig_curves_riskcoverage_vacuity_20shot.png")
roc_fig(MODELS_EVID, "vacuity", 5, "fig_curves_roc_vacuity_5shot.png")
roc_fig(MODELS_EVID, "vacuity", 20, "fig_curves_roc_vacuity_20shot.png")
print("done")
