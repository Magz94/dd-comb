#!/usr/bin/env python3
"""Post-hoc calibration heatmaps for the thesis: ECE before vs after temperature
scaling (TS). For each (model, dataset, shot, run) the temperature T is fit on the
held-out calibration split by minimising NLL, then applied to the test logits; ECE
(15-bin) is recomputed. One figure per shot with two panels: ECE raw | ECE after TS.
Lower is better (reversed colormap: green = well-calibrated). Sensoy = MLP head only.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                     "axes.linewidth": 0.8, "savefig.dpi": 220})

DATASETS = [("deepweeds_no_bg_mergeval", "DeepWeeds"),
            ("plant_seedlings_v2_split", "Plant Seedlings"),
            ("bloodmnist_224_imagefolder_mergeval", "BloodMNIST"),
            ("ssl_patches", "SSL-Patches")]
MODELS = [
    ("DINOv2", "dinov2_standard/{DS}", False),
    ("ConvNeXt-V2", "convnextv2_standard/{DS}", False),
    ("DINOv2 · MC-Dropout", "baselines/mc_dropout/dinov2_vitb14/{DS}", False),
    ("DINOv2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}", False),
    ("ConvNeXt-V2 · MC-Dropout", "baselines/mc_dropout/convnextv2_tiny/{DS}", False),
    ("ConvNeXt-V2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}", False),
    ("Vanilla Deep Ensemble", "vanilla_ens/{DS}", False),
    ("DD-CoMB (ours)", "dino_cnx_dempster/{DS}", True),
]
# priority of result roots per dataset (first that has the logits+calib is used)
ROOTS = {"deepweeds_no_bg_mergeval": ["results_calib"],
         "plant_seedlings_v2_split": ["results_calib_fix_exp", "results_calib"],
         "ssl_patches": ["results_calib_fix_exp", "results_calib"],
         "bloodmnist_224_imagefolder_mergeval": ["results_calib", "results_calib_fix_exp"]}

def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def ece15(logits, y, nb=15):
    p = softmax(logits); conf = p.max(1); pred = p.argmax(1); corr = (pred == y).astype(float)
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.any(): e += abs(corr[m].mean() - conf[m].mean()) * m.mean()
    return e
def fit_T(clog, cy):
    idx = np.arange(len(cy)); gr = 0.6180339887
    def nll(T):
        p = np.clip(softmax(clog / T), 1e-12, None); return -np.log(p[idx, cy]).mean()
    a, b = 0.30, 8.0; c = b - gr * (b - a); d = a + gr * (b - a); fc, fd = nll(c), nll(d)
    for _ in range(28):
        if fc < fd: b, d, fd = d, c, fc; c = b - gr * (b - a); fc = nll(c)
        else: a, c, fc = c, d, fd; d = a + gr * (b - a); fd = nll(d)
    return (a + b) / 2
def find_npz(tmpl, dskey, s, r):
    for root in ROOTS[dskey]:
        f = os.path.join("kshot", root, tmpl.format(DS=dskey), f"{s}_shot_run_{r}_logits.npz")
        if os.path.isfile(f):
            z = np.load(f)
            if "calib_logits" in z and "test_logits" in z: return z
    return None
def raw_ts(tmpl, dskey, s):
    raws, tss = [], []
    for r in range(1, 21):
        z = find_npz(tmpl, dskey, s, r)
        if z is None: continue
        tl, ty, cl, cy = z["test_logits"], z["test_y"], z["calib_logits"], z["calib_y"]
        raws.append(ece15(tl, ty)); tss.append(ece15(tl / fit_T(cl, cy), ty))
    return (np.mean(raws) if raws else np.nan, np.mean(tss) if tss else np.nan)

def matrix(kind, s):
    M = np.full((len(MODELS), len(DATASETS) + 1), np.nan)
    for i, (lab, tmpl, ours) in enumerate(MODELS):
        for j, (dskey, _) in enumerate(DATASETS):
            r, t = raw_ts(tmpl, dskey, s); M[i, j] = r if kind == "raw" else t
        M[i, -1] = np.nanmax(M[i, :-1])   # worst-case for ECE = highest (worst)
    return M

def draw(ax, M, vmin, vmax, title):
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DATASETS) + 1))
    ax.set_xticklabels([d[1] for d in DATASETS] + ["Worst-case"], fontsize=8.5,
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m[0] for m in MODELS], fontsize=9)
    for k, (lab, _, ours) in enumerate(MODELS):
        if ours: ax.get_yticklabels()[k].set_fontweight("bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8.3, color="#111")
    ax.axvline(len(DATASETS) - 0.5, color="black", lw=2.4)
    for k, (lab, _, ours) in enumerate(MODELS):
        if ours: ax.add_patch(Rectangle((-0.5, k - 0.5), M.shape[1], 1, fill=False, edgecolor="#1a5cff", lw=2.2))
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True); ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=11, pad=6)
    return im

VMIN, VMAX = 0.02, 0.35
for s in [5, 20]:
    fig, axes = plt.subplots(1, 2, figsize=(3.1 + 0.9 * len(DATASETS), 0.62 * len(MODELS) + 1.6))
    draw(axes[0], matrix("raw", s), VMIN, VMAX, "ECE (raw)")
    im = draw(axes[1], matrix("ts", s), VMIN, VMAX, "ECE (after temperature scaling)")
    axes[1].set_yticklabels([])
    fig.suptitle(f"Post-hoc calibration — Expected Calibration Error, {s}-shot  (lower better)",
                 fontsize=12, y=1.02, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02); cbar.ax.tick_params(labelsize=8)
    fig.savefig(f"docs/fig_heatmap_calibration_{s}shot.png", bbox_inches="tight"); plt.close(fig)
    print(f"wrote docs/fig_heatmap_calibration_{s}shot.png")
print("done")
