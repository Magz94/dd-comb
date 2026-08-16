#!/usr/bin/env python3
"""Calibration bar charts for the thesis: ECE and MCE, before vs after temperature
scaling, per model, one panel per dataset (2x2), at 20-shot. Two bars per model
(raw / after-TS); lower is better. Common y-axis across all datasets. Sensoy = MLP head.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "savefig.dpi": 220})
SHOTS = [5, 20]
DATASETS = [("deepweeds_no_bg_mergeval", "DeepWeeds"),
            ("plant_seedlings_v2_split", "Plant Seedlings V2"),
            ("bloodmnist_224_imagefolder_mergeval", "BloodMNIST"),
            ("ssl_patches", "SSL-Patches")]
MODELS = [
    ("DINOv2", "dinov2_standard/{DS}", False),
    ("ConvNeXt-V2", "convnextv2_standard/{DS}", False),
    ("DINOv2·MC-Drop", "baselines/mc_dropout/dinov2_vitb14/{DS}", False),
    ("DINOv2·Sensoy", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}", False),
    ("ConvNeXt·MC-Drop", "baselines/mc_dropout/convnextv2_tiny/{DS}", False),
    ("ConvNeXt·Sensoy", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}", False),
    ("Vanilla Ens.", "vanilla_ens/{DS}", False),
    ("DD-CoMB (ours)", "dino_cnx_dempster/{DS}", True),
]
ROOTS = {"deepweeds_no_bg_mergeval": ["results_calib"],
         "plant_seedlings_v2_split": ["results_calib_fix_exp", "results_calib"],
         "ssl_patches": ["results_calib_fix_exp", "results_calib"],
         "bloodmnist_224_imagefolder_mergeval": ["results_calib", "results_calib_fix_exp"]}
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def _bins(lg, y, nb=15):
    p = softmax(lg); c = p.max(1); pr = p.argmax(1); ok = (pr == y).astype(float)
    b = np.linspace(0, 1, nb + 1); gaps = []
    for i in range(nb):
        m = (c > b[i]) & (c <= b[i + 1])
        if m.any(): gaps.append((abs(ok[m].mean() - c[m].mean()), m.mean()))
    return gaps
def ece15(lg, y): return sum(g * w for g, w in _bins(lg, y))
def mce15(lg, y):
    g = _bins(lg, y); return max((x for x, _ in g), default=0.0)
_Ts = np.linspace(0.4, 6.0, 48)
def fit_T(cl, cy):
    z = cl[:, :, None] / _Ts[None, None, :]; z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return _Ts[int(np.argmin(-np.log(np.clip(p[np.arange(len(cy)), cy, :], 1e-12, None)).mean(0)))]
def _nll(lg, y):
    p = softmax(lg); return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
def _brier(lg, y):
    p = softmax(lg); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return float(((p - oh) ** 2).sum(1).mean())
def raw_ts(tmpl, dskey, shot):
    er, et, mr, mt, nr, nt, brr, brt = [], [], [], [], [], [], [], []
    for r in range(1, 21):
        z = None
        for root in ROOTS[dskey]:
            f = os.path.join("kshot", root, tmpl.format(DS=dskey), f"{shot}_shot_run_{r}_logits.npz")
            if os.path.isfile(f):
                zz = np.load(f)
                if "calib_logits" in zz: z = zz; break
        if z is None: continue
        tl, ty, cl, cy = z["test_logits"], z["test_y"], z["calib_logits"], z["calib_y"]
        T = fit_T(cl, cy)
        er.append(ece15(tl, ty)); et.append(ece15(tl / T, ty))
        mr.append(mce15(tl, ty)); mt.append(mce15(tl / T, ty))
        nr.append(_nll(tl, ty)); nt.append(_nll(tl / T, ty))
        brr.append(_brier(tl, ty)); brt.append(_brier(tl / T, ty))
    f = lambda a: (np.mean(a) if a else np.nan)
    return {"ece_raw": f(er), "ece_ts": f(et), "mce_raw": f(mr), "mce_ts": f(mt),
            "nll_raw": f(nr), "nll_ts": f(nt), "brier_raw": f(brr), "brier_ts": f(brt)}

# compute once for all shots
G = {(sh, dskey, tmpl): raw_ts(tmpl, dskey, sh)
     for sh in SHOTS for dskey, _ in DATASETS for _, tmpl, _ in MODELS}
def ylim_for(rk, tk):
    m = np.nanmax([max(G[(sh, d, t)][rk], G[(sh, d, t)][tk])
                   for sh in SHOTS for d, _ in DATASETS for _, t, _ in MODELS])
    return (0.0, float(np.ceil(m / 0.05) * 0.05 + 0.02))

def make_fig(shot, rk, tk, ylim, ylabel, title, fname):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x = np.arange(len(MODELS)); w = 0.4
    for ax, (dskey, dsname) in zip(axes.flat, DATASETS):
        raw = [G[(shot, dskey, t)][rk] for _, t, _ in MODELS]
        ts = [G[(shot, dskey, t)][tk] for _, t, _ in MODELS]
        ax.bar(x - w / 2, raw, w, color="#e8896b", label="raw", edgecolor="#7a3b28", lw=0.5)
        ax.bar(x + w / 2, ts, w, color="#4a9c6d", label="after TS", edgecolor="#245038", lw=0.5)
        for k, (lab, _, ours) in enumerate(MODELS):
            if ours: ax.axvspan(k - 0.5, k + 0.5, color="#1a5cff", alpha=0.07, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MODELS], rotation=35, ha="right", fontsize=8.5)
        for k, (lab, _, ours) in enumerate(MODELS):
            if ours: ax.get_xticklabels()[k].set_fontweight("bold")
        ax.set_title(dsname, fontsize=11); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3, lw=0.6); ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"docs/{fname}", bbox_inches="tight"); plt.close(fig); print("wrote docs/" + fname)

ece_ylim = ylim_for("ece_raw", "ece_ts")
mce_ylim = ylim_for("mce_raw", "mce_ts")
nll_ylim = ylim_for("nll_raw", "nll_ts")
brier_ylim = ylim_for("brier_raw", "brier_ts")
for shot in SHOTS:
    suf = "" if shot == 20 else f"_{shot}shot"
    make_fig(shot, "ece_raw", "ece_ts", ece_ylim, "Expected Calibration Error",
             f"Post-hoc calibration: ECE before vs after temperature scaling ({shot}-shot, lower better)",
             f"fig_bars_calibration{suf}.png")
    make_fig(shot, "mce_raw", "mce_ts", mce_ylim, "Maximum Calibration Error",
             f"Post-hoc calibration: MCE before vs after temperature scaling ({shot}-shot, lower better)",
             f"fig_bars_calibration_mce{suf}.png")
    make_fig(shot, "nll_raw", "nll_ts", nll_ylim, "Negative Log-Likelihood",
             f"Post-hoc calibration: NLL before vs after temperature scaling ({shot}-shot, lower better)",
             f"fig_bars_calibration_nll{suf}.png")
    make_fig(shot, "brier_raw", "brier_ts", brier_ylim, "Brier Score",
             f"Post-hoc calibration: Brier before vs after temperature scaling ({shot}-shot, lower better)",
             f"fig_bars_calibration_brier{suf}.png")
