#!/usr/bin/env python3
"""Few-shot curves for the thesis: Accuracy, Macro-F1 and AUROC-H (entropy-based
misclassification detection) as a function of support size k in {1,5,10,20}, one panel
per dataset (2x2), one line per model. DD-CoMB is emphasised. Sensoy = MLP head only.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "savefig.dpi": 220})

SHOTS = [1, 5, 10, 20]
DATASETS = [("deepweeds_no_bg_mergeval", "DeepWeeds"),
            ("plant_seedlings_v2_split", "Plant Seedlings V2"),
            ("bloodmnist_224_imagefolder_mergeval", "BloodMNIST"),
            ("ssl_patches", "SSL-Patches")]
# (label, template, type, color, linestyle, is_ours)
MODELS = [
    ("DINOv2", "dinov2_standard/{DS}", "main", "#1f77b4", "-", False),
    ("ConvNeXt-V2", "convnextv2_standard/{DS}", "main", "#ff7f0e", "-", False),
    ("DINOv2 · MC-Dropout", "baselines/mc_dropout/dinov2_vitb14/{DS}", "main", "#2ca02c", "--", False),
    ("DINOv2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}", "mlp", "#8c564b", ":", False),
    ("ConvNeXt-V2 · MC-Dropout", "baselines/mc_dropout/convnextv2_tiny/{DS}", "main", "#9467bd", "--", False),
    ("ConvNeXt-V2 · Sensoy-EDL", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}", "mlp", "#e377c2", ":", False),
    ("Vanilla Deep Ensemble", "vanilla_ens/{DS}", "main", "#7f7f7f", "-.", False),
    ("DD-CoMB (ours)", "dino_cnx_dempster/{DS}", "dempster", "#000000", "-", True),
]
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def mce15(lg, y, nb=15):
    p = softmax(lg); c = p.max(1); pr = p.argmax(1); ok = (pr == y).astype(float)
    b = np.linspace(0, 1, nb + 1); mx = 0.0
    for i in range(nb):
        m = (c > b[i]) & (c <= b[i + 1])
        if m.any(): mx = max(mx, abs(ok[m].mean() - c[m].mean()))
    return mx
def ece15(lg, y, nb=15):
    p = softmax(lg); c = p.max(1); pr = p.argmax(1); ok = (pr == y).astype(float)
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (c > b[i]) & (c <= b[i + 1])
        if m.any(): e += abs(ok[m].mean() - c[m].mean()) * m.mean()
    return e
def root_json(mtype, dskey):
    if mtype == "mlp":
        return "results_calib" if dskey == "deepweeds_no_bg_mergeval" else "results_calib_fix_exp"
    return "results_calib_fix" if dskey in ("plant_seedlings_v2_split", "ssl_patches") else "results_calib"
LOGIT_ROOTS = {"deepweeds_no_bg_mergeval": ["results_calib"],
               "plant_seedlings_v2_split": ["results_calib_fix_exp", "results_calib"],
               "ssl_patches": ["results_calib_fix_exp", "results_calib"],
               "bloodmnist_224_imagefolder_mergeval": ["results_calib", "results_calib_fix_exp"]}
_Ts = np.linspace(0.4, 6.0, 48)
def fit_T(cl, cy):
    z = cl[:, :, None] / _Ts[None, None, :]; z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return _Ts[int(np.argmin(-np.log(np.clip(p[np.arange(len(cy)), cy, :], 1e-12, None)).mean(0)))]
def _find_npz(tmpl, dskey, s, r):
    for root in LOGIT_ROOTS[dskey]:
        f = os.path.join("kshot", root, tmpl.format(DS=dskey), f"{s}_shot_run_{r}_logits.npz")
        if os.path.isfile(f):
            z = np.load(f)
            if "calib_logits" in z: return z
    return None
_ts_cache = {}
def _nll(lg, y):
    p = softmax(lg); return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
def _brier(lg, y):
    p = softmax(lg); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return float(((p - oh) ** 2).sum(1).mean())
def ts_vals(tmpl, dskey, s):
    key = (tmpl, dskey, s)
    if key in _ts_cache: return _ts_cache[key]
    et, mt, nr, nt, br, bt = [], [], [], [], [], []
    for r in range(1, 21):
        z = _find_npz(tmpl, dskey, s, r)
        if z is None: continue
        tl, ty, cl, cy = z["test_logits"], z["test_y"], z["calib_logits"], z["calib_y"]
        T = fit_T(cl, cy)
        et.append(ece15(tl / T, ty)); mt.append(mce15(tl / T, ty))
        nr.append(_nll(tl, ty)); nt.append(_nll(tl / T, ty))
        br.append(_brier(tl, ty)); bt.append(_brier(tl / T, ty))
    f = lambda a: (np.nanmean(a), np.nanstd(a)) if a else (np.nan, 0.0)
    _ts_cache[key] = {"ece_ts": f(et), "mce_ts": f(mt),
                      "nll": f(nr), "nll_ts": f(nt), "brier": f(br), "brier_ts": f(bt)}
    return _ts_cache[key]
def val(metric, tmpl, mtype, dskey, s):
    if metric in ("ece_ts", "mce_ts", "nll", "nll_ts", "brier", "brier_ts"):
        return ts_vals(tmpl, dskey, s)[metric]
    base = os.path.join("kshot", root_json(mtype, dskey), tmpl.format(DS=dskey))
    vs = []
    for r in range(1, 21):
        f = os.path.join(base, f"{s}_shot_run_{r}.json")
        if not os.path.isfile(f): continue
        d = json.load(open(f))
        if metric == "acc": vs.append(d["test_accuracy"])
        elif metric == "f1":
            p, rc = d.get("test_precision"), d.get("test_recall")
            vs.append(2 * p * rc / (p + rc) if (p and rc and p + rc) else 0.0)
        elif metric == "auroc": vs.append(d.get("auroc_err_pred_entropy", np.nan))
        elif metric == "ece": vs.append(d["test_ece"])
        elif metric == "mce":
            if "test_mce" in d: vs.append(d["test_mce"])
            else:
                lf = os.path.join(base, f"{s}_shot_run_{r}_logits.npz")
                if os.path.isfile(lf):
                    z = np.load(lf); vs.append(mce15(z["test_logits"], z["test_y"]))
    return (np.nanmean(vs), np.nanstd(vs)) if vs else (np.nan, 0.0)

def figure(metric, ylabel, fname, ylim=None):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True)
    for ax, (dskey, dsname) in zip(axes.flat, DATASETS):
        for lab, tmpl, mtype, col, ls, ours in MODELS:
            ys = [val(metric, tmpl, mtype, dskey, s)[0] for s in SHOTS]
            ax.plot(SHOTS, ys, ls, color=col, marker="o",
                    lw=2.8 if ours else 1.6, ms=7 if ours else 4.5,
                    zorder=5 if ours else 2, label=lab)
        ax.set_xscale("log"); ax.set_xticks(SHOTS); ax.set_xticklabels(SHOTS)
        ax.set_title(dsname, fontsize=11); ax.grid(alpha=0.3, lw=0.6)
        if ylim: ax.set_ylim(*ylim)
    for ax in axes[:, 0]: ax.set_ylabel(ylabel)
    for ax in axes[1, :]: ax.set_xlabel("support size $k$ (shots per class)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(ylabel + " vs. support size — per dataset", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(f"docs/{fname}", bbox_inches="tight"); plt.close(fig); print("wrote docs/" + fname)

# common y-axis 0.0–1.0 across all curve figures for consistency
figure("acc", "Accuracy", "fig_curves_accuracy.png", ylim=(0.0, 1.0))
figure("f1", "Macro-F1", "fig_curves_macrof1.png", ylim=(0.0, 1.0))
figure("auroc", "AUROC-H (misclassification detection)", "fig_curves_auroc.png", ylim=(0.0, 1.0))
figure("ece", "Expected Calibration Error (raw, lower better)", "fig_curves_ece.png", ylim=(0.0, 1.0))
figure("mce", "Maximum Calibration Error (raw, lower better)", "fig_curves_mce.png", ylim=(0.0, 1.0))
figure("ece_ts", "Expected Calibration Error (after TS, lower better)", "fig_curves_ece_ts.png", ylim=(0.0, 1.0))
figure("mce_ts", "Maximum Calibration Error (after TS, lower better)", "fig_curves_mce_ts.png", ylim=(0.0, 1.0))
# proper scoring rules (NLL unbounded -> its own y-scale; Brier in [0,2], values <1)
figure("nll", "Negative Log-Likelihood (raw, lower better)", "fig_curves_nll.png", ylim=(0.0, 2.5))
figure("nll_ts", "Negative Log-Likelihood (after TS, lower better)", "fig_curves_nll_ts.png", ylim=(0.0, 2.5))
figure("brier", "Brier Score (raw, lower better)", "fig_curves_brier.png", ylim=(0.0, 1.0))
figure("brier_ts", "Brier Score (after TS, lower better)", "fig_curves_brier_ts.png", ylim=(0.0, 1.0))
print("done")
