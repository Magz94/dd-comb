#!/usr/bin/env python3
"""Rebuild the four-dataset comparison HTML with 8 metrics per dataset:
Accuracy, Macro-F1, ECE (raw), ECE (temp-scaled), MCE (raw), MCE (temp-scaled),
AUROC-H, FPR95-H. Raw/TS ECE & MCE are computed from the exported logits (T fit per
run on the held-out calib split); accuracy/F1/AUROC/FPR from the result JSONs.
"""
import json, os, glob
import numpy as np

SHOTS = [1, 5, 10, 20]
DSETS = [("deepweeds_no_bg_mergeval", "DeepWeeds"),
         ("plant_seedlings_v2_split", "Plant Seedlings V2"),
         ("bloodmnist_224_imagefolder_mergeval", "BloodMNIST"),
         ("ssl_patches", "SSL-Patches")]
# (label, template, cls)
MODELS = [
    ("DINOv2 (single, MLP)", "dinov2_standard/{DS}", "single"),
    ("ConvNeXt-V2-Tiny (single, MLP)", "convnextv2_standard/{DS}", "single"),
    ("DINOv2 — MC Dropout", "baselines/mc_dropout/dinov2_vitb14/{DS}", "base"),
    ("DINOv2 — Sensoy EDL (linear)", "baselines/sensoy/dinov2_vitb14/{DS}", "base"),
    ("DINOv2 — Sensoy EDL (MLP)", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}", "base"),
    ("ConvNeXt-V2-Tiny — MC Dropout", "baselines/mc_dropout/convnextv2_tiny/{DS}", "base"),
    ("ConvNeXt-V2-Tiny — Sensoy EDL (linear)", "baselines/sensoy/convnextv2_tiny/{DS}", "base"),
    ("ConvNeXt-V2-Tiny — Sensoy EDL (MLP)", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}", "base"),
    ("DINOv2 — Sensoy EDL (shared head, exp-CE)", "edl_headmatched_dinov2_ce/{DS}", "base"),
    ("DINOv2 — Sensoy EDL (shared head, Brier)", "edl_headmatched_dinov2_brier/{DS}", "base"),
    ("ConvNeXt-V2-Tiny — Sensoy EDL (shared head, exp-CE)", "edl_headmatched_cnx_ce/{DS}", "base"),
    ("ConvNeXt-V2-Tiny — Sensoy EDL (shared head, Brier)", "edl_headmatched_cnx_brier/{DS}", "base"),
    ("Vanilla deep ensemble (DINOv2 + ConvNeXt-V2, MLP)", "vanilla_ens/{DS}", "base"),
    ("DD-CoBE: DINOv2 + ConvNeXt-V2-Tiny (Dempster, ours)", "dino_cnx_dempster/{DS}", "ours"),
]
LOGIT_ROOTS = {"deepweeds_no_bg_mergeval": ["results_calib"],
               "plant_seedlings_v2_split": ["results_calib_fix_exp", "results_calib"],
               "ssl_patches": ["results_calib_fix_exp", "results_calib"],
               "bloodmnist_224_imagefolder_mergeval": ["results_calib", "results_calib_fix_exp"]}

def base_root(tmpl, dskey):
    if "edl_headmatched" in tmpl:
        return "results_calib"          # all head-matched EDL runs live here
    if "baselines_edl_mlp" in tmpl:
        return "results_calib" if dskey == "deepweeds_no_bg_mergeval" else "results_calib_fix_exp"
    return "results_calib_fix" if dskey in ("plant_seedlings_v2_split", "ssl_patches") else "results_calib"

def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def ece15(lg, y, nb=15):
    p = softmax(lg); c = p.max(1); pr = p.argmax(1); ok = (pr == y).astype(float)
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (c > b[i]) & (c <= b[i + 1])
        if m.any(): e += abs(ok[m].mean() - c[m].mean()) * m.mean()
    return e
def mce15(lg, y, nb=15):
    p = softmax(lg); c = p.max(1); pr = p.argmax(1); ok = (pr == y).astype(float)
    b = np.linspace(0, 1, nb + 1); mx = 0.0
    for i in range(nb):
        m = (c > b[i]) & (c <= b[i + 1])
        if m.any(): mx = max(mx, abs(ok[m].mean() - c[m].mean()))
    return mx
def nll_(lg, y):
    p = softmax(lg); return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
def brier_(lg, y):
    p = softmax(lg); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return float(((p - oh) ** 2).sum(1).mean())
_Ts = np.linspace(0.4, 6.0, 48)
def fit_T(clog, cy):
    z = clog[:, :, None] / _Ts[None, None, :]
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    nll = -np.log(np.clip(p[np.arange(len(cy)), cy, :], 1e-12, None)).mean(0)
    return _Ts[int(np.argmin(nll))]
def find_npz(tmpl, dskey, s, r):
    for root in LOGIT_ROOTS[dskey]:
        f = os.path.join("kshot", root, tmpl.format(DS=dskey), f"{s}_shot_run_{r}_logits.npz")
        if os.path.isfile(f):
            z = np.load(f)
            if "calib_logits" in z and "test_logits" in z: return z
    return None
def ms(a):
    a = np.array(a, float)
    return (float(np.nanmean(a)), float(np.nanstd(a, ddof=1))) if len(a) > 1 else (float('nan'), 0.0)

data = {}
for dskey, dsname in DSETS:
    data[dsname] = {}
    for lab, tmpl, cls in MODELS:
        broot = base_root(tmpl, dskey)
        data[dsname][lab] = {"cls": cls}
        for s in SHOTS:
            acc = []; f1 = []; au = []; fp = []
            er = []; et = []; mr = []; mt = []
            nr = []; nt = []; brr = []; brt = []
            for r in range(1, 21):
                jf = os.path.join("kshot", broot, tmpl.format(DS=dskey), f"{s}_shot_run_{r}.json")
                if os.path.isfile(jf):
                    d = json.load(open(jf)); acc.append(d["test_accuracy"])
                    p, rc = d.get("test_precision"), d.get("test_recall")
                    f1.append(2 * p * rc / (p + rc) if (p and rc and p + rc) else 0.0)
                    au.append(d.get("auroc_err_pred_entropy", np.nan)); fp.append(d.get("fpr95_err_pred_entropy", np.nan))
                z = find_npz(tmpl, dskey, s, r)
                if z is not None:
                    tl, ty, cl, cy = z["test_logits"], z["test_y"], z["calib_logits"], z["calib_y"]
                    T = fit_T(cl, cy)
                    er.append(ece15(tl, ty)); et.append(ece15(tl / T, ty))
                    mr.append(mce15(tl, ty)); mt.append(mce15(tl / T, ty))
                    nr.append(nll_(tl, ty)); nt.append(nll_(tl / T, ty))
                    brr.append(brier_(tl, ty)); brt.append(brier_(tl / T, ty))
            data[dsname][lab][s] = {"Accuracy": ms(acc), "Macro-F1": ms(f1),
                "ECE": ms(er), "ECE (TS)": ms(et), "MCE": ms(mr), "MCE (TS)": ms(mt),
                "NLL": ms(nr), "NLL (TS)": ms(nt), "Brier": ms(brr), "Brier (TS)": ms(brt),
                "AUROC-H": ms(au), "FPR95-H": ms(fp)}
    print("computed", dsname)

# ---------------- render ----------------
DSORDER = [d[1] for d in DSETS]
METRICS = [("Accuracy", "↑", True), ("Macro-F1", "↑", True),
           ("ECE", "↓", False), ("ECE (TS)", "↓", False),
           ("MCE", "↓", False), ("MCE (TS)", "↓", False),
           ("NLL", "↓", False), ("NLL (TS)", "↓", False),
           ("Brier", "↓", False), ("Brier (TS)", "↓", False),
           ("AUROC-H", "↑", True), ("FPR95-H", "↓", False)]
def slug(s): return s.lower().replace(" ", "-")
def cell(t):
    mn, sd = t; return "—" if mn != mn else f'{mn:.3f}<span class="s">±{sd:.3f}</span>'
def rowcls(m):
    if m.startswith("DD-CoBE"): return "ours"
    if "single, MLP" in m: return "single"
    return "base"
H = ['<!doctype html><html><head><meta charset="utf-8"><title>Four-dataset comparison</title>',
'<style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:26px;background:#fafafa;color:#111}',
'h1{font-size:22px}h2{font-size:18px;margin-top:30px;border-bottom:3px solid #111;padding-bottom:4px}',
'nav{margin:10px 0 18px;font-size:14px}nav a{margin-right:14px}',
'table{border-collapse:collapse;margin:6px 0 20px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.07)}',
'caption{caption-side:top;text-align:left;font-weight:700;font-size:13px;padding:5px 0}',
'th,td{border:1px solid #ddd;padding:5px 10px;font-size:12.5px;text-align:right}',
'th{background:#111;color:#fff}th:first-child,td.m{text-align:left;white-space:nowrap}td.m{font-weight:600}',
'.s{color:#888;font-size:10px;margin-left:3px}',
'tr.ours td{background:#eef6ff}tr.ours td.m{font-weight:800}td.win{font-weight:800;background:#dff5e1;color:#0b4f16}',
'tr.single td{background:#fffbe9}tr.base td{background:#f3f0fa}</style></head><body>',
'<h1>DD-CoBE vs single backbones & uncertainty baselines — four few-shot OOD datasets</h1>',
'<p>Frozen backbones, few-shot (k∈{1,5,10,20}), <b>mean ± std over 20 runs</b>. Twelve metrics per dataset, '
'including calibration and proper scoring rules <b>before</b> (ECE, MCE, NLL, Brier) and <b>after temperature '
'scaling</b> (ECE (TS), MCE (TS), NLL (TS), Brier (TS)); T is fit per run on the held-out calibration split by '
'minimising NLL. Green = best per column (higher better for Accuracy/Macro-F1/AUROC-H, '
'lower for ECE/MCE/NLL/Brier/FPR95-H). Blue = our fusion, yellow = single backbones, purple = baselines. '
'Plant Seedlings V2 / SSL-Patches from the validated clean re-run; DeepWeeds/BloodMNIST verified unaffected.</p>']
H.append('<nav><b>Jump to:</b> ' + ' '.join(f'<a href="#{slug(d)}">{d}</a>' for d in DSORDER) + '</nav>')
for ds in DSORDER:
    H.append(f'<h2 id="{slug(ds)}">{ds}</h2>')
    for met, arrow, hib in METRICS:
        best = {}
        for s in SHOTS:
            vals = [(m, data[ds][m][s][met][0]) for m, _, _ in MODELS]
            vals = [(m, v) for m, v in vals if v == v]
            if vals: best[s] = (max if hib else min)(vals, key=lambda x: x[1])[0]
        H.append(f'<table><caption>{met} {arrow}</caption><thead><tr><th>Model</th>' +
                 ''.join(f'<th>{s}-shot</th>' for s in SHOTS) + '</tr></thead><tbody>')
        for m, tmpl, cls in MODELS:
            H.append(f'<tr class="{rowcls(m)}"><td class="m">{m}</td>')
            for s in SHOTS:
                c = "win" if best.get(s) == m else ""
                H.append(f'<td class="{c}">{cell(data[ds][m][s][met])}</td>')
            H.append('</tr>')
        H.append('</tbody></table>')
H.append('</body></html>')
open("docs/four_dataset_comparison.html", "w").write("\n".join(H))
print("wrote docs/four_dataset_comparison.html  (12 metrics x 4 datasets)")
