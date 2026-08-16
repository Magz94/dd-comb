#!/usr/bin/env python3
"""Compute NLL and Brier (raw + after temperature scaling), all models x datasets,
to decide whether to report NLL, Brier, or both. Lower is better for both."""
import os
import numpy as np

DATASETS = [("deepweeds_no_bg_mergeval", "DeepWeeds"),
            ("plant_seedlings_v2_split", "Plant"),
            ("bloodmnist_224_imagefolder_mergeval", "BloodMNIST"),
            ("ssl_patches", "SSL-Patches")]
MODELS = [
    ("DINOv2", "dinov2_standard/{DS}"),
    ("ConvNeXt-V2", "convnextv2_standard/{DS}"),
    ("DINOv2.MC-Drop", "baselines/mc_dropout/dinov2_vitb14/{DS}"),
    ("DINOv2.Sensoy", "baselines_edl_mlp/sensoy/dinov2_vitb14/{DS}"),
    ("ConvNeXt.MC-Drop", "baselines/mc_dropout/convnextv2_tiny/{DS}"),
    ("ConvNeXt.Sensoy", "baselines_edl_mlp/sensoy/convnextv2_tiny/{DS}"),
    ("Vanilla Ens.", "vanilla_ens/{DS}"),
    ("DD-CoMB (ours)", "dino_cnx_dempster/{DS}"),
]
ROOTS = {"deepweeds_no_bg_mergeval": ["results_calib"],
         "plant_seedlings_v2_split": ["results_calib_fix_exp", "results_calib"],
         "ssl_patches": ["results_calib_fix_exp", "results_calib"],
         "bloodmnist_224_imagefolder_mergeval": ["results_calib", "results_calib_fix_exp"]}

def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def nll(lg, y):
    p = softmax(lg)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())

def brier(lg, y):
    p = softmax(lg); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return float(((p - oh) ** 2).sum(1).mean())

_Ts = np.linspace(0.4, 6.0, 48)
def fit_T(cl, cy):
    z = cl[:, :, None] / _Ts[None, None, :]; z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return _Ts[int(np.argmin(-np.log(np.clip(p[np.arange(len(cy)), cy, :], 1e-12, None)).mean(0)))]

def gather(tmpl, dskey, shot):
    nr, nt, br, bt = [], [], [], []
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
        nr.append(nll(tl, ty)); nt.append(nll(tl / T, ty))
        br.append(brier(tl, ty)); bt.append(brier(tl / T, ty))
    f = lambda a: (np.mean(a) if a else np.nan)
    return f(nr), f(nt), f(br), f(bt)

DATA = {(shot, tmpl, dk): gather(tmpl, dk, shot)
        for shot in (5, 20) for _, tmpl in MODELS for dk, _ in DATASETS}

for shot in (5, 20):
    print(f"\n================= {shot}-shot =================")
    for mi, mname in [(0, "NLL (raw)"), (1, "NLL (after TS)"), (2, "Brier (raw)"), (3, "Brier (after TS)")]:
        print(f"\n-- {mname}")
        print(f"{'Model':<17}" + "".join(f"{d[1]:>12}" for d in DATASETS) + f"{'WORST':>12}")
        for lab, tmpl in MODELS:
            vals = [DATA[(shot, tmpl, dk)][mi] for dk, _ in DATASETS]
            worst = np.nanmax(vals)
            cells = "".join(f"{v:>12.3f}" if not np.isnan(v) else f"{'--':>12}" for v in vals)
            print(f"{lab:<17}{cells}{worst:>12.3f}")
print("\ndone")
