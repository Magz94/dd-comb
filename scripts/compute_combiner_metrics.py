#!/usr/bin/env python3
"""AURC(entropy), AURC(vacuity), MCE, NLL, Brier (all raw) for the DeepWeeds modern
combiner comparison (Tables 13/14), from results_calib logits. Prints acc too, to
verify the run set matches the expose."""
import os, numpy as np
DS = "deepweeds_no_bg_mergeval"; ROOT = "kshot/results_calib"; SHOTS = [1, 5, 10, 20]
MODELS = [
    ("DINOv2 (single, MLP)", "dinov2_standard"),
    ("ConvNeXt-V2-Tiny (single, MLP)", "convnextv2_standard"),
    ("Vanilla deep ensemble", "vanilla_ens"),
    ("Softmax ensemble", "dino_cnx_softmax_ensemble"),
    ("Mean-evidence ensemble", "dino_cnx_mean_evidence"),
    ("DD-CoBE Dempster (ours)", "dino_cnx_dempster"),
]
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def mce15(lg, y, nb=15):
    p = softmax(lg); c = p.max(1); pr = p.argmax(1); ok = (pr == y).astype(float)
    b = np.linspace(0, 1, nb + 1); mx = 0.0
    for i in range(nb):
        m = (c > b[i]) & (c <= b[i + 1])
        if m.any(): mx = max(mx, abs(ok[m].mean() - c[m].mean()))
    return mx
def nll(lg, y): p = softmax(lg); return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
def brier(lg, y): p = softmax(lg); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1; return float(((p - oh) ** 2).sum(1).mean())
def aurc(sc, inc): o = np.argsort(sc, kind="mergesort"); e = inc[o].astype(float); return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())
for lab, tmpl in MODELS:
    row = []
    for s in SHOTS:
        A, M, N, B, Ue, Uv = [], [], [], [], [], []
        for r in range(1, 21):
            f = os.path.join(ROOT, tmpl, DS, f"{s}_shot_run_{r}_logits.npz")
            if not os.path.isfile(f): continue
            z = np.load(f); tl, ty = z["test_logits"], z["test_y"]
            p = softmax(tl); pred = p.argmax(1); inc = (pred != ty).astype(int)
            ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
            A.append((pred == ty).mean()); M.append(mce15(tl, ty)); N.append(nll(tl, ty))
            B.append(brier(tl, ty)); Ue.append(aurc(ent, inc))
            if "test_alpha" in z:
                al = z["test_alpha"]; vac = al.shape[1] / np.clip(al.sum(1), 1e-9, None)
                Uv.append(aurc(vac, (al.argmax(1) != ty).astype(int)))
        f_ = lambda a: (np.mean(a) if a else float('nan'))
        row.append((s, f_(A), f_(M), f_(N), f_(B), f_(Ue), (f_(Uv) if Uv else None)))
    print(lab)
    for s, a, m, n, b, ue, uv in row:
        extra = f"  AURCvac={uv:.4f}" if uv is not None else ""
        print(f"  {s:>2}-shot: acc={a:.4f}  MCE={m:.4f}  NLL={n:.4f}  Brier={b:.4f}  AURC={ue:.4f}{extra}")
