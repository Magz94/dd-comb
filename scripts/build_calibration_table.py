#!/usr/bin/env python3
"""Temperature-scaling + native-uncertainty-score comparison table.

For each method, loads the per-(shot,run) logits exports (test + calib(val)),
fits one temperature T per run on the calibration split (NLL minimization),
and reports, averaged over runs per shot:
  * Accuracy (T-invariant),
  * ECE raw  and ECE after temperature scaling (TS),
  * AUROC-H / FPR95-H under the COMMON score (predictive entropy) for every method,
  * AUROC-H / FPR95-H under the NATIVE evidential score (vacuity = C/S) for the
    methods that produce a Dirichlet (Dempster, mean-evidence, Sensoy EDL).

Logits files store log-predictive-probabilities, so temperature scaling is
softmax(log_p / T); accuracy (argmax) is unchanged by T. Writes an HTML table.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SHOTS = [1, 5, 10, 20]
DSET = "deepweeds_no_bg_mergeval"
CALIB = "kshot/results_calib"

# (label, css-class, logits-dir, is_evidential)
# Group 1: calibration study & combine-rule ablation (plain-trained methods).
ABLATION = [
    ("DINOv2 (single)", "", f"{CALIB}/dinov2_standard/{DSET}", False),
    ("ConvNeXt-V2-Tiny (single)", "single", f"{CALIB}/convnextv2_standard/{DSET}", False),
    ("Vanilla deep ensemble (dino+cnx)", "", f"{CALIB}/vanilla_ens/{DSET}", False),
    ("DD-CoBE softmax ensemble (dino+cnx)", "ours", f"{CALIB}/dino_cnx_softmax_ensemble/{DSET}", False),
    ("DD-CoBE mean-evidence (dino+cnx)", "ours", f"{CALIB}/dino_cnx_mean_evidence/{DSET}", True),
    ("DD-CoBE Dempster (dino+cnx, ours)", "ours", f"{CALIB}/dino_cnx_dempster/{DSET}", True),
]
# Group 2: dedicated uncertainty baselines (separate comparison, presented later).
BASELINES = [
    ("DINOv2 — Sensoy EDL", "base", f"{CALIB}/baselines/sensoy/dinov2_vitb14/{DSET}", True),
    ("DINOv2 — MC Dropout", "base", f"{CALIB}/baselines/mc_dropout/dinov2_vitb14/{DSET}", False),
    ("ConvNeXt-V2-Tiny — Sensoy EDL", "base", f"{CALIB}/baselines/sensoy/convnextv2_tiny/{DSET}", True),
    ("ConvNeXt-V2-Tiny — MC Dropout", "base", f"{CALIB}/baselines/mc_dropout/convnextv2_tiny/{DSET}", False),
]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, nb=15):
    conf = probs.max(1); pred = probs.argmax(1); ok = (pred == y).astype(float)
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def entropy(p):
    p = np.clip(p, 1e-9, 1); return -(p * np.log(p)).sum(1)


def fpr95(score, label):
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    thr = np.quantile(pos, 0.05)
    return float((neg >= thr).mean())


def fit_T(calib_logits, calib_y):
    """Grid-search temperature minimizing NLL on calibration logits (=log probs)."""
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.20, 8.0, 400):
        p = softmax(calib_logits / T)
        nll = -np.log(np.clip(p[np.arange(len(calib_y)), calib_y], 1e-9, 1)).mean()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


def load_runs(d):
    return sorted(glob.glob(str(REPO / d / f"*_shot_run_*_logits.npz")))


def eval_method(d, evidential, shot):
    files = [f for f in load_runs(d) if Path(f).name.startswith(f"{shot}_shot_run_")]
    if not files:
        return None
    acc, ece_raw, ece_ts, au_e, fp_e, au_v, fp_v = ([] for _ in range(7))
    for f in files:
        z = np.load(f)
        tl, ty = z["test_logits"], z["test_y"]
        cl, cy = z["calib_logits"], z["calib_y"]
        p_raw = softmax(tl)
        pred = p_raw.argmax(1); inc = (pred != ty).astype(int)
        acc.append((pred == ty).mean())
        ece_raw.append(ece(p_raw, ty))
        T = fit_T(cl, cy)
        ece_ts.append(ece(softmax(tl / T), ty))
        au_e.append(roc_auc_score(inc, entropy(p_raw)) if inc.sum() else np.nan)
        fp_e.append(fpr95(entropy(p_raw), inc))
        if evidential and "test_alpha" in z:
            S = z["test_alpha"].sum(1); C = z["test_alpha"].shape[1]
            vac = C / np.clip(S, 1e-9, None)
            au_v.append(roc_auc_score(inc, vac) if inc.sum() else np.nan)
            fp_v.append(fpr95(vac, inc))
    out = {"acc": np.mean(acc), "ece_raw": np.mean(ece_raw), "ece_ts": np.mean(ece_ts),
           "au_e": np.nanmean(au_e), "fp_e": np.nanmean(fp_e)}
    out["au_v"] = np.nanmean(au_v) if au_v else None
    out["fp_v"] = np.nanmean(fp_v) if fp_v else None
    return out


COLS = [("Acc ↑", "acc", True), ("ECE raw ↓", "ece_raw", False), ("ECE +TS ↓", "ece_ts", False),
        ("AUROC-H entropy ↑", "au_e", True), ("FPR95-H entropy ↓", "fp_e", False),
        ("AUROC-H vacuity ↑", "au_v", True), ("FPR95-H vacuity ↓", "fp_v", False)]


def table(rows, shot):
    best = {}
    for _, key, hi in COLS:
        vals = [r[1][key] for r in rows if r[1].get(key) is not None]
        best[key] = (max if hi else min)(vals) if vals else None
    h = [f'<table><caption>{shot}-shot</caption><thead><tr><th>Method</th>'
         + "".join(f"<th>{c}</th>" for c, _, _ in COLS) + "</tr></thead><tbody>"]
    for label, m, cls in rows:
        tds = []
        for _, key, _ in COLS:
            v = m.get(key)
            if v is None:
                tds.append('<td>—</td>'); continue
            w = "win" if (best[key] is not None and abs(v - best[key]) < 1e-9) else (
                "num" if cls in ("", "ours") else "")
            tds.append(f'<td class="{w}">{v:.4f}</td>' if w else f'<td>{v:.4f}</td>')
        trc = f' class="{cls}"' if cls else ''
        h.append(f'<tr{trc}><td class="m">{label}</td>{"".join(tds)}</tr>')
    h.append("</tbody></table>")
    return "\n".join(h)


CSS = """body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:26px;background:#fafafa;color:#111}
h1{font-size:20px}caption{caption-side:top;text-align:left;font-weight:700;font-size:14px;padding:6px 0}
table{border-collapse:collapse;margin:10px 0 22px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.07)}
th,td{border:1px solid #ddd;padding:6px 10px;font-size:12.5px;text-align:right}
th{background:#111;color:#fff}th:first-child,td.m{text-align:left;white-space:nowrap}td.m{font-weight:650}
tr.ours td{background:#eef6ff}tr.ours td.m{font-weight:800}td.win{font-weight:800;background:#dff5e1;color:#0b4f16}
tr.single td{background:#fffbe9}tr.base td{background:#f3f0fa}p{font-size:13px;max-width:1000px}"""

INTRO = ("<p>DeepWeeds, mean over 20 runs. Temperature T fit per run on the validation split (NLL), "
         "applied to the test split — accuracy is unchanged by T. <b>ECE +TS</b> is the calibrated ECE "
         "(applied to every method, fairly). The common error-detection score is predictive <b>entropy</b> "
         "(all methods); the <b>vacuity</b> columns use the evidential score C/S, available only for "
         "Dirichlet methods (Sensoy, mean-evidence, Dempster). Green = best per column.</p>")


def section(methods, heading, note):
    out = [f"<h2>{heading}</h2>", note]
    has = False
    for shot in SHOTS:
        rows = []
        for label, cls, d, ev in methods:
            m = eval_method(d, ev, shot)
            if m is not None:
                rows.append((label, m, cls)); has = True
        out.append(table(rows, shot) if rows else f"<p><i>{shot}-shot: no logits exports found yet.</i></p>")
    return "\n".join(out), has


def main():
    html = [f'<!doctype html><html><head><meta charset="utf-8"><title>Calibration + native scores</title>'
            f'<style>{CSS}</style></head><body>',
            '<h1>Temperature scaling &amp; native uncertainty scores — DeepWeeds</h1>', INTRO]
    s1, h1 = section(ABLATION, "Calibration study &amp; combine-rule ablation",
                     "<p>Single backbones, vanilla deep ensemble, and our fusion under three combine "
                     "rules. Raw vs temperature-scaled (TS) calibration; vacuity columns apply only to "
                     "the evidential rows (mean-evidence, Dempster).</p>")
    s2, h2 = section(BASELINES, "Dedicated uncertainty baselines (separate comparison)",
                     "<p>Sensoy EDL and MC-Dropout, presented separately from the calibration ablation.</p>")
    html += [s1, s2, "</body></html>"]
    out = REPO / "docs" / "calibration_native_scores.html"
    out.write_text("\n".join(html))
    print(f"[{'done' if (h1 or h2) else 'EMPTY'}] wrote {out}")


if __name__ == "__main__":
    main()
