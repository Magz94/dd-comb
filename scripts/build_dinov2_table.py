#!/usr/bin/env python3
"""Regenerate docs/dinov2_vs_ddcobe.html with the full DeepWeeds comparison
(singles, single-backbone uncertainty baselines, and our DD-CoBE fusions),
pulling fusion/baseline numbers from the k-shot result JSONs and auto-marking the
best value per column. Plant Seedlings V2 section is kept as previously reported.
"""
from __future__ import annotations
import json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHOTS = [1, 5, 10, 20]


def jget(path):
    """Return dict metric->[4 shots] from k-shot summaries, or None if absent."""
    out = {"acc": [], "f1": [], "ece": [], "auroc": [], "fpr95": []}
    for s in SHOTS:
        fp = REPO / path / f"{s}_shot_summary.json"
        if not fp.is_file():
            return None
        sm = json.loads(fp.read_text())["summary"]
        P, R = sm["precision_mean"], sm["recall_mean"]
        out["acc"].append(sm["accuracy_mean"])
        out["f1"].append(2 * P * R / (P + R) if (P + R) else 0.0)
        out["ece"].append(sm["ece_mean"])
        out["auroc"].append(sm.get("auroc_err_pred_entropy_mean", 0.0))
        out["fpr95"].append(sm.get("fpr95_err_pred_entropy_mean", 0.0))
    return out


def lit(acc, f1, ece, auroc, fpr95):
    return {"acc": acc, "f1": f1, "ece": ece, "auroc": auroc, "fpr95": fpr95}

# ---- DeepWeeds rows -------------------------------------------------------
DW = []
DW.append(("DINOv2 (single, linear probe)", "", lit(
    [0.4953, 0.7326, 0.8004, 0.8657], [0.5145, 0.7390, 0.8091, 0.8675],
    [0.0918, 0.2307, 0.1581, 0.1147], [0.6994, 0.8135, 0.8395, 0.8749],
    [0.7409, 0.5370, 0.4742, 0.3751])))
DW.append(("ConvNeXt-V2-Tiny (single, linear probe)", "single", lit(
    [0.3255, 0.5576, 0.6631, 0.7635], [0.3452, 0.5656, 0.6715, 0.7668],
    [0.0652, 0.1719, 0.1746, 0.1529], [0.6424, 0.7129, 0.7515, 0.7940],
    [0.8275, 0.7584, 0.6835, 0.5833])))
DW.append(("SigLIP2-B (single, linear probe)", "single", lit(
    [0.3825, 0.5747, 0.6566, 0.7564], [0.3941, 0.5932, 0.6758, 0.7616],
    [0.1287, 0.3004, 0.3036, 0.2880], [0.6434, 0.6488, 0.7138, 0.7446],
    [0.8532, 0.8683, 0.8036, 0.7236])))
# single-backbone uncertainty baselines (from JSON)
BASE = [
    ("DINOv2 — MC Dropout (baseline)", "base", "kshot/results_modern_baselines/mc_dropout/dinov2_vitb14/deepweeds_no_bg_mergeval"),
    ("DINOv2 — Sensoy EDL (baseline)", "base", "kshot/results_modern_baselines/sensoy/dinov2_vitb14/deepweeds_no_bg_mergeval"),
    ("ConvNeXt-V2-Tiny — MC Dropout (baseline)", "base", "kshot/results_modern_baselines/mc_dropout/convnextv2_tiny/deepweeds_no_bg_mergeval"),
    ("ConvNeXt-V2-Tiny — Sensoy EDL (baseline)", "base", "kshot/results_modern_baselines/sensoy/convnextv2_tiny/deepweeds_no_bg_mergeval"),
]
for label, cls, path in BASE:
    m = jget(path)
    if m:
        DW.append((label, cls, m))
# our fusions of the pair (from JSON)
FUS = [
    ("DD-CoBE: DINOv2 + ConvNeXt-V2-Tiny (Dempster, ours)", "ours", "kshot/results_dino_pairs_mlp/dino_cnx_dempster/deepweeds_no_bg_mergeval"),
    ("DD-CoBE: DINOv2 + ConvNeXt-V2-Tiny (softmax ensemble)", "ours", "kshot/results_dino_pairs_mlp/dino_cnx_softmax_ensemble/deepweeds_no_bg_mergeval"),
    ("DD-CoBE: DINOv2 + ConvNeXt-V2-Tiny (mean-evidence)", "ours", "kshot/results_dino_pairs_mlp/dino_cnx_mean_evidence/deepweeds_no_bg_mergeval"),
]
for label, cls, path in FUS:
    m = jget(path)
    if m:
        DW.append((label, cls, m))
    else:
        print(f"[warn] missing fusion results: {path}")
# 5-backbone DD-CoBE (previously reported)
DW.append(("DD-CoBE (ours, 5 original backbones)", "ours", lit(
    [0.3553, 0.5540, 0.6544, 0.7385], [0.3600, 0.5619, 0.6586, 0.7407],
    [0.2011, 0.0582, 0.0376, 0.0324], [0.6635, 0.7307, 0.7567, 0.7961],
    [0.7995, 0.7183, 0.6913, 0.6181])))

# ---- Plant Seedlings V2 rows (as previously reported) --------------------
PS = []
PS.append(("DINOv2 (single, linear probe)", "", lit(
    [0.4321, 0.6287, 0.7430, 0.8389], [0.4669, 0.6390, 0.7410, 0.8333],
    [0.0832, 0.2374, 0.2072, 0.1652], [0.7036, 0.7198, 0.7497, 0.7871],
    [0.7511, 0.6866, 0.6543, 0.5759])))
PS.append(("DD-CoBE (ours, 5 original backbones)", "ours", lit(
    [0.2316, 0.4576, 0.5855, 0.6834], [0.2540, 0.4666, 0.5893, 0.6898],
    [0.2193, 0.0682, 0.1096, 0.1258], [0.6233, 0.6810, 0.7219, 0.7412],
    [0.8891, 0.8123, 0.7609, 0.7283])))

METRICS = [("Accuracy ↑", "acc", True), ("Macro-F1 ↑", "f1", True),
           ("ECE (15-bin) ↓", "ece", False), ("AUROC-H ↑", "auroc", True),
           ("FPR95-H ↓", "fpr95", False)]


def table(rows, cap, key, higher):
    best = [None] * 4
    for j in range(4):
        vals = [r[2][key][j] for r in rows]
        best[j] = (max if higher else min)(vals)
    h = [f'<table><caption>{cap}</caption><thead><tr><th>Model</th>'
         '<th>1-shot</th><th>5-shot</th><th>10-shot</th><th>20-shot</th></tr></thead><tbody>']
    for label, cls, m in rows:
        trc = f' class="{cls}"' if cls else ''
        cells = []
        for j in range(4):
            v = m[key][j]
            c = "win" if abs(v - best[j]) < 1e-9 else ("num" if cls in ("", "ours") else "")
            cells.append(f'<td class="{c}">{v:.4f}</td>' if c else f'<td>{v:.4f}</td>')
        h.append(f'<tr{trc}><td class="m">{label}</td>{"".join(cells)}</tr>')
    h.append('</tbody></table>')
    return "\n".join(h)


CSS = """body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:28px;background:#fafafa;color:#111}
h1{font-size:21px}h2{font-size:17px;margin-top:26px;border-bottom:2px solid #111;padding-bottom:3px}
table{border-collapse:collapse;margin:12px 0 22px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.07)}
caption{caption-side:top;text-align:left;font-weight:700;font-size:14px;padding:6px 0}
th,td{border:1px solid #ddd;padding:6px 11px;font-size:13px;text-align:right}
th{background:#111;color:#fff}th:first-child,td.m{text-align:left;white-space:nowrap}td.m{font-weight:650}
tr.ours td{background:#eef6ff}tr.ours td.m{font-weight:800}td.win{font-weight:800;background:#dff5e1;color:#0b4f16}
tr.single td{background:#fffbe9}tr.base td{background:#f3f0fa}"""

INTRO = ("<p>DeepWeeds: frozen backbones, few-shot (mean over 20 runs). "
         "Compares single foundation backbones (DINOv2 ViT-B/14, ConvNeXt-V2-Tiny, SigLIP2-B), "
         "their standard single-backbone uncertainty baselines (MC Dropout, Sensoy EDL; mlp head "
         "for MC Dropout at native width, linear head for Sensoy), and our DD-CoBE fusion of "
         "DINOv2 + ConvNeXt-V2-Tiny (mlp head, native widths) under three combine rules. "
         "Green = best per column; higher better for Accuracy/Macro-F1/AUROC-H, lower for ECE/FPR95-H. "
         "Blue = our fusions, yellow = extra single backbones, purple = single-backbone baselines. "
         "Plant Seedlings V2 shows the 5-backbone DD-CoBE vs DINOv2 as previously reported.</p>")

def build(out_path, dw_rows, intro):
    html = [f'<!doctype html><html><head><meta charset="utf-8"><title>DINOv2 vs DD-CoBE</title><style>{CSS}</style></head><body>',
            '<h1>DINOv2 vs DD-CoBE — single backbones, uncertainty baselines, and fusion</h1>',
            intro, '<h2>DeepWeeds</h2>']
    for cap, key, higher in METRICS:
        html.append(table(dw_rows, cap, key, higher))
    html.append('<h2>Plant Seedlings V2</h2>')
    for cap, key, higher in METRICS:
        html.append(table(PS, cap, key, higher))
    html.append('</body></html>')
    out_path.write_text("\n".join(html))
    print(f"[done] wrote {out_path}  ({len(dw_rows)} DeepWeeds rows)")


# Full version (with MC Dropout)
build(REPO / "docs" / "dinov2_vs_ddcobe.html", DW, INTRO)

# Variant WITHOUT MC Dropout rows
DW_no_mcd = [r for r in DW if "MC Dropout" not in r[0]]
INTRO_no_mcd = INTRO.replace("MC Dropout, Sensoy EDL; mlp head "
                             "for MC Dropout at native width, linear head for Sensoy",
                             "Sensoy EDL; linear head")
build(REPO / "docs" / "dinov2_vs_ddcobe_no_mcdropout.html", DW_no_mcd, INTRO_no_mcd)
