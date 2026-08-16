#!/usr/bin/env python3
"""Regenerate docs/dgmobe_router_ablation_deepweeds.html with three sections:

1. The original DG-MoBE router ablation on DeepWeeds (all router families).
2. DD-CoBE (Dempster-Dirichlet Combination of Backbone Experts) arms on
   DeepWeeds, paired against the uniform-router DG-MoBE baseline.
3. DD-CoBE across bloodmnist / plant_seedlings_v2 / ssl_patches, paired
   against the Dirichlet-router DG-MoBE (headdiv) baseline.

Pure stdlib; reads the per-run k-shot JSONs directly.
"""
from __future__ import annotations

import glob
import html
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[1]
KSHOT = REPO_ROOT / "kshot"
OUTPUT = REPO_ROOT / "docs" / "dgmobe_router_ablation_deepweeds.html"
SHOTS = (1, 5, 10, 20)
DW = "deepweeds_no_bg_mergeval"

METRICS = (
    ("test_accuracy", "Accuracy", +1),
    ("test_ece", "ECE", -1),
    ("auroc_err_pred_entropy", "AUROC-H", +1),
    ("fpr95_err_pred_entropy", "FPR95-H", -1),
)

ROUTER_VARIANTS = [
    ("results_dirichlet_logit_moe_headdiv", "DG-MoBE Dirichlet router (original)"),
    ("results_dgmobe_uniform_router_deepweeds", "DG-MoBE uniform router"),
    ("results_dgmobe_mixed_router_deepweeds/b025", "Mixed router beta=0.25"),
    ("results_dgmobe_mixed_router_deepweeds/b050", "Mixed router beta=0.50"),
    ("results_dgmobe_mixed_router_deepweeds/b075", "Mixed router beta=0.75"),
    ("results_dgmobe_uncertainty_router_deepweeds/concentration_b050", "Mean + concentration router b=0.50"),
    ("results_dgmobe_uncertainty_router_deepweeds/variance_b050", "Mean + variance router b=0.50"),
    ("results_dgmobe_support_ce_router_deepweeds/support_ce", "Support-CE router"),
    ("results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b025", "Support-CE mixed b=0.25"),
    ("results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b050", "Support-CE mixed b=0.50"),
    ("results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b075", "Support-CE mixed b=0.75"),
    ("results_dgmobe_class_stacking_router_deepweeds/kl1e-4", "Class-stacking router KL=1e-4"),
    ("results_dgmobe_class_stacking_router_deepweeds/kl1e-3", "Class-stacking router KL=1e-3"),
    ("results_dgmobe_class_stacking_router_deepweeds/kl1e-2", "Class-stacking router KL=1e-2"),
]

DDCOBE_DW_VARIANTS = [
    ("results_dgmobe_dempster_fusion_deepweeds/pure_kl0p1_fv", "DD-CoBE pure KL=0.1 (fused view)"),
    ("results_dgmobe_dempster_fusion_deepweeds/pure_kl1_fv", "DD-CoBE pure KL=1.0 (fused view)"),
    ("results_dgmobe_dempster_fusion_deepweeds/pure_kl1_nofv", "DD-CoBE pure KL=1.0 (no fused view)"),
    ("results_dgmobe_dempster_fusion_deepweeds/anchored_edl0p5_kl1", "DD-CoBE CE-anchored edl=0.5 KL=1.0"),
    ("results_dgmobe_dempster_fusion_deepweeds/mean_ev_kl1_fv", "Mean-evidence control KL=1.0 (fused view)"),
]

MULTI_DATASETS = [
    ("bloodmnist_224_imagefolder_mergeval", "BloodMNIST"),
    ("plant_seedlings_v2_split", "Plant Seedlings V2"),
    ("ssl_patches", "SSL Patches (plant vs meadow)"),
]
MULTI_ARMS = [
    ("results_ddcobe_dempster_multidataset/pure_kl0p1_fv", "DD-CoBE pure KL=0.1 (fused view)"),
    ("results_ddcobe_dempster_multidataset/pure_kl0p1_nofv", "DD-CoBE pure KL=0.1 (no fused view)"),
    ("results_ddcobe_dempster_multidataset/mean_ev_kl0p1_fv", "Mean-evidence control KL=0.1 (fused view)"),
]


def load_runs(rel_root: str, dataset: str) -> dict:
    out = {}
    pattern = str(KSHOT / rel_root / "**" / dataset / "*_shot_run_*.json")
    for p in glob.glob(pattern, recursive=True):
        d = json.load(open(p))
        out[(int(d["shot"]), int(d["run_index"]))] = d
    return out


def fmt(v, digits=4):
    return f"{v:.{digits}f}" if v is not None and math.isfinite(v) else "-"


def cell_mean_std(runs, shot, key):
    vals = [r[key] for (s, _), r in runs.items() if s == shot and key in r]
    if not vals:
        return "-"
    return f"{fmt(mean(vals))} ± {fmt(pstdev(vals))}"


def cell_paired(runs, base, shot, key, better):
    keys = [k for k in runs if k[0] == shot and k in base and key in runs[k] and key in base[k]]
    if len(keys) < 2:
        return "-", ""
    deltas = [runs[k][key] - base[k][key] for k in keys]
    n = len(deltas)
    dm = mean(deltas)
    dsd = pstdev(deltas)
    t = dm / (dsd / math.sqrt(n)) if dsd > 0 else float("inf")
    wins = sum(1 for x in deltas if x * better > 0)
    cls = ""
    if abs(t) >= 2.1:  # ~p<0.05 paired, n=20
        cls = "good" if dm * better > 0 else "bad"
    return f"{dm:+.4f} (t={t:+.1f}, {wins}/{n})", cls


def table_means(variants, dataset):
    head = "<tr><th>Variant</th>" + "".join(
        f"<th>{label}<br>{shot}-shot</th>" for key, label, _ in METRICS for shot in SHOTS
    ) + "</tr>"
    rows = []
    for rel, label in variants:
        runs = load_runs(rel, dataset)
        if not runs:
            continue
        cells = "".join(
            f"<td>{cell_mean_std(runs, shot, key)}</td>"
            for key, _, _ in METRICS for shot in SHOTS
        )
        rows.append(f"<tr><td class='name'>{html.escape(label)}</td>{cells}</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def table_paired(variants, baseline_rel, baseline_label, dataset):
    base = load_runs(baseline_rel, dataset)
    head = (
        "<tr><th rowspan='2'>Variant</th>"
        + "".join(f"<th colspan='4'>{label}</th>" for _, label, _ in METRICS)
        + "</tr><tr>"
        + "".join(f"<th>{s}s</th>" for _ in METRICS for s in SHOTS)
        + "</tr>"
    )
    rows = []
    # baseline reference row (means)
    cells = "".join(
        f"<td>{cell_mean_std(base, shot, key)}</td>" for key, _, _ in METRICS for shot in SHOTS
    )
    rows.append(f"<tr class='base'><td class='name'>{html.escape(baseline_label)} (reference, mean ± std)</td>{cells}</tr>")
    for rel, label in variants:
        runs = load_runs(rel, dataset)
        if not runs:
            continue
        mean_cells = "".join(
            f"<td>{cell_mean_std(runs, shot, key)}</td>" for key, _, _ in METRICS for shot in SHOTS
        )
        rows.append(f"<tr><td class='name'>{html.escape(label)} (mean ± std)</td>{mean_cells}</tr>")
        d_cells = []
        for key, _, better in METRICS:
            for shot in SHOTS:
                txt, cls = cell_paired(runs, base, shot, key, better)
                d_cells.append(f"<td class='{cls}'>{txt}</td>")
        rows.append(
            f"<tr class='delta'><td class='name'>&nbsp;&nbsp;paired &Delta; vs baseline</td>{''.join(d_cells)}</tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def main():
    parts = []
    parts.append("""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>DG-MoBE router ablation &amp; DD-CoBE report</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#1a1a1a}
h1{font-size:22px} h2{font-size:18px;margin-top:36px} h3{font-size:15px;margin-top:24px}
p.note{color:#555;max-width:1000px}
table{border-collapse:collapse;font-size:11px;margin:12px 0}
th,td{border:1px solid #bbb;padding:3px 6px;text-align:center}
th{background:#f0f0f0}
td.name{text-align:left;white-space:nowrap;font-weight:bold}
tr.base{background:#eef4ff}
tr.delta td{color:#333;font-size:10px}
td.good{background:#d9f2d9;font-weight:bold}
td.bad{background:#f8d7da;font-weight:bold}
</style></head><body>
<h1>DG-MoBE router ablation &amp; DD-CoBE (Dempster&ndash;Dirichlet Combination of Backbone Experts)</h1>
<p class='note'>All runs: 20 seeds per (dataset, shot) on shared k-shot manifests; paired
&Delta; rows compare run-by-run against the reference (green/red = paired |t| &ge; 2.1, i.e. ~p&lt;0.05).
Stage-A source: img100_dirichlet_imagenet_1_acc85. Metrics: Accuracy, ECE (15-bin, from the
predictive distribution), AUROC-H / FPR95-H (misclassification detection via predictive entropy).</p>
""")

    parts.append("<h2>1. DG-MoBE router ablation &mdash; DeepWeeds</h2>")
    parts.append("<p class='note'>All router families tie with the uniform router (within noise): "
                 "per-sample routing signals are not learnable under few-shot supervision.</p>")
    parts.append(table_means(ROUTER_VARIANTS, DW))

    parts.append("<h2>2. DD-CoBE on DeepWeeds &mdash; paired vs uniform-router DG-MoBE</h2>")
    parts.append("<p class='note'>DD-CoBE replaces the routed logit mixture with a reduced "
                 "Dempster&ndash;Shafer combination of per-expert Dirichlet opinions "
                 "(TMC-style, Han et al. ICLR 2021), trained with expected-CE + annealed "
                 "wrong-evidence KL. No router parameters.</p>")
    parts.append(table_paired(DDCOBE_DW_VARIANTS,
                              "results_dgmobe_uniform_router_deepweeds",
                              "DG-MoBE uniform router", DW))

    parts.append("<h2>3. DD-CoBE across datasets &mdash; paired vs Dirichlet-router DG-MoBE (headdiv)</h2>")
    parts.append("<p class='note'>Baseline here is the original Dirichlet-router DG-MoBE "
                 "(the uniform-router baseline has not yet been run for these datasets; on "
                 "DeepWeeds the two are statistically indistinguishable). The mean-evidence "
                 "control shares opinions, losses and views with DD-CoBE and differs only in "
                 "the combination rule &mdash; it isolates the contribution of the Dempster rule.</p>")
    for ds_key, ds_label in MULTI_DATASETS:
        parts.append(f"<h3>{html.escape(ds_label)}</h3>")
        parts.append(table_paired(MULTI_ARMS,
                                  "results_dirichlet_logit_moe_headdiv",
                                  "DG-MoBE Dirichlet router", ds_key))

    parts.append("</body></html>")
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
