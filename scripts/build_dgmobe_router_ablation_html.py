#!/usr/bin/env python3
"""Build a standalone DG-MoBE router ablation HTML report for DeepWeeds."""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev, pvariance
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
KSHOT_ROOT = REPO_ROOT / "kshot"
OUTPUT_PATH = REPO_ROOT / "docs" / "dgmobe_router_ablation_deepweeds.html"
DATASET_KEY = "deepweeds_no_bg_mergeval"
STAGE_SOURCE = "img100_dirichlet_imagenet_1_acc85"
SHOTS = (1, 5, 10, 20)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool = True
    digits: int = 4


METRICS = (
    MetricSpec("accuracy", "Accuracy"),
    MetricSpec("precision", "Precision"),
    MetricSpec("recall", "Recall"),
    MetricSpec("macro_f1", "Macro-F1"),
    MetricSpec("auroc_err_pred_entropy", "AUROC-H"),
    MetricSpec("fpr95_err_pred_entropy", "FPR95-H", higher_is_better=False),
)

DIAGNOSTICS = (
    MetricSpec("dgmobe_router_entropy", "Router entropy", higher_is_better=False),
    MetricSpec("dgmobe_router_max", "Router max", higher_is_better=True),
    MetricSpec("dgmobe_router_l1_from_uniform", "Router L1 from uniform", higher_is_better=True),
    MetricSpec("dgmobe_expert_logit_cos", "Expert logit cosine", higher_is_better=False),
    MetricSpec("dgmobe_expert_logit_std", "Expert logit std", higher_is_better=True),
)


VARIANT_ORDER = (
    "results_dirichlet_logit_moe_headdiv",
    "results_dgmobe_uniform_router_deepweeds",
    "results_dgmobe_mixed_router_deepweeds/b025",
    "results_dgmobe_mixed_router_deepweeds/b050",
    "results_dgmobe_mixed_router_deepweeds/b075",
    "results_dgmobe_uncertainty_router_deepweeds/concentration_b050",
    "results_dgmobe_uncertainty_router_deepweeds/variance_b050",
    "results_dgmobe_support_ce_router_deepweeds/support_ce",
    "results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b025",
    "results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b050",
    "results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b075",
    "results_dgmobe_class_stacking_router_deepweeds/kl1e-4",
    "results_dgmobe_class_stacking_router_deepweeds/kl1e-3",
    "results_dgmobe_class_stacking_router_deepweeds/kl1e-2",
    "results_dgmobe_dempster_fusion_deepweeds/anchored_edl0p5_kl1",
    "results_dgmobe_dempster_fusion_deepweeds/pure_kl0p1_fv",
    "results_dgmobe_dempster_fusion_deepweeds/pure_kl1_fv",
    "results_dgmobe_dempster_fusion_deepweeds/pure_kl1_nofv",
)

ALLOWED_VARIANTS = set(VARIANT_ORDER)


DISPLAY_NAMES = {
    "results_dirichlet_logit_moe_headdiv": "DG-MoBE original Dirichlet router",
    "results_dgmobe_uniform_router_deepweeds": "Uniform router",
    "results_dgmobe_mixed_router_deepweeds/b025": "Mixed router beta=0.25",
    "results_dgmobe_mixed_router_deepweeds/b050": "Mixed router beta=0.50",
    "results_dgmobe_mixed_router_deepweeds/b075": "Mixed router beta=0.75",
    "results_dgmobe_uncertainty_router_deepweeds/concentration_b050": "Mean + concentration router beta=0.50",
    "results_dgmobe_uncertainty_router_deepweeds/variance_b050": "Mean + variance router beta=0.50",
    "results_dgmobe_support_ce_router_deepweeds/support_ce": "Support-CE router",
    "results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b025": "Support-CE mixed beta=0.25",
    "results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b050": "Support-CE mixed beta=0.50",
    "results_dgmobe_support_ce_router_deepweeds/support_ce_mixed_b075": "Support-CE mixed beta=0.75",
    "results_dgmobe_class_stacking_router_deepweeds/kl1e-4": "Class-stacking router KL=1e-4",
    "results_dgmobe_class_stacking_router_deepweeds/kl1e-3": "Class-stacking router KL=1e-3",
    "results_dgmobe_class_stacking_router_deepweeds/kl1e-2": "Class-stacking router KL=1e-2",
    "results_dgmobe_dempster_fusion_deepweeds/anchored_edl0p5_kl1": "Dempster fusion anchored EDL=0.5 KL=1",
    "results_dgmobe_dempster_fusion_deepweeds/pure_kl0p1_fv": "Dempster fusion pure KL=0.1 FV",
    "results_dgmobe_dempster_fusion_deepweeds/pure_kl1_fv": "Dempster fusion pure KL=1 FV",
    "results_dgmobe_dempster_fusion_deepweeds/pure_kl1_nofv": "Dempster fusion pure KL=1 no-FV",
}

EXPECTED_ROOTS = (
    "results_dgmobe_uniform_router_deepweeds",
    "results_dgmobe_mixed_router_deepweeds",
    "results_dgmobe_uncertainty_router_deepweeds",
    "results_dgmobe_support_ce_router_deepweeds",
    "results_dgmobe_class_stacking_router_deepweeds",
    "results_dgmobe_dempster_fusion_deepweeds",
)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def variant_key_for_summary(path: Path) -> str | None:
    try:
        parts = path.relative_to(KSHOT_ROOT).parts
    except ValueError:
        return None
    if STAGE_SOURCE not in parts:
        return None
    stage_idx = parts.index(STAGE_SOURCE)
    if stage_idx < 1:
        return None
    if len(parts) <= stage_idx + 2 or parts[stage_idx + 1] != DATASET_KEY:
        return None
    return "/".join(parts[:stage_idx])


def read_summary(path: Path) -> dict[str, float | int | None]:
    payload = json.loads(path.read_text())
    summary = payload.get("summary", {})
    row: dict[str, float | int | None] = {
        "runs": safe_float(summary.get("num_runs")),
    }
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "auroc_err_pred_entropy",
        "fpr95_err_pred_entropy",
        "dgmobe_router_entropy",
        "dgmobe_router_max",
        "dgmobe_router_l1_from_uniform",
        "dgmobe_expert_logit_cos",
        "dgmobe_expert_logit_std",
    ):
        row[metric] = safe_float(summary.get(f"{metric}_mean"))
        row[f"{metric}_std"] = safe_float(summary.get(f"{metric}_std"))
    precision = safe_float(row.get("precision"))
    recall = safe_float(row.get("recall"))
    row["macro_f1"] = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return row


def aggregate_from_runs(results_dir: Path, shot: int) -> dict[str, float | int | None] | None:
    run_paths = sorted(results_dir.glob(f"{shot}_shot_run_*.json"))
    rows: list[dict[str, float]] = []
    for path in run_paths:
        payload = json.loads(path.read_text())
        row: dict[str, float] = {}
        for key in (
            "test_accuracy",
            "test_precision",
            "test_recall",
            "auroc_err_pred_entropy",
            "fpr95_err_pred_entropy",
            "dgmobe_router_entropy_mean",
            "dgmobe_router_max_mean",
            "dgmobe_router_l1_from_uniform_mean",
            "dgmobe_expert_logit_cos_mean",
            "dgmobe_expert_logit_std_mean",
        ):
            value = safe_float(payload.get(key))
            if value is not None:
                row[key] = value
        if row:
            rows.append(row)
    if not rows:
        return None

    mapping = {
        "test_accuracy": "accuracy",
        "test_precision": "precision",
        "test_recall": "recall",
        "auroc_err_pred_entropy": "auroc_err_pred_entropy",
        "fpr95_err_pred_entropy": "fpr95_err_pred_entropy",
        "dgmobe_router_entropy_mean": "dgmobe_router_entropy",
        "dgmobe_router_max_mean": "dgmobe_router_max",
        "dgmobe_router_l1_from_uniform_mean": "dgmobe_router_l1_from_uniform",
        "dgmobe_expert_logit_cos_mean": "dgmobe_expert_logit_cos",
        "dgmobe_expert_logit_std_mean": "dgmobe_expert_logit_std",
    }
    out: dict[str, float | int | None] = {"runs": float(len(rows))}
    for source, target in mapping.items():
        vals = [row[source] for row in rows if source in row]
        if vals:
            out[target] = mean(vals)
            out[f"{target}_std"] = pstdev(vals)
    precision = safe_float(out.get("precision"))
    recall = safe_float(out.get("recall"))
    out["macro_f1"] = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return out


def load_rows() -> dict[str, dict[int, dict[str, float | int | None]]]:
    rows: dict[str, dict[int, dict[str, float | int | None]]] = {}
    summary_paths = sorted(KSHOT_ROOT.rglob("*_shot_summary.json"))
    for path in summary_paths:
        variant_key = variant_key_for_summary(path)
        if variant_key is None:
            continue
        if variant_key not in ALLOWED_VARIANTS:
            continue
        try:
            shot = int(path.name.split("_shot_summary.json")[0])
        except ValueError:
            continue
        if shot not in SHOTS:
            continue
        rows.setdefault(variant_key, {})[shot] = read_summary(path)

    # Some older folders may have runs but no summary yet.
    for results_dir in sorted(KSHOT_ROOT.rglob(DATASET_KEY)):
        variant_root = variant_key_for_summary(results_dir / "1_shot_summary.json")
        if variant_root is None:
            continue
        if variant_root not in ALLOWED_VARIANTS:
            continue
        if variant_root in rows and all(shot in rows[variant_root] for shot in SHOTS):
            continue
        for shot in SHOTS:
            if shot in rows.get(variant_root, {}):
                continue
            aggregated = aggregate_from_runs(results_dir, shot)
            if aggregated is not None:
                rows.setdefault(variant_root, {})[shot] = aggregated
    return rows


def sort_variants(keys: list[str]) -> list[str]:
    order = {key: idx for idx, key in enumerate(VARIANT_ORDER)}
    return sorted(keys, key=lambda key: (order.get(key, 10_000), key))


def display_name(variant_key: str) -> str:
    if variant_key in DISPLAY_NAMES:
        return DISPLAY_NAMES[variant_key]
    label = variant_key.replace("results_dgmobe_", "").replace("_deepweeds", "")
    label = label.replace("_", " ").replace("/", " / ")
    return label.title()


def metric_winners(rows: dict[str, dict[int, dict[str, float | int | None]]], shot: int, specs) -> dict[str, set[str]]:
    winners: dict[str, set[str]] = {}
    variants = [key for key in rows if shot in rows[key]]
    for spec in specs:
        values = [
            (key, safe_float(rows[key][shot].get(spec.key)))
            for key in variants
        ]
        values = [(key, value) for key, value in values if value is not None]
        if not values:
            winners[spec.key] = set()
            continue
        best = max(value for _, value in values) if spec.higher_is_better else min(value for _, value in values)
        winners[spec.key] = {key for key, value in values if abs(value - best) <= 1e-12}
    return winners


def table_for_shot(rows: dict[str, dict[int, dict[str, float | int | None]]], shot: int, specs) -> str:
    variants = [key for key in sort_variants(list(rows)) if shot in rows[key]]
    winners = metric_winners(rows, shot, specs)
    headers = ["Variant", "Runs", *[spec.label for spec in specs]]
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    out.append("</tr></thead><tbody>")
    for variant in variants:
        row = rows[variant][shot]
        out.append("<tr>")
        out.append(f"<td class=\"model\">{html.escape(display_name(variant))}</td>")
        out.append(f"<td>{fmt(safe_float(row.get('runs')), 0)}</td>")
        for spec in specs:
            value = safe_float(row.get(spec.key))
            text = fmt(value, spec.digits)
            cls = "best" if variant in winners.get(spec.key, set()) and value is not None else ""
            out.append(f"<td class=\"num {cls}\">{text}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def best_summary(rows: dict[str, dict[int, dict[str, float | int | None]]]) -> str:
    cards: list[str] = []
    for shot in SHOTS:
        if not any(shot in shot_rows for shot_rows in rows.values()):
            continue
        winners = metric_winners(rows, shot, METRICS)
        acc_winners = ", ".join(display_name(key) for key in sort_variants(list(winners.get("accuracy", set()))))
        f1_winners = ", ".join(display_name(key) for key in sort_variants(list(winners.get("macro_f1", set()))))
        cards.append(
            f"<div class=\"card\"><span>{shot}-shot</span>"
            f"<strong>Accuracy:</strong> {html.escape(acc_winners or '-')}"
            f"<br><strong>Macro-F1:</strong> {html.escape(f1_winners or '-')}</div>"
        )
    return "\n".join(cards)


def missing_roots(rows: dict[str, dict[int, dict[str, float | int | None]]]) -> list[str]:
    present = set(rows)
    missing = []
    for root in EXPECTED_ROOTS:
        if not any(key == root or key.startswith(f"{root}/") for key in present):
            missing.append(root)
    return missing


def build_html(rows: dict[str, dict[int, dict[str, float | int | None]]]) -> str:
    missing = missing_roots(rows)
    available = sort_variants(list(rows))
    missing_html = ""
    if missing:
        items = "".join(f"<li><code>kshot/{html.escape(root)}</code></li>" for root in missing)
        missing_html = (
            "<section class=\"warning\"><h2>Missing ablation folders</h2>"
            "<p>The report was generated, but these expected folders are not available locally yet:</p>"
            f"<ul>{items}</ul>"
            "<p>Copy them into <code>kshot/</code> and rerun "
            "<code>python scripts/build_dgmobe_router_ablation_html.py</code>.</p></section>"
        )

    available_html = "".join(
        f"<li><code>kshot/{html.escape(key)}</code> <span>{html.escape(display_name(key))}</span></li>"
        for key in available
    ) or "<li>No matching result folders found.</li>"

    performance_tables = "\n".join(
        f"<section><h2>{shot}-shot performance</h2>{table_for_shot(rows, shot, METRICS)}</section>"
        for shot in SHOTS
        if any(shot in shot_rows for shot_rows in rows.values())
    )
    diagnostic_tables = "\n".join(
        f"<section><h2>{shot}-shot DG-MoBE diagnostics</h2>{table_for_shot(rows, shot, DIAGNOSTICS)}</section>"
        for shot in SHOTS
        if any(shot in shot_rows for shot_rows in rows.values())
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DG-MoBE Router Ablation - DeepWeeds</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --paper: #fffdf8;
      --ink: #1f2933;
      --muted: #6b7280;
      --line: #ddd3c2;
      --green: #0f766e;
      --blue: #1d4ed8;
      --orange: #b45309;
      --red: #b91c1c;
      --best: #fff0b3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.13), transparent 34rem),
        radial-gradient(circle at 82% 8%, rgba(180, 83, 9, 0.14), transparent 30rem),
        var(--bg);
      color: var(--ink);
      font-family: ui-serif, Georgia, "Times New Roman", serif;
      line-height: 1.45;
    }}
    main {{ width: min(1320px, calc(100vw - 36px)); margin: 0 auto; padding: 36px 0 64px; }}
    header {{
      background: linear-gradient(135deg, rgba(255,253,248,0.96), rgba(255,250,235,0.92));
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 28px 32px;
      box-shadow: 0 18px 42px rgba(86, 64, 37, 0.12);
    }}
    h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 48px); line-height: 1.05; letter-spacing: -0.035em; }}
    h2 {{ margin: 0 0 14px; font-size: 23px; letter-spacing: -0.015em; }}
    p {{ margin: 8px 0; }}
    code {{ background: #f2eadb; border: 1px solid #e2d5be; border-radius: 6px; padding: 1px 5px; }}
    .subtitle {{ max-width: 900px; color: var(--muted); font-size: 17px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-top: 18px; }}
    .card {{
      background: #0f172a;
      color: #f8fafc;
      border-radius: 16px;
      padding: 14px 16px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 14px;
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.16);
    }}
    .card span {{ display: block; color: #fbbf24; font-weight: 800; margin-bottom: 6px; }}
    section {{
      margin-top: 26px;
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 14px 30px rgba(86, 64, 37, 0.08);
      overflow-x: auto;
    }}
    .warning {{ border-color: #f59e0b; background: #fffbeb; }}
    .note {{ color: var(--muted); font-size: 14px; }}
    ul {{ margin: 8px 0 0; }}
    li span {{ color: var(--muted); margin-left: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 840px; font-family: ui-sans-serif, system-ui, sans-serif; }}
    th {{
      background: #1f2937;
      color: white;
      padding: 10px 9px;
      text-align: right;
      font-size: 13px;
      border: 1px solid #374151;
      position: sticky;
      top: 0;
    }}
    th:first-child {{ text-align: left; }}
    td {{
      padding: 9px;
      border: 1px solid #e5e0d5;
      text-align: right;
      font-variant-numeric: tabular-nums;
      background: rgba(255, 255, 255, 0.72);
    }}
    tr:nth-child(even) td {{ background: rgba(248, 245, 238, 0.78); }}
    td.model {{ text-align: left; font-weight: 700; min-width: 280px; }}
    td.best {{ background: var(--best) !important; color: #78350f; font-weight: 900; }}
    footer {{ color: var(--muted); margin-top: 28px; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>DG-MoBE Router Ablation: DeepWeeds</h1>
    <p class="subtitle">
      Separate ablation report for uniform, mixed, support-calibrated, class-stacking, Dempster fusion, concentration-confidence, and variance router variants.
      The original DG-MoBE head-diversity run is included when available as context.
    </p>
    <p class="note">Macro-F1 is computed from the reported macro precision and macro recall means: 2PR/(P+R). Bold values mark the best available variant per shot and metric.</p>
    <div class="grid">{best_summary(rows)}</div>
  </header>
  {missing_html}
  <section>
    <h2>Loaded variants</h2>
    <ul>{available_html}</ul>
  </section>
  {performance_tables or '<section><h2>No performance tables</h2><p>No matching summaries were found.</p></section>'}
  {diagnostic_tables or ''}
  <footer>
    Generated from <code>kshot/</code>. Rebuild with <code>python scripts/build_dgmobe_router_ablation_html.py</code>.
  </footer>
</main>
</body>
</html>
"""


def main() -> None:
    rows = load_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_html(rows))
    print(f"Wrote {OUTPUT_PATH}")
    if missing := missing_roots(rows):
        print("Missing expected folders:")
        for root in missing:
            print(f"  kshot/{root}")


if __name__ == "__main__":
    main()
