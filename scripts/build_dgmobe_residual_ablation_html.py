#!/usr/bin/env python3
"""Build a standalone DG-MoBE residual-fusion ablation HTML report for DeepWeeds."""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
KSHOT_ROOT = REPO_ROOT / "kshot"
OUTPUT_PATH = REPO_ROOT / "docs" / "dgmobe_residual_fusion_ablation_deepweeds.html"
DATASET_KEY = "deepweeds_no_bg_mergeval"
STAGE_SOURCE = "img100_dirichlet_imagenet_1_acc85"
ROOT = "results_dgmobe_residual_fusion_ablation_deepweeds"
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

VARIANTS = (
    ("fused_only", "Fused-only final logits"),
    ("moe_only", "MoE-only final logits"),
    ("residual_g0p5", "Residual fusion gamma=0.5"),
    ("residual_g1p0", "Residual fusion gamma=1.0"),
)


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def load_summary(label: str, shot: int) -> dict[str, Any] | None:
    path = KSHOT_ROOT / ROOT / label / STAGE_SOURCE / DATASET_KEY / f"{shot}_shot_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_value(summary: dict[str, Any] | None, metric: MetricSpec) -> float | None:
    if summary is None:
        return None
    values = summary.get("summary", {})
    value = safe_float(values.get(f"{metric.key}_mean"))
    if value is not None:
        return value
    if metric.key == "macro_f1":
        precision = safe_float(values.get("precision_mean"))
        recall = safe_float(values.get("recall_mean"))
        if precision is not None and recall is not None and precision + recall > 0.0:
            return 2.0 * precision * recall / (precision + recall)
    return None


def best_by_metric(rows: list[dict[str, Any]], metric: MetricSpec) -> float | None:
    vals = [row[metric.key] for row in rows if row[metric.key] is not None]
    if not vals:
        return None
    return max(vals) if metric.higher_is_better else min(vals)


def render_table(shot: int, rows: list[dict[str, Any]]) -> str:
    best = {metric.key: best_by_metric(rows, metric) for metric in METRICS}
    header = "".join(f"<th>{html.escape(metric.label)}</th>" for metric in METRICS)
    body = []
    for row in rows:
        cells = [
            f"<td class='model'>{html.escape(row['name'])}</td>",
            f"<td>{int(row['runs']) if row['runs'] is not None else '-'}</td>",
        ]
        for metric in METRICS:
            value = row[metric.key]
            value_text = fmt(value, metric.digits)
            cls = ""
            if value is not None and best[metric.key] is not None and abs(value - best[metric.key]) < 1e-12:
                cls = " class='best'"
            cells.append(f"<td{cls}>{value_text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
    <section>
      <h2>{shot}-shot</h2>
      <table>
        <thead><tr><th>Variant</th><th>Runs</th>{header}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </section>
    """


def collect_rows(shot: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, name in VARIANTS:
        summary = load_summary(label, shot)
        summary_values = summary.get("summary", {}) if summary else {}
        row: dict[str, Any] = {
            "label": label,
            "name": name,
            "runs": safe_float(summary_values.get("num_runs")) if summary else None,
        }
        for metric in METRICS:
            row[metric.key] = metric_value(summary, metric)
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    missing = [
        str(KSHOT_ROOT / ROOT / label / STAGE_SOURCE / DATASET_KEY / f"{shot}_shot_summary.json")
        for label, _ in VARIANTS
        for shot in SHOTS
        if not (KSHOT_ROOT / ROOT / label / STAGE_SOURCE / DATASET_KEY / f"{shot}_shot_summary.json").exists()
    ]
    sections = "\n".join(render_table(shot, collect_rows(shot)) for shot in SHOTS)
    missing_html = ""
    if missing:
        missing_html = (
            "<div class='warn'><strong>Missing summaries:</strong><br>"
            + "<br>".join(html.escape(path) for path in missing[:20])
            + ("<br>..." if len(missing) > 20 else "")
            + "</div>"
        )
    OUTPUT_PATH.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DG-MoBE Residual Fusion Ablation: DeepWeeds</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --ink: #172033;
      --muted: #64748b;
      --card: #ffffff;
      --line: #dbe3ef;
      --best: #d9f99d;
      --accent: #2563eb;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 22px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: -0.03em;
    }}
    .sub {{
      color: var(--muted);
      max-width: 880px;
      line-height: 1.45;
      margin-bottom: 22px;
    }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin: 16px 0;
      box-shadow: 0 16px 42px rgba(15, 23, 42, 0.06);
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 19px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      white-space: normal;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .model {{
      font-weight: 650;
    }}
    .best {{
      background: var(--best);
      font-weight: 800;
      color: #1f3b08;
    }}
    .warn {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 12px;
      padding: 12px;
      margin: 14px 0;
      color: #9a3412;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    code {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <main>
    <h1>DG-MoBE Residual Fusion Ablation: DeepWeeds</h1>
    <p class="sub">
      This report isolates the final logit-combination idea. All variants keep the same DG-MoBE
      expert heads, auxiliary expert loss, responsibility KL, uniform router, and head-diversity
      setup. Only the final logits differ: fused-only, MoE-only, or residual fusion.
      Results are read from <code>kshot/{html.escape(ROOT)}</code>.
    </p>
    {missing_html}
    {sections}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")
    if missing:
        print(f"Missing {len(missing)} expected summaries.")


if __name__ == "__main__":
    main()
