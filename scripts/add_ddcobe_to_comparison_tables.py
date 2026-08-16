#!/usr/bin/env python3
"""Insert DD-CoBE (pure KL=0.1, fused view) rows into
docs/kshot_full_comparison_tables.html and recompute winner highlighting.

Conventions copied from the existing tables:
- one mean value per cell, 4 decimals;
- Macro-F1 = harmonic mean of macro precision and macro recall (computed
  from the displayed means, matching the existing rows);
- winner per column: max for Accuracy/Precision/Recall/Macro-F1/AUROC-H,
  min for FPR95-H.
"""
from __future__ import annotations

import glob
import json
import re
import shutil
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
HTML_PATH = REPO / "docs" / "kshot_full_comparison_tables.html"
KSHOT = REPO / "kshot"
MODEL_NAME = "DD-CoBE (Dempster, KL=0.1)"
SHOTS = (1, 5, 10, 20)

DATASETS = {
    "DeepWeeds": ("results_dgmobe_dempster_fusion_deepweeds/pure_kl0p1_fv", "deepweeds_no_bg_mergeval"),
    "Plant Seedlings V2": ("results_ddcobe_dempster_multidataset/pure_kl0p1_fv", "plant_seedlings_v2_split"),
    "BloodMNIST": ("results_ddcobe_dempster_multidataset/pure_kl0p1_fv", "bloodmnist_224_imagefolder_mergeval"),
    "SSL_Patches": ("results_ddcobe_dempster_multidataset/pure_kl0p1_fv", "ssl_patches"),
}


def ddcobe_stats(rel_root: str, dataset: str, shot: int):
    pattern = str(KSHOT / rel_root / "**" / dataset / f"{shot}_shot_run_*.json")
    rows = [json.load(open(p)) for p in glob.glob(pattern, recursive=True)]
    if not rows:
        return None
    acc = mean(r["test_accuracy"] for r in rows)
    prec = mean(r["test_precision"] for r in rows)
    rec = mean(r["test_recall"] for r in rows)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auroc = mean(r["auroc_err_pred_entropy"] for r in rows if "auroc_err_pred_entropy" in r)
    fpr = mean(r["fpr95_err_pred_entropy"] for r in rows if "fpr95_err_pred_entropy" in r)
    return [acc, prec, rec, f1, auroc, fpr]


ROW_RE = re.compile(
    r'<tr><td class="model[^"]*">(?P<name>[^<]+)</td>\s*'
    r'(?P<cells>(?:<td class="num[^"]*">[0-9.]+</td>\s*){6})</tr>',
    re.S,
)
CELL_RE = re.compile(r'<td class="num[^"]*">([0-9.]+)</td>')


def rebuild_table(table_html: str, new_vals):
    """Insert/replace the DD-CoBE row and recompute winner classes."""
    rows = []  # (name, [6 floats])
    for m in ROW_RE.finditer(table_html):
        name = m.group("name")
        vals = [float(x) for x in CELL_RE.findall(m.group("cells"))]
        if name != MODEL_NAME:
            rows.append((name, vals))
    if new_vals is not None:
        rows.append((MODEL_NAME, new_vals))

    # winners per column
    winners = []
    for col in range(6):
        col_vals = [v[col] for _, v in rows]
        best = min(col_vals) if col == 5 else max(col_vals)
        winners.append(best)

    body = []
    for name, vals in rows:
        cells = []
        for col, v in enumerate(vals):
            is_win = abs(v - winners[col]) < 5e-5
            cls = "num winner" if is_win else "num "
            cells.append(f'<td class="{cls}">{v:.4f}</td>')
        body.append(
            f'<tr><td class="model fusion-row">{name}</td>\n' + "\n".join(cells) + "\n</tr>"
        )
    header = table_html.split("<tbody>")[0] + "<tbody>\n"
    return header + "\n".join(body) + "\n</tbody></table>"


def main():
    html = HTML_PATH.read_text(encoding="utf-8")
    backup = HTML_PATH.with_suffix(".html.bak")
    if not backup.exists():
        shutil.copy(HTML_PATH, backup)

    # split into dataset sections by <h2>
    sections = re.split(r"(<h2>)", html)
    out = [sections[0]]
    i = 1
    while i < len(sections):
        chunk = sections[i] + sections[i + 1]  # '<h2>' + content
        title_m = re.match(r"<h2>([^<\n]+)", chunk)
        title = title_m.group(1).strip() if title_m else ""
        if title in DATASETS:
            rel_root, ds_key = DATASETS[title]
            tables = re.split(r"(<table>.*?</table>)", chunk, flags=re.S)
            shot_idx = 0
            rebuilt = []
            for part in tables:
                if part.startswith("<table>") and shot_idx < len(SHOTS):
                    vals = ddcobe_stats(rel_root, ds_key, SHOTS[shot_idx])
                    part = rebuild_table(part, vals)
                    shot_idx += 1
                rebuilt.append(part)
            chunk = "".join(rebuilt)
            print(f"updated {title}: {shot_idx} tables")
        out.append(chunk)
        i += 2

    html = "".join(out)
    html = html.replace(
        "DirAttn + KL, and Logit-MoE + HeadDiv.",
        "DirAttn + KL, Logit-MoE + HeadDiv, and DD-CoBE "
        "(Dempster&ndash;Dirichlet Combination of Backbone Experts, pure loss, KL=0.1, fused view).",
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_PATH} (backup at {backup.name})")


if __name__ == "__main__":
    main()
