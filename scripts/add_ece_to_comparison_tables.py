#!/usr/bin/env python3
"""Append an ECE column (15-bin, lower is better) to every table in
docs/kshot_full_comparison_tables.html, with winner highlighting (min).

Sources verified against the displayed accuracy values:
- single backbones:   kshot/results_single_backbone_uncertainty/<ds>/<model>/**
- Fusion Baseline:    kshot/results_baseline_fusion_uncertainty/**/<ds>/
- DirAttn + KL:       kshot/results_dirichlet_attention_kl1e-4_warmup3_uncertainty_20runs/**/<ds>/
- Logit-MoE + HeadDiv kshot/results_dirichlet_logit_moe_headdiv/**/<ds>/
- DD-CoBE:            dempster result folders.
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
SHOTS = (1, 5, 10, 20)

DATASET_KEYS = {
    "DeepWeeds": "deepweeds_no_bg_mergeval",
    "Plant Seedlings V2": "plant_seedlings_v2_split",
    "BloodMNIST": "bloodmnist_224_imagefolder_mergeval",
    "APTOS 2019": "aptos2019_split",
    "LeafSugar": "leafsugar_disease_split",
    "CottonLeaf": "sar_cld_original_split",
    "SSL_Patches": "ssl_patches",
}

SINGLE = {
    "ViT-B16": "vit_b16",
    "DenseNet121": "densenet121",
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet18": "resnet18",
    "VGG16": "vgg16",
}


def patterns_for(model: str, ds: str) -> list[str]:
    if model in SINGLE:
        return [f"results_single_backbone_uncertainty/{ds}/{SINGLE[model]}/**"]
    if model == "Fusion Baseline":
        return [f"results_baseline_fusion_uncertainty/**/{ds}"]
    if model == "DirAttn + KL":
        return [f"results_dirichlet_attention_kl1e-4_warmup3_uncertainty_20runs/**/{ds}"]
    if model == "Logit-MoE + HeadDiv":
        return [f"results_dirichlet_logit_moe_headdiv/**/{ds}"]
    if model.startswith("DD-CoBE"):
        if ds == "deepweeds_no_bg_mergeval":
            return [f"results_dgmobe_dempster_fusion_deepweeds/pure_kl0p1_fv/**/{ds}"]
        return [f"results_ddcobe_dempster_multidataset/pure_kl0p1_fv/**/{ds}"]
    return []


def ece_mean(model: str, ds: str, shot: int):
    for pat in patterns_for(model, ds):
        files = glob.glob(str(KSHOT / pat / f"{shot}_shot_run_*.json"), recursive=True)
        vals = []
        for p in files:
            d = json.load(open(p))
            if "test_ece" in d and d["test_ece"] is not None:
                vals.append(float(d["test_ece"]))
        if vals:
            return mean(vals)
    return None


ROW_RE = re.compile(
    r'(<tr><td class="model[^"]*">(?P<name>[^<]+)</td>\s*(?:<td class="num[^"]*">[0-9.]+</td>\s*)+)</tr>',
    re.S,
)


def process_table(table_html: str, ds_key: str, shot: int) -> str:
    # collect ECE per row
    names = [m.group("name") for m in ROW_RE.finditer(table_html)]
    eces = {n: ece_mean(n, ds_key, shot) for n in names}
    present = [v for v in eces.values() if v is not None]
    best = min(present) if present else None

    # header: add ECE column if not present
    if "<th>ECE" not in table_html:
        table_html = table_html.replace("<th>FPR95-H</th>", "<th>FPR95-H</th><th>ECE</th>", 1)

    def add_cell(m):
        name = m.group("name")
        v = eces.get(name)
        if v is None:
            cell = '<td class="num ">-</td>'
        else:
            cls = "num winner" if best is not None and abs(v - best) < 5e-5 else "num "
            cell = f'<td class="{cls}">{v:.4f}</td>'
        return m.group(1) + cell + "</tr>"

    return ROW_RE.sub(add_cell, table_html)


def main():
    html = HTML_PATH.read_text(encoding="utf-8")
    backup = HTML_PATH.with_suffix(".html.bak2")
    if not backup.exists():
        shutil.copy(HTML_PATH, backup)

    sections = re.split(r"(<h2>)", html)
    out = [sections[0]]
    i = 1
    while i < len(sections):
        chunk = sections[i] + sections[i + 1]
        title_m = re.match(r"<h2>([^<\n]+)", chunk)
        title = title_m.group(1).strip() if title_m else ""
        ds_key = DATASET_KEYS.get(title)
        if ds_key:
            tables = re.split(r"(<table>.*?</table>)", chunk, flags=re.S)
            shot_idx = 0
            rebuilt = []
            for part in tables:
                if part.startswith("<table>") and shot_idx < len(SHOTS):
                    part = process_table(part, ds_key, SHOTS[shot_idx])
                    shot_idx += 1
                rebuilt.append(part)
            chunk = "".join(rebuilt)
            print(f"{title}: ECE added to {shot_idx} tables")
        out.append(chunk)
        i += 2

    html = "".join(out)
    html = html.replace(
        "lower is better for FPR95-H.",
        "lower is better for FPR95-H and ECE (15-bin expected calibration error).",
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_PATH} (backup: {backup.name})")


if __name__ == "__main__":
    main()
