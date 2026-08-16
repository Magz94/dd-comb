#!/usr/bin/env python3
"""Vanilla (heterogeneous) deep-ensemble baseline from single-backbone exports.

For each (shot, run) it loads the five single-backbone *_predictions.npz files
(ResNet18, VGG16, DenseNet121, EfficientNet-B0, ViT-B/16), averages their
softmax probabilities per sample, and recomputes the standard metrics, writing
one JSON per run plus a per-shot summary under

    kshot/results_deep_ensemble/<dataset>/

so the rows drop straight into the comparison / report builders.

Independent training + probability averaging = the canonical deep ensemble
(Lakshminarayanan et al., 2017). Pure NumPy; no torch required. Run after the
single-backbone prediction-export sweep has written the .npz files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev, pvariance

import numpy as np

REPO = Path(__file__).resolve().parents[1]
KSHOT = REPO / "kshot"
SB_ROOT = KSHOT / "results_single_backbone_uncertainty"
OUT_ROOT = KSHOT / "results_deep_ensemble"
MODE_DIR = "head_only_basic"
MEMBERS = ["resnet18", "vgg16", "densenet121", "efficientnet_b0", "vit_b16"]
DATASET = "deepweeds_no_bg_mergeval"
SHOTS = (1, 5, 10, 20)
RUNS = range(1, 21)
N_BINS = 15


def pred_path(model: str, shot: int, run: int) -> Path:
    return SB_ROOT / DATASET / model / MODE_DIR / f"{shot}_shot_run_{run}_predictions.npz"


def ece_top_label(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(ece)


def macro_pr(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> tuple[float, float]:
    pred = probs.argmax(axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = (tp / np.clip(cm.sum(0), 1, None)).mean()
    rec = (tp / np.clip(cm.sum(1), 1, None)).mean()
    return float(prec), float(rec)


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def binary_auroc(scores: np.ndarray, pos: np.ndarray) -> float:
    # AUROC of `scores` ranking positives (pos==1, here = incorrect) above negatives.
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = pos.sum()
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[pos == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def fpr_at_tpr(scores: np.ndarray, pos: np.ndarray, target_tpr: float = 0.95) -> float:
    pos_scores = np.sort(scores[pos == 1])
    neg_scores = scores[pos == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    # threshold that captures >= target_tpr of positives (high score = positive)
    idx = int(math.floor((1.0 - target_tpr) * len(pos_scores)))
    idx = min(max(idx, 0), len(pos_scores) - 1)
    thr = pos_scores[idx]
    return float((neg_scores >= thr).mean())


def metrics_for_run(shot: int, run: int):
    probs_list, labels_ref = [], None
    for m in MEMBERS:
        p = pred_path(m, shot, run)
        if not p.is_file():
            return None
        d = np.load(p)
        probs_list.append(d["y_prob"].astype(np.float64))
        if labels_ref is None:
            labels_ref = d["y_true"].astype(np.int64)
        elif not np.array_equal(labels_ref, d["y_true"].astype(np.int64)):
            raise RuntimeError(f"Label mismatch across members at shot={shot} run={run} (member {m}).")
    ens = np.mean(probs_list, axis=0)
    ens = ens / ens.sum(axis=1, keepdims=True)
    labels = labels_ref
    num_classes = ens.shape[1]
    pred = ens.argmax(axis=1)
    correct = (pred == labels)
    prec, rec = macro_pr(ens, labels, num_classes)
    ent = predictive_entropy(ens)
    incorrect = (~correct).astype(int)
    return {
        "dataset": DATASET,
        "shot": shot,
        "run_index": run,
        "test_accuracy": float(correct.mean()),
        "test_precision": prec,
        "test_recall": rec,
        "test_ece": ece_top_label(ens, labels),
        "auroc_err_pred_entropy": binary_auroc(ent, incorrect),
        "fpr95_err_pred_entropy": fpr_at_tpr(ent, incorrect, 0.95),
        "model_type": "deep_ensemble_avg_softmax",
        "members": MEMBERS,
    }


def summarize(rows):
    keys = ["test_accuracy", "test_precision", "test_recall", "test_ece",
            "auroc_err_pred_entropy", "fpr95_err_pred_entropy"]
    out = {"num_runs": float(len(rows))}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None and math.isfinite(r[k])]
        if vals:
            name = k.replace("test_", "")
            out[f"{name}_mean"] = mean(vals)
            out[f"{name}_variance"] = pvariance(vals)
            out[f"{name}_std"] = pstdev(vals)
    return out


def main():
    import argparse
    global SB_ROOT, OUT_ROOT, DATASET
    ap = argparse.ArgumentParser(description="Average single-backbone predictions into a deep ensemble.")
    ap.add_argument("--sb-root", default=str(SB_ROOT), help="root holding <dataset>/<model>/head_only_basic/*_predictions.npz")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="output root for the deep-ensemble results")
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()
    SB_ROOT = Path(args.sb_root)
    OUT_ROOT = Path(args.out_root)
    DATASET = args.dataset
    print(f"[deep-ensemble] sb_root={SB_ROOT}  out_root={OUT_ROOT}  dataset={DATASET}")

    out_dir = OUT_ROOT / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    for shot in SHOTS:
        rows = []
        for run in RUNS:
            r = metrics_for_run(shot, run)
            if r is None:
                print(f"[skip] missing member predictions for shot={shot} run={run}")
                continue
            (out_dir / f"{shot}_shot_run_{run}.json").write_text(json.dumps(r, indent=2))
            rows.append(r)
        if rows:
            summ = {"dataset": DATASET, "shot": shot, "runs": len(rows), "summary": summarize(rows)}
            (out_dir / f"{shot}_shot_summary.json").write_text(json.dumps(summ, indent=2))
            s = summ["summary"]
            print(f"{shot:>2}-shot (n={len(rows)}): "
                  f"acc={s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}  "
                  f"ece={s['ece_mean']:.4f}  auroc={s['auroc_err_pred_entropy_mean']:.4f}")
    print(f"Wrote deep-ensemble results under {out_dir}")


if __name__ == "__main__":
    main()
