#!/usr/bin/env python3
"""Vanilla deep ensemble over modern backbones (e.g. DINOv2 + ConvNeXt-V2-Tiny).

Each backbone gets its OWN independently-trained classifier probe (plain
cross-entropy, mlp head at native width); the ensemble prediction is the average
of the per-probe softmax distributions. This is the *vanilla deep ensemble*
(independent training, no joint/evidential combination), as distinct from the
DD-CoBE softmax_ensemble (jointly trained heads).

Writes the k-shot result schema and, with --export-logits, the test+calib(val)
logits per (shot,run) for temperature scaling. No evidential alpha (entropy is
the only uncertainty score).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_modern_backbone_experiment as M  # noqa: E402

from ddcomb.evaluation.metrics import (  # noqa: E402
    binary_auroc, ece_from_logits, mce_from_logits, fpr_at_tpr, uncertainty_predictive_entropy_from_logits,
)
from ddcomb.kshot_results import update_legacy_shot_summary, write_json_atomic  # noqa: E402


def make_probe(d, c):
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.LayerNorm(d), nn.Linear(d, c))


def _macro_pr(labels, probs, n):
    preds = probs.argmax(1)
    idx = labels.long() * n + preds.long()
    cm = torch.bincount(idx, minlength=n * n).reshape(n, n)
    tp = cm.diag().float()
    return (tp / cm.sum(0).float().clamp(min=1)).mean().item(), (tp / cm.sum(1).float().clamp(min=1)).mean().item()


def train_eval(train_embs, train_lab, test_embs, test_lab, dims, n_classes, device, seed,
               epochs, lr, wd, bs, calib_embs=None, calib_lab=None, export_logits_path=None):
    torch.manual_seed(seed); np.random.seed(seed)
    probes = [make_probe(int(d), n_classes).to(device) for d in dims]
    train_lab = train_lab.to(device)
    for k, probe in enumerate(probes):  # independent training per backbone
        z = train_embs[k].to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
        for epoch in range(epochs):
            probe.train()
            perm = torch.randperm(z.shape[0], device=device)
            for s in range(0, z.shape[0], bs):
                b = perm[s:s + bs]
                loss = F.cross_entropy(probe(z[b]), train_lab[b])
                opt.zero_grad(); loss.backward(); opt.step()
    for probe in probes:
        probe.eval()

    def _infer(embs, lab):
        out = []
        with torch.no_grad():
            for s in range(0, lab.shape[0], bs):
                ps = [torch.softmax(probe(embs[k][s:s + bs].to(device)), -1) for k, probe in enumerate(probes)]
                out.append(torch.stack(ps).mean(0).cpu())
        probs = torch.cat(out)
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-8)
        return probs, probs.clamp_min(1e-8).log()

    probs, logits = _infer(test_embs, test_lab); labels = test_lab
    prec, rec = _macro_pr(labels, probs, n_classes)
    ent = uncertainty_predictive_entropy_from_logits(logits)
    inc = (probs.argmax(1) != labels).long()
    if export_logits_path is not None and calib_embs is not None:
        c_probs, c_logits = _infer(calib_embs, calib_lab)
        export_logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(export_logits_path, test_logits=logits.numpy().astype(np.float32),
                 test_y=labels.numpy().astype(np.int64), calib_logits=c_logits.numpy().astype(np.float32),
                 calib_y=calib_lab.numpy().astype(np.int64))
    return {"test_accuracy": (probs.argmax(1) == labels).float().mean().item(),
            "test_precision": prec, "test_recall": rec,
            "test_ece": ece_from_logits(logits, labels),
            "test_mce": mce_from_logits(logits, labels),
            "auroc_err_pred_entropy": binary_auroc(ent, inc),
            "fpr95_err_pred_entropy": fpr_at_tpr(ent, inc, target_tpr=0.95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="dinov2_vitb14,convnextv2_tiny")
    ap.add_argument("--dataset", default="deepweeds_no_bg_mergeval")
    ap.add_argument("--shots", default="1,5,10,20")
    ap.add_argument("--runs", default="1-20")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--manifest-root", default="kshot/manifests")
    ap.add_argument("--cache-root", default="embedding_cache_modern")
    ap.add_argument("--deepweeds-root", default="/datasets/DeepWeeds")
    ap.add_argument("--imagefolder-root", default="data/plant_seedlings_v2_split")
    ap.add_argument("--expected-runs", type=int, default=20)
    ap.add_argument("--export-logits", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [b.strip() for b in args.backbones.split(",") if b.strip()]
    shots = [int(s) for s in args.shots.split(",")]
    runs = M._parse_runs(args.runs)

    train_all, test, n_classes, _ = M.build_samples(
        args.dataset, Path(args.deepweeds_root), M.resolve_repo(args.imagefolder_root))
    id_to_train = {}
    for p_, lab, mid in train_all:
        id_to_train.setdefault(mid, (p_, lab, mid))
    cache_root = M.resolve_repo(args.cache_root)
    manifest_root = M.resolve_repo(args.manifest_root)
    results_dir = M.resolve_repo(args.results_root) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    backbones, dims, test_embs = [], [], []
    for name in names:
        bb = M.build_backbone(name); bb.model.to(device)
        backbones.append(bb); dims.append(bb.dim)
        emb, test_lab = M.cached_embeddings(bb, test, device, args.batch_size,
                                            cache_root / name / args.dataset / "test.npz")
        test_embs.append(emb)

    calib_embs, calib_lab = None, None
    if args.export_logits:
        cs = M.build_calib_samples(args.dataset, Path(args.deepweeds_root),
                                   M.resolve_repo(args.imagefolder_root))
        if cs:
            calib_embs = []
            for bb, name in zip(backbones, names):
                emb, calib_lab = M.cached_embeddings(bb, cs, device, args.batch_size,
                                                     cache_root / name / args.dataset / "calib_val.npz")
                calib_embs.append(emb)

    for shot in shots:
        for run in runs:
            rj = results_dir / f"{shot}_shot_run_{run}.json"
            if rj.is_file():
                print(f"skip {rj}"); continue
            sel = M._load_manifest_ids(manifest_root / args.dataset / f"{shot}_shot_run_{run}.json")
            train_samples = [id_to_train[i] for i in sel if i in id_to_train]
            train_embs = []
            for bb, name in zip(backbones, names):
                emb, train_lab = M.cached_embeddings(
                    bb, train_samples, device, args.batch_size,
                    cache_root / name / args.dataset / f"train_{shot}shot_run{run}.npz")
                train_embs.append(emb)
            exlog = (results_dir / f"{shot}_shot_run_{run}_logits.npz") if args.export_logits else None
            m = train_eval(train_embs, train_lab, test_embs, test_lab, dims, n_classes, device,
                           args.seed, args.epochs, args.lr, args.weight_decay, args.batch_size,
                           calib_embs=calib_embs, calib_lab=calib_lab, export_logits_path=exlog)
            row = {"dataset": args.dataset, "stage_a_source": "+".join(names) + "_vanilla_ens",
                   "shot": shot, "run_index": run, "postfix": "vanilla_deep_ensemble",
                   "run_tag": f"vanilla_ens_{'+'.join(names)}_{shot}shot_run{run}_{args.dataset}",
                   "backbones": names, **{k: float(v) for k, v in m.items()}}
            write_json_atomic(rj, row)
            print(f"[VANILLA-ENS] {shot}sh run{run}: acc={m['test_accuracy']:.4f} ece={m['test_ece']:.4f}")
        update_legacy_shot_summary(results_dir=results_dir, shot=shot, expected_runs=args.expected_runs,
                                   summary_prefix={"dataset": args.dataset,
                                                   "stage_a_source": "+".join(names) + "_vanilla_ens",
                                                   "base_seed": args.seed})


if __name__ == "__main__":
    main()
