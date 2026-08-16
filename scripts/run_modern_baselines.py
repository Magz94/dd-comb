#!/usr/bin/env python3
"""Single-backbone uncertainty baselines (Sensoy EDL, MC Dropout) on a MODERN
frozen backbone (e.g. DINOv2, ConvNeXt-V2), reusing the modern-backbone
embedding pipeline so the few-shot manifests/metrics match the rest of the study.

Replicates the *standard* recipe used for the original backbones
(``scripts/sweep_kshot_baselines.py``):
  * sensoy     -> SensoyEDLModel, linear head (hidden=0), relu evidence, MSE/Brier
                  loss, KL warmup 10.
  * mc_dropout -> MCDropoutModel, hidden 512, p=0.5, T=50 test-time passes.
Both: AdamW lr 1e-3, weight_decay 0, 10 epochs (mc_dropout uses --weight-decay
if you want the L2 term of Gal's objective).

Writes the k-shot baseline JSON schema, one file per (shot, run), plus the
legacy shot summary, under ``--results-root/<dataset>/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_modern_backbone_experiment as M  # noqa: E402  (build_samples, cached_embeddings, ...)

from ddcomb.evaluation.metrics import (  # noqa: E402
    binary_auroc, ece_from_logits, mce_from_logits, fpr_at_tpr, uncertainty_predictive_entropy_from_logits,
)
from ddcomb.kshot_results import update_legacy_shot_summary, write_json_atomic  # noqa: E402
from ddcomb.models.baselines_single import MCDropoutModel, SensoyEDLModel  # noqa: E402


def _macro_pr(labels, probs, n):
    preds = probs.argmax(1)
    idx = labels.long() * n + preds.long()
    cm = torch.bincount(idx, minlength=n * n).reshape(n, n)
    tp = cm.diag().float()
    return (tp / cm.sum(0).float().clamp(min=1)).mean().item(), (tp / cm.sum(1).float().clamp(min=1)).mean().item()


def train_eval(method, train_emb, train_lab, test_emb, test_lab, in_dim, n_classes,
               device, seed, epochs, lr, wd, hidden, p, n_samples, kl_warmup,
               calib_emb=None, calib_lab=None, export_logits_path=None):
    torch.manual_seed(seed); np.random.seed(seed)
    is_mcd = method == "mc_dropout"
    if is_mcd:
        model = MCDropoutModel(in_dim=in_dim, n_classes=n_classes,
                               hidden_dim=(hidden if hidden > 0 else 512), p=p, n_samples=n_samples).to(device)
    else:
        model = SensoyEDLModel(in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden,
                               evidence="relu", loss="mse", kl_warmup_epochs=kl_warmup).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_emb = train_emb.to(device); train_lab = train_lab.to(device)
    n_tr = train_lab.shape[0]; bs = 64
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, bs):
            b = perm[s:s + bs]
            out = model.classify(train_emb[b])
            loss = model.compute_loss(out, train_lab[b], is_train=True, epoch_idx=epoch)
            opt.zero_grad(); loss.backward(); opt.step()
    # eval
    if is_mcd:
        model.train()  # keep dropout active at test (Gal 2016); no BatchNorm present
    else:
        model.eval()

    def _infer(emb, lab):
        lg, pb, al = [], [], []
        with torch.no_grad():
            for s in range(0, lab.shape[0], bs):
                z = emb[s:s + bs].to(device)
                out = model.predict_mc(z) if is_mcd else model.classify(z)
                lg.append(out["logits"].cpu()); pb.append(out["probs"].cpu())
                al.append(out["alpha"].cpu() if (not is_mcd and "alpha" in out) else None)
        alpha = torch.cat(al) if al and al[0] is not None else None
        return torch.cat(lg), torch.cat(pb), alpha

    logits, probs, alpha = _infer(test_emb, test_lab); labels = test_lab
    if export_logits_path is not None and calib_emb is not None:
        c_lg, _c_pb, c_al = _infer(calib_emb, calib_lab)
        export_logits_path.parent.mkdir(parents=True, exist_ok=True)
        d = {"test_logits": logits.numpy().astype(np.float32), "test_y": labels.numpy().astype(np.int64),
             "calib_logits": c_lg.numpy().astype(np.float32), "calib_y": calib_lab.numpy().astype(np.int64)}
        if alpha is not None:
            d["test_alpha"] = alpha.numpy().astype(np.float32)
            d["calib_alpha"] = c_al.numpy().astype(np.float32)
        np.savez(export_logits_path, **d)
    prec, rec = _macro_pr(labels, probs, n_classes)
    ent = uncertainty_predictive_entropy_from_logits(logits)
    inc = (probs.argmax(1) != labels).long()
    return {
        "test_accuracy": (probs.argmax(1) == labels).float().mean().item(),
        "test_precision": prec, "test_recall": rec,
        "test_ece": ece_from_logits(logits, labels),
        "test_mce": mce_from_logits(logits, labels),
        "auroc_err_pred_entropy": binary_auroc(ent, inc),
        "fpr95_err_pred_entropy": fpr_at_tpr(ent, inc, target_tpr=0.95),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="modern backbone name (e.g. dinov2_vitb14)")
    ap.add_argument("--method", required=True, choices=["sensoy", "mc_dropout", "both"])
    ap.add_argument("--dataset", default="deepweeds_no_bg_mergeval")
    ap.add_argument("--shots", default="1,5,10,20")
    ap.add_argument("--runs", default="1-20")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=None, help="override head hidden dim; default per method")
    ap.add_argument("--mc-p", type=float, default=0.5)
    ap.add_argument("--mc-samples", type=int, default=50)
    ap.add_argument("--kl-warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--manifest-root", default="kshot/manifests")
    ap.add_argument("--cache-root", default="embedding_cache_modern")
    ap.add_argument("--deepweeds-root", default="/datasets/DeepWeeds")
    ap.add_argument("--imagefolder-root", default="data/plant_seedlings_v2_split")
    ap.add_argument("--expected-runs", type=int, default=20)
    ap.add_argument("--export-logits", action="store_true",
                    help="save test+calib(val) logits and alpha per (shot,run) for TS / vacuity scoring")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ["sensoy", "mc_dropout"] if args.method == "both" else [args.method]
    shots = [int(s) for s in args.shots.split(",")]
    runs = M._parse_runs(args.runs)

    train_all, test, n_classes, _ = M.build_samples(
        args.dataset, Path(args.deepweeds_root), M.resolve_repo(args.imagefolder_root))
    id_to_train = {}
    for p_, lab, mid in train_all:
        id_to_train.setdefault(mid, (p_, lab, mid))

    cache_root = M.resolve_repo(args.cache_root)
    manifest_root = M.resolve_repo(args.manifest_root)

    bb = M.build_backbone(args.backbone); bb.model.to(device)
    te_cache = cache_root / args.backbone / args.dataset / "test.npz"
    test_emb, test_lab = M.cached_embeddings(bb, test, device, args.batch_size, te_cache)

    calib_emb, calib_lab = None, None
    if args.export_logits:
        calib_samples = M.build_calib_samples(args.dataset, Path(args.deepweeds_root),
                                              M.resolve_repo(args.imagefolder_root))
        if calib_samples:
            ca_cache = cache_root / args.backbone / args.dataset / "calib_val.npz"
            calib_emb, calib_lab = M.cached_embeddings(bb, calib_samples, device, args.batch_size, ca_cache)

    for method in methods:
        # MC-Dropout: native width (= embedding dim) to match the fusion's per-backbone
        # mlp heads (fair head capacity). Sensoy: linear head (hidden=0, standard recipe).
        hidden = args.hidden if args.hidden is not None else (bb.dim if method == "mc_dropout" else 0)
        results_dir = M.resolve_repo(args.results_root) / method / args.backbone / args.dataset
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BASELINE-MODERN] {method} on {args.backbone} hidden={hidden} device={device}")
        for shot in shots:
            for run in runs:
                rj = results_dir / f"{shot}_shot_run_{run}.json"
                if rj.is_file():
                    print(f"  skip {rj}"); continue
                manifest = manifest_root / args.dataset / f"{shot}_shot_run_{run}.json"
                sel = M._load_manifest_ids(manifest)
                train_samples = [id_to_train[i] for i in sel if i in id_to_train]
                if len(train_samples) != len(sel):
                    raise ValueError(f"{len(sel)-len(train_samples)} manifest ids missing for {args.dataset}.")
                tr_cache = cache_root / args.backbone / args.dataset / f"train_{shot}shot_run{run}.npz"
                train_emb, train_lab = M.cached_embeddings(bb, train_samples, device, args.batch_size, tr_cache)
                exlog = (results_dir / f"{shot}_shot_run_{run}_logits.npz") if args.export_logits else None
                metrics = train_eval(method, train_emb, train_lab, test_emb, test_lab, bb.dim, n_classes,
                                     device, args.seed, args.epochs, args.lr, args.weight_decay,
                                     hidden, args.mc_p, args.mc_samples, args.kl_warmup,
                                     calib_emb=calib_emb, calib_lab=calib_lab, export_logits_path=exlog)
                row = {"dataset": args.dataset, "stage_a_source": args.backbone, "shot": shot,
                       "run_index": run, "postfix": f"{method}_{args.backbone}",
                       "run_tag": f"baseline_{method}_{args.backbone}_{shot}shot_run{run}_{args.dataset}",
                       "baseline_method": method, "baseline_backbone": args.backbone,
                       **{k: float(v) for k, v in metrics.items()}}
                write_json_atomic(rj, row)
                print(f"  {method} {args.backbone} {shot}sh run{run}: acc={metrics['test_accuracy']:.4f} "
                      f"ece={metrics['test_ece']:.4f}")
            update_legacy_shot_summary(results_dir=results_dir, shot=shot, expected_runs=args.expected_runs,
                                       summary_prefix={"dataset": args.dataset, "stage_a_source": args.backbone,
                                                       "base_seed": args.seed})


if __name__ == "__main__":
    main()
