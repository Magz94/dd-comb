#!/usr/bin/env python3
"""Clean, self-contained modern-backbone DD-CoBE experiment.

Independent of the shared-transform embedding pipeline. For one or more modern
frozen backbones (DINOv2 / OpenCLIP / ConvNeXt-V2), it:

  1. loads the dataset (k-shot train via the shared manifests, full test);
  2. extracts + caches each backbone's embeddings using THAT backbone's own
     preprocessing (test embeddings cached once; train per shot/run);
  3. trains the requested model on the cached embeddings -- the same
     ``DDCoBEEnsemble`` used elsewhere (combine = dempster | mean_evidence |
     softmax_ensemble), over the selected backbones (K=1 = single backbone);
  4. evaluates with the shared metrics and writes results in the k-shot schema.

Backbones are loaded once and looped over shots x runs for efficiency.

Example (3-backbone DD-CoBE, common width, DeepWeeds):
    python scripts/run_modern_backbone_experiment.py \
        --dataset deepweeds_no_bg_mergeval \
        --backbones dinov2_vitb14,openclip_vitb16,convnextv2_tiny \
        --combine dempster --kl 0.1 --width common \
        --shots 1,5,10,20 --runs 1-20 \
        --results-root kshot/results_modern/dino_clip_cnx_dempster_common
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from ddcomb.evaluation.metrics import (
    binary_auroc,
    ece_from_logits,
    mce_from_logits,
    fpr_at_tpr,
    uncertainty_predictive_entropy_from_logits,
)
from ddcomb.kshot_results import update_legacy_shot_summary, write_json_atomic
from ddcomb.models.ddcobe_ensemble import DDCoBEEnsemble
from ddcomb.models.modern_backbones import build_backbone
from ddcomb.training.single_backbone import (
    DeepWeedsCSVDataset,
    resolve_deepweeds_csv,
)

REPO = Path(__file__).resolve().parents[1]
Sample = Tuple[str, int, str]  # (abs_path, label, manifest_id)


# ----------------------------------------------------------------- datasets
def _load_manifest_ids(manifest_path: Path) -> set:
    m = json.loads(Path(manifest_path).read_text())
    return {str(it["id"]) for it in m["selected"]}


def build_samples(dataset: str, deepweeds_root: Path, imagefolder_root: Path):
    """Return (train_all, test, num_classes, manifest_kind)."""
    if dataset == "deepweeds_no_bg_mergeval":
        images_dir = deepweeds_root / "images"
        labels_dir = deepweeds_root / "labels"
        drop = {8}
        tr_csv = resolve_deepweeds_csv(labels_dir, "subset0", "train")
        va_csv = resolve_deepweeds_csv(labels_dir, "subset0", "val")
        te_csv = resolve_deepweeds_csv(labels_dir, "subset0", "test")
        tr = DeepWeedsCSVDataset(images_dir=images_dir, csv_path=tr_csv, transform=None, drop_labels=drop)
        va = DeepWeedsCSVDataset(images_dir=images_dir, csv_path=va_csv, transform=None, drop_labels=drop)
        te = DeepWeedsCSVDataset(images_dir=images_dir, csv_path=te_csv, transform=None, drop_labels=drop)
        train_all = [(tr._resolve_image_path(fn), lab, fn) for fn, lab in (tr.samples + va.samples)]
        test = [(te._resolve_image_path(fn), lab, fn) for fn, lab in te.samples]
        return train_all, test, 8, "filename"

    if dataset == "plant_seedlings_v2_split":
        root = imagefolder_root
        tr = datasets.ImageFolder(root=str(root / "train"))
        te = datasets.ImageFolder(root=str(root / "test"))
        train_all = [(p, lab, _relpath(p, root)) for p, lab in tr.samples]
        test = [(p, lab, _relpath(p, root)) for p, lab in te.samples]
        return train_all, test, len(tr.classes), "relpath"

    if dataset == "ssl_patches":
        # train/val only; the original single-backbone pipeline uses 'val' as the
        # evaluation (test) split. Few-shot pool = train (matches the manifests).
        root = imagefolder_root
        tr = datasets.ImageFolder(root=str(root / "train"))
        te = datasets.ImageFolder(root=str(root / "val"))
        train_all = [(p, lab, _relpath(p, root)) for p, lab in tr.samples]
        test = [(p, lab, _relpath(p, root)) for p, lab in te.samples]
        return train_all, test, len(tr.classes), "relpath"

    if dataset == "bloodmnist_224_imagefolder_mergeval":
        # Official train/val/test. 'mergeval': the few-shot pool = train + val
        # (matches the manifest class_pool_sizes), test = official test split.
        root = imagefolder_root
        tr = datasets.ImageFolder(root=str(root / "train"))
        va = datasets.ImageFolder(root=str(root / "val"))
        te = datasets.ImageFolder(root=str(root / "test"))
        train_all = [(p, lab, _relpath(p, root)) for p, lab in (tr.samples + va.samples)]
        test = [(p, lab, _relpath(p, root)) for p, lab in te.samples]
        return train_all, test, len(tr.classes), "relpath"

    raise ValueError(f"Unsupported dataset: {dataset}")


def _relpath(path: str, root: Path) -> str:
    import os
    return os.path.relpath(path, root).replace("\\", "/")


def _stratified_calib(samples, pct: int = 15):
    """Deterministic held-out calibration subset for datasets without a dedicated
    val split (Plant Seedlings, SSL-Patches): the last ceil(pct%) samples of each
    class, sorted by id. No RNG, so it is identical across runs."""
    from collections import defaultdict
    by = defaultdict(list)
    for s in samples:
        by[s[1]].append(s)
    out = []
    for lab in sorted(by):
        items = sorted(by[lab], key=lambda x: x[2])
        k = max(1, (len(items) * pct + 99) // 100)
        out.extend(items[-k:])
    return out


def build_calib_samples(dataset: str, deepweeds_root: Path, imagefolder_root: "Path | None" = None):
    """Validation samples used ONLY to fit the post-hoc temperature. DeepWeeds and
    BloodMNIST have a dedicated 'val' split; Plant Seedlings and SSL-Patches do not,
    so a deterministic stratified subset is carved from their train pool."""
    if dataset == "deepweeds_no_bg_mergeval":
        images_dir = deepweeds_root / "images"
        labels_dir = deepweeds_root / "labels"
        va = DeepWeedsCSVDataset(images_dir=images_dir,
                                 csv_path=resolve_deepweeds_csv(labels_dir, "subset0", "val"),
                                 transform=None, drop_labels={8})
        return [(va._resolve_image_path(fn), lab, fn) for fn, lab in va.samples]

    if dataset == "bloodmnist_224_imagefolder_mergeval":
        # Dedicated val split (also merged into the few-shot pool, as in DeepWeeds).
        root = imagefolder_root
        va = datasets.ImageFolder(root=str(root / "val"))
        return [(p, lab, _relpath(p, root)) for p, lab in va.samples]

    if dataset in ("plant_seedlings_v2_split", "ssl_patches"):
        # No dedicated val: carve a fixed stratified subset from the train pool.
        root = imagefolder_root
        tr = datasets.ImageFolder(root=str(root / "train"))
        train_all = [(p, lab, _relpath(p, root)) for p, lab in tr.samples]
        return _stratified_calib(train_all, pct=15)

    return []


# -------------------------------------------------------------- embeddings
@torch.no_grad()
def extract_embeddings(backbone, samples: List[Sample], device: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    embs, labels = [], []
    buf_imgs, buf_lab = [], []

    def flush():
        if not buf_imgs:
            return
        batch = torch.stack(buf_imgs).to(device)
        feats = backbone.embed(batch).float().detach().cpu()
        embs.append(feats)
        labels.extend(buf_lab)
        buf_imgs.clear()
        buf_lab.clear()

    for path, lab, _id in samples:
        img = Image.open(path).convert("RGB")
        buf_imgs.append(backbone.transform(img))
        buf_lab.append(int(lab))
        if len(buf_imgs) >= batch_size:
            flush()
    flush()
    return torch.cat(embs), torch.tensor(labels, dtype=torch.long)


def cached_embeddings(backbone, samples, device, batch_size, cache_path: Path):
    if cache_path.is_file():
        try:
            d = np.load(cache_path)
            return torch.from_numpy(d["emb"]).float(), torch.from_numpy(d["labels"]).long()
        except (EOFError, OSError, ValueError, zipfile.BadZipFile):
            # Corrupt/truncated cache from an interrupted or out-of-space prior
            # run: drop it and recompute instead of crashing.
            cache_path.unlink(missing_ok=True)
    emb, lab = extract_embeddings(backbone, samples, device, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically (temp file + rename) so an interrupted write can never
    # leave a poisoned cache behind for the next run.
    tmp = cache_path.parent / (cache_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, emb=emb.numpy(), labels=lab.numpy())
    tmp.replace(cache_path)
    return emb, lab


# -------------------------------------------------------------- train/eval
def _macro_pr(labels, probs, n):
    preds = probs.argmax(1)
    idx = labels.long() * n + preds.long()
    cm = torch.bincount(idx, minlength=n * n).reshape(n, n)
    tp = cm.diag().float()
    return (tp / cm.sum(0).float().clamp(min=1)).mean().item(), (tp / cm.sum(1).float().clamp(min=1)).mean().item()


def train_eval(train_embs, train_lab, test_embs, test_lab, dims, n_classes, combine, kl, width,
               epochs, lr, wd, kl_warmup, batch_size, device, seed, head="mlp", loss_mode="expected_ce",
               export_path: "Path | None" = None, calib_embs=None,
               calib_lab=None, export_logits_path: "Path | None" = None) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_dims = [int(min(dims))] * len(dims) if width == "common" else [int(d) for d in dims]
    model = DDCoBEEnsemble(
        input_dims=[int(d) for d in dims], n_classes=n_classes, combine=combine,
        kl_lambda=kl, kl_warmup_epochs=kl_warmup, hidden_dims=hidden_dims, head_type=head,
        loss_mode=loss_mode,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_embs = [e.to(device) for e in train_embs]
    train_lab = train_lab.to(device)
    n_tr = train_lab.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, batch_size):
            bidx = perm[s:s + batch_size]
            out = model.classify([e[bidx] for e in train_embs])
            loss = model.compute_loss(out, train_lab[bidx], is_train=True, epoch_idx=epoch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()

    def _infer(embs, lab):
        lg, pb, al, lb = [], [], [], []
        n = lab.shape[0]
        with torch.no_grad():
            for s in range(0, n, batch_size):
                sl = slice(s, s + batch_size)
                out = model.classify([e[sl].to(device) for e in embs])
                lg.append(out["logits"].cpu()); pb.append(out["probs"].cpu()); lb.append(lab[sl])
                al.append(out["alpha"].cpu() if out.get("alpha") is not None else None)
        alpha = torch.cat(al) if al and al[0] is not None else None
        return torch.cat(lg), torch.cat(pb), torch.cat(lb), alpha

    logits, probs, labels, alpha = _infer(test_embs, test_lab)
    prec, rec = _macro_pr(labels, probs, n_classes)
    ent = uncertainty_predictive_entropy_from_logits(logits)
    incorrect = (probs.argmax(1) != labels).long()
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(export_path, head_pred=probs.argmax(1).numpy().astype(np.int64),
                 y_true=labels.numpy().astype(np.int64), probs=probs.numpy().astype(np.float32))
    if export_logits_path is not None and calib_embs is not None:
        c_lg, _c_pb, c_lb, c_al = _infer(calib_embs, calib_lab)
        export_logits_path.parent.mkdir(parents=True, exist_ok=True)
        d = {"test_logits": logits.numpy().astype(np.float32),
             "test_y": labels.numpy().astype(np.int64),
             "calib_logits": c_lg.numpy().astype(np.float32),
             "calib_y": c_lb.numpy().astype(np.int64)}
        if alpha is not None:
            d["test_alpha"] = alpha.numpy().astype(np.float32)
            d["calib_alpha"] = c_al.numpy().astype(np.float32)
        np.savez(export_logits_path, **d)
    return {
        "test_accuracy": (probs.argmax(1) == labels).float().mean().item(),
        "test_precision": prec,
        "test_recall": rec,
        "test_ece": ece_from_logits(logits, labels),
        "test_mce": mce_from_logits(logits, labels),
        "auroc_err_pred_entropy": binary_auroc(ent, incorrect),
        "fpr95_err_pred_entropy": fpr_at_tpr(ent, incorrect, target_tpr=0.95),
    }


# ------------------------------------------------------------------- main
def _parse_runs(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbones", required=True, help="comma-separated modern backbone names")
    ap.add_argument("--combine", default="dempster", choices=["dempster", "mean_evidence", "softmax_ensemble", "product_softmax"])
    ap.add_argument("--loss-mode", default="expected_ce", choices=["expected_ce", "brier"],
                    help="evidential data term for dempster/mean_evidence (expected_ce | brier)")
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--kl-warmup", type=int, default=10)
    ap.add_argument("--width", default="common", choices=["common", "native"])
    ap.add_argument("--head", default="mlp", choices=["mlp", "linear_ln"],
                    help="per-backbone head: mlp adapter+LN, or linear_ln probe (no hidden adapter)")
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
    ap.add_argument("--export-preds", action="store_true",
                    help="save per-sample test predictions npz per (shot,run) for disagreement/oracle")
    ap.add_argument("--export-logits", action="store_true",
                    help="save test+calib(val) logits and evidential alpha per (shot,run) for "
                         "temperature scaling and vacuity-based scoring")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone_names = [b.strip() for b in args.backbones.split(",") if b.strip()]
    shots = [int(s) for s in args.shots.split(",")]
    runs = _parse_runs(args.runs)
    stage_label = "+".join(backbone_names) + f"_{args.width}_{args.head}"
    print(f"[MODERN] dataset={args.dataset} backbones={backbone_names} combine={args.combine} "
          f"width={args.width} device={device}")

    train_all, test, n_classes, kind = build_samples(
        args.dataset, Path(args.deepweeds_root), resolve_repo(args.imagefolder_root))
    id_to_train = {}
    for p, lab, mid in train_all:
        id_to_train.setdefault(mid, (p, lab, mid))

    results_root = resolve_repo(args.results_root)
    cache_root = resolve_repo(args.cache_root)
    manifest_root = resolve_repo(args.manifest_root)
    results_dir = results_root / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build backbones once; extract+cache test embeddings once per backbone.
    backbones, dims, test_embs = [], [], []
    for name in backbone_names:
        bb = build_backbone(name)
        bb.model.to(device)
        backbones.append(bb)
        dims.append(bb.dim)
        te_cache = cache_root / name / args.dataset / "test.npz"
        emb, lab_te = cached_embeddings(bb, test, device, args.batch_size, te_cache)
        test_embs.append(emb)
    test_lab = lab_te  # labels identical across backbones (same sample order)

    # Calibration (val) embeddings once per backbone, for temperature scaling.
    calib_embs, calib_lab = None, None
    if args.export_logits:
        calib_samples = build_calib_samples(args.dataset, Path(args.deepweeds_root),
                                            resolve_repo(args.imagefolder_root))
        if calib_samples:
            calib_embs = []
            for bb, name in zip(backbones, backbone_names):
                ca_cache = cache_root / name / args.dataset / "calib_val.npz"
                emb, calib_lab = cached_embeddings(bb, calib_samples, device, args.batch_size, ca_cache)
                calib_embs.append(emb)

    for shot in shots:
        for run in runs:
            run_json = results_dir / f"{shot}_shot_run_{run}.json"
            if run_json.is_file():
                print(f"[MODERN] skip existing {run_json}")
                continue
            manifest = manifest_root / args.dataset / f"{shot}_shot_run_{run}.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"Missing manifest: {manifest}")
            sel = _load_manifest_ids(manifest)
            train_samples = [id_to_train[i] for i in sel if i in id_to_train]
            if len(train_samples) != len(sel):
                missing = len(sel) - len(train_samples)
                raise ValueError(f"{missing} manifest ids not found in train pool ({args.dataset}).")

            train_embs = []
            for bb, name in zip(backbones, backbone_names):
                tr_cache = cache_root / name / args.dataset / f"train_{shot}shot_run{run}.npz"
                emb, lab_tr = cached_embeddings(bb, train_samples, device, args.batch_size, tr_cache)
                train_embs.append(emb)
            train_lab = lab_tr

            export_path = (results_dir / f"{shot}_shot_run_{run}_headpreds.npz"
                           if args.export_preds else None)
            export_logits_path = (results_dir / f"{shot}_shot_run_{run}_logits.npz"
                                  if args.export_logits else None)
            metrics = train_eval(
                train_embs, train_lab, test_embs, test_lab, dims, n_classes,
                args.combine, args.kl, args.width, args.epochs, args.lr,
                args.weight_decay, args.kl_warmup, args.batch_size, device, args.seed,
                head=args.head, loss_mode=args.loss_mode, export_path=export_path,
                calib_embs=calib_embs, calib_lab=calib_lab, export_logits_path=export_logits_path,
            )
            row = {
                "dataset": args.dataset, "stage_a_source": stage_label,
                "shot": shot, "run_index": run, "postfix": stage_label,
                "run_tag": f"modern_{args.combine}_{stage_label}_{shot}shot_run{run}_{args.dataset}",
                "backbones": backbone_names, "width": args.width, "head": args.head,
                "ddcobe_combine": args.combine, "ddcobe_kl_lambda": float(args.kl),
                **{k: float(v) for k, v in metrics.items()},
            }
            write_json_atomic(run_json, row)
            print(f"[MODERN] {args.dataset} {shot}-shot run{run}: acc={metrics['test_accuracy']:.4f} "
                  f"ece={metrics['test_ece']:.4f}")

        update_legacy_shot_summary(
            results_dir=results_dir, shot=shot, expected_runs=args.expected_runs,
            summary_prefix={"dataset": args.dataset, "stage_a_source": stage_label, "base_seed": args.seed},
        )


def resolve_repo(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO / path)


if __name__ == "__main__":
    main()
