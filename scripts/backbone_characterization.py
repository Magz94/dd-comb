#!/usr/bin/env python3
"""Backbone characterization for the backbone-selection study.

For ONE backbone (run one per array task), on a dataset's test split, it records:
  * number of parameters (frozen feature extractor),
  * inference latency (ms/image at batch=1 and batch=32, GPU),
  * the test-set embeddings ([N, D]) for the CKA / UMAP redundancy analysis.

Handles both the original torchvision backbones (ResNet18, VGG16, DenseNet121,
EfficientNet-B0, ViT-B/16 -- stock IMAGENET1K_V1 weights) and the modern frozen
foundation backbones (DINOv2, OpenCLIP, ConvNeXt-V2, SigLIP2, MAE-ViT) via the
``modern_backbones`` registry. Outputs go to ``<out_root>/`` as
``emb_<name>.npz`` (emb, y_true) and ``cost_<name>.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ddcomb.models.modern_backbones import build_backbone
from ddcomb.training.single_backbone import DeepWeedsCSVDataset, resolve_deepweeds_csv

REPO = Path(__file__).resolve().parents[1]
# CLI name -> EmbeddingModel registry name for the original torchvision backbones.
ORIG = {"resnet18": "ResNet18", "vgg16": "VGG16", "densenet121": "DenseNet121",
        "efficientnet_b0": "EfficientNetB0", "vit_b16": "ViT-B16"}
_BASE_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def build_any(name: str):
    """Return (model, transform, embed_fn, n_params) for original or modern backbone."""
    if name in ORIG:
        from ddcomb.models.embedding_models import prepare_embedding_models
        em = prepare_embedding_models(load_path=None, pretrained=True, num_classes=2)[ORIG[name]]
        model = em.eval()
        for p in model.parameters():
            p.requires_grad = False
        n = sum(p.numel() for p in model.feature_extractor.parameters())
        return model, _BASE_TF, (lambda m, x: m(x, return_embedding=True)), n
    bb = build_backbone(name)
    n = sum(p.numel() for p in bb.model.parameters())
    return bb.model, bb.transform, (lambda m, x: bb.embed(x)), n


def deepweeds_test_samples(root: Path):
    images_dir = root / "images"
    labels_dir = root / "labels"
    te = DeepWeedsCSVDataset(images_dir=images_dir,
                             csv_path=resolve_deepweeds_csv(labels_dir, "subset0", "test"),
                             transform=None, drop_labels={8})
    return [(te._resolve_image_path(fn), int(lab)) for fn, lab in te.samples]


@torch.no_grad()
@torch.no_grad()
def bench_latency(model, embed_fn, transform, sample_img, device, bs, head=None, iters=50, warmup=5):
    """End-to-end per-image latency: backbone forward, and (if head is given) the head
    forward on the resulting embedding. Frozen backbones are run in inference mode."""
    x = transform(sample_img).unsqueeze(0).repeat(bs, 1, 1, 1).to(device)

    def forward():
        e = embed_fn(model, x)
        if head is not None:
            head(e)

    for _ in range(warmup):
        forward()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        forward()
    if device == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / (iters * bs)  # ms per image (backbone + head)


@torch.no_grad()
def extract(model, embed_fn, transform, samples, device, batch_size):
    embs, labels, buf, lab = [], [], [], []
    def flush():
        if buf:
            x = torch.stack(buf).to(device)
            embs.append(embed_fn(model, x).float().detach().cpu())
            labels.extend(lab); buf.clear(); lab.clear()
    for path, y in samples:
        buf.append(transform(Image.open(path).convert("RGB"))); lab.append(y)
        if len(buf) >= batch_size:
            flush()
    flush()
    return torch.cat(embs).numpy(), np.array(labels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset", default="deepweeds_no_bg_mergeval")
    ap.add_argument("--deepweeds-root", default="/datasets/DeepWeeds")
    ap.add_argument("--out-root", default="kshot/backbone_study")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--classes", type=int, default=8,
                    help="number of classes for the head used in the end-to-end latency (DeepWeeds = 8)")
    ap.add_argument("--latency-only", action="store_true",
                    help="re-measure end-to-end latency (b=1, 32, 64) and update cost_<name>.json only, "
                         "without re-extracting or overwriting emb_<name>.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = (REPO / args.out_root) if not Path(args.out_root).is_absolute() else Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[char] backbone={args.backbone} device={device}")

    model, transform, embed_fn, n_params = build_any(args.backbone)
    model = model.to(device)
    samples = deepweeds_test_samples(Path(args.deepweeds_root))
    sample_img = Image.open(samples[0][0]).convert("RGB")

    # End-to-end latency = frozen backbone + the same MLP head (native width) used by every model.
    from ddcomb.models.ddcobe_ensemble import BackboneEvidenceHead
    with torch.no_grad():
        emb_dim = int(embed_fn(model, transform(sample_img).unsqueeze(0).to(device)).shape[1])
    head = BackboneEvidenceHead(in_dim=emb_dim, n_classes=args.classes,
                                hidden_dim=emb_dim, head_type="mlp").to(device).eval()

    lat_b1 = bench_latency(model, embed_fn, transform, sample_img, device, 1, head=head)
    lat_b32 = bench_latency(model, embed_fn, transform, sample_img, device, 32, head=head)
    lat_b64 = bench_latency(model, embed_fn, transform, sample_img, device, 64, head=head)
    cost_path = out / f"cost_{args.backbone}.json"

    if args.latency_only:
        # Update latency fields only; keep the cached embeddings untouched.
        prev = json.loads(cost_path.read_text()) if cost_path.exists() else {}
        cost = {**prev, "backbone": args.backbone, "dataset": args.dataset,
                "n_params": int(n_params), "embedding_dim": emb_dim, "latency_includes_head": True,
                "latency_ms_per_img_b1": float(lat_b1), "latency_ms_per_img_b32": float(lat_b32),
                "latency_ms_per_img_b64": float(lat_b64)}
        cost_path.write_text(json.dumps(cost, indent=2))
        print(f"[char] {args.backbone} (latency-only, end-to-end): params={n_params/1e6:.1f}M "
              f"b1={lat_b1:.3f} b32={lat_b32:.3f} b64={lat_b64:.3f} ms  -> {cost_path}")
        return

    emb, y = extract(model, embed_fn, transform, samples, device, args.batch_size)
    np.savez(out / f"emb_{args.backbone}.npz", emb=emb, y_true=y)
    cost = {"backbone": args.backbone, "dataset": args.dataset, "n_params": int(n_params),
            "embedding_dim": int(emb.shape[1]), "n_test": int(emb.shape[0]), "latency_includes_head": True,
            "latency_ms_per_img_b1": float(lat_b1), "latency_ms_per_img_b32": float(lat_b32),
            "latency_ms_per_img_b64": float(lat_b64)}
    cost_path.write_text(json.dumps(cost, indent=2))
    print(f"[char] {args.backbone} (end-to-end): params={n_params/1e6:.1f}M dim={emb.shape[1]} "
          f"b1={lat_b1:.3f} b32={lat_b32:.3f} b64={lat_b64:.3f} ms  -> {out}")


if __name__ == "__main__":
    main()
