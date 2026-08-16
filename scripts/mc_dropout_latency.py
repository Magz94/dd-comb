#!/usr/bin/env python3
"""End-to-end MC-Dropout inference latency (frozen backbone + T-sample dropout head).

For ONE frozen backbone, this times the *standard* test-time MC-Dropout forward:
a single frozen-backbone feature extraction followed by ``T`` stochastic passes
of the dropout MLP head with dropout kept active (``predict_mc``), at batch sizes
1, 32 and 64. This measures the real ``L^MC ~ L_backbone + T * L_head`` cost
empirically rather than estimating it by multiplication.

It reuses the exact backbone builder, DeepWeeds sample and timing discipline of
``backbone_characterization.py`` (GPU warmup + ``torch.cuda.synchronize`` around
the timed region, averaged over many iterations, per image), and the exact
MC-Dropout head used in the experiments (``baseline_hidden_dim=512``, ``p=0.5``,
``T=50``). Writes ``cost_mcdropout_<backbone>.json`` next to the other
``cost_*.json`` files, mirroring their schema and adding ``mc_dropout_samples``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Reuse the backbone builder + DeepWeeds sample loader from the main cost run.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_characterization import build_any, deepweeds_test_samples, REPO  # noqa: E402

from ddcomb.models.baselines_single import MCDropoutModel  # noqa: E402


@torch.no_grad()
def bench_mc_latency(model, embed_fn, transform, sample_img, device, bs, head, iters=50, warmup=5):
    """End-to-end per-image MC-Dropout latency: one frozen-backbone forward, then the
    dropout head sampled T=head.n_samples times (dropout active). Frozen backbone in
    inference mode; the dropout head is kept in train() so dropout stays on."""
    x = transform(sample_img).unsqueeze(0).repeat(bs, 1, 1, 1).to(device)

    def forward():
        e = embed_fn(model, x)          # backbone features, once
        head.predict_mc(e)              # T stochastic head passes, dropout active

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
    return total_ms / (iters * bs)      # ms per image (backbone + T head passes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset", default="deepweeds_no_bg_mergeval")
    ap.add_argument("--deepweeds-root", default="/datasets/DeepWeeds")
    ap.add_argument("--out-root", default="kshot/backbone_study")
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=512, help="MC-Dropout head width (experiments used 512)")
    ap.add_argument("--dropout-p", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=50, help="T stochastic test-time passes")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = (REPO / args.out_root) if not Path(args.out_root).is_absolute() else Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[mc-lat] backbone={args.backbone} device={device} T={args.samples}")

    model, transform, embed_fn, n_params = build_any(args.backbone)
    model = model.to(device)
    samples = deepweeds_test_samples(Path(args.deepweeds_root))
    sample_img = Image.open(samples[0][0]).convert("RGB")

    with torch.no_grad():
        emb_dim = int(embed_fn(model, transform(sample_img).unsqueeze(0).to(device)).shape[1])

    # Exact MC-Dropout head from the experiments: 1-hidden MLP (512), p=0.5, T=50.
    head = MCDropoutModel(in_dim=emb_dim, n_classes=args.classes, hidden_dim=args.hidden_dim,
                          p=args.dropout_p, n_samples=args.samples).to(device)
    head.train()  # keep dropout active at test time (no BatchNorm, so this only affects dropout)

    lat_b1 = bench_mc_latency(model, embed_fn, transform, sample_img, device, 1, head)
    lat_b32 = bench_mc_latency(model, embed_fn, transform, sample_img, device, 32, head)
    lat_b64 = bench_mc_latency(model, embed_fn, transform, sample_img, device, 64, head)

    cost = {
        "method": "mc_dropout", "backbone": args.backbone, "dataset": args.dataset,
        "n_params": int(n_params), "embedding_dim": emb_dim,
        "mc_dropout_samples": int(args.samples), "mc_dropout_p": float(args.dropout_p),
        "baseline_hidden_dim": int(args.hidden_dim),
        "latency_includes_head": True,
        "latency_ms_per_img_b1": float(lat_b1),
        "latency_ms_per_img_b32": float(lat_b32),
        "latency_ms_per_img_b64": float(lat_b64),
    }
    cost_path = out / f"cost_mcdropout_{args.backbone}.json"
    cost_path.write_text(json.dumps(cost, indent=2))
    print(f"[mc-lat] {args.backbone} (end-to-end, T={args.samples}): params={n_params/1e6:.1f}M "
          f"b1={lat_b1:.3f} b32={lat_b32:.3f} b64={lat_b64:.3f} ms  -> {cost_path}")


if __name__ == "__main__":
    main()
