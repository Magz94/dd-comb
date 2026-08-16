#!/usr/bin/env python3
"""Backbone-selection study analysis (local, post-sync).

Consumes the artefacts produced on the cluster by
``scripts/backbone_characterization.py`` (per-backbone params/latency/embeddings)
and the single-backbone test predictions (5 original via the single-backbone
pipeline, 5 modern via ``run_modern_backbone_experiment.py --export-preds``) and
produces the professor-requested backbone-selection study for DeepWeeds:

  1. COST -- system-level FIRST (DINOv2+linear probe vs the whole DD-CoBE
     5-backbone ensemble: total #params and total inference latency), then a
     per-backbone cost table (#params, embedding dim, ms/image @ b=1 and b=32).
  2. REDUNDANCY -- linear-CKA similarity matrix between the frozen latent spaces
     of all available backbones, plus a 2D "map of backbones" (MDS on 1-CKA).
  3. DISAGREEMENT -- pairwise prediction-disagreement matrix and the oracle
     (at-least-one-correct) coverage across the pool, at a chosen shot.
  4. SYNTHESIS -- cost vs. diversity-from-DINOv2 vs. accuracy, to surface
     candidates that are both cheap-ish and complementary to DINOv2.

Every section is guarded: whatever subset of inputs is present is analysed; the
rest is skipped with a note, so the script is runnable as data trickles back.

Outputs (under ``--out-root``, default kshot/backbone_study/analysis):
  cost_per_backbone.csv, cost_system_level.json,
  cka_matrix.csv, disagreement_<shot>shot.csv, oracle_<shot>shot.json,
  fig_cost_system.(pdf|png), fig_cost_per_backbone.(pdf|png),
  fig_cka_matrix.(pdf|png), fig_backbone_map.(pdf|png),
  fig_disagreement_<shot>shot.(pdf|png), fig_synthesis.(pdf|png),
  backbone_study_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

# Canonical pool: 5 required original + widened modern candidates.
ORIGINAL = ["resnet18", "vgg16", "densenet121", "efficientnet_b0", "vit_b16"]
MODERN = ["dinov2_vitb14", "openclip_vitb16", "convnextv2_tiny", "siglip2_base", "mae_vitb16"]
POOL = ORIGINAL + MODERN
PRETTY = {
    "resnet18": "ResNet-18", "vgg16": "VGG-16", "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0", "vit_b16": "ViT-B/16",
    "dinov2_vitb14": "DINOv2 ViT-B/14", "openclip_vitb16": "OpenCLIP ViT-B/16",
    "convnextv2_tiny": "ConvNeXt-V2-T", "siglip2_base": "SigLIP2-B", "mae_vitb16": "MAE ViT-B/16",
}
ORIG_DIRNAME = dict(zip(ORIGINAL, ORIGINAL))  # single_backbone uses same short names


# ----------------------------------------------------------------- loaders
def load_costs(study_root: Path) -> Dict[str, dict]:
    out = {}
    for name in POOL:
        f = study_root / f"cost_{name}.json"
        if f.is_file():
            out[name] = json.loads(f.read_text())
    return out


def load_embeddings(study_root: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name in POOL:
        f = study_root / f"emb_{name}.npz"
        if f.is_file():
            d = np.load(f)
            out[name] = (d["emb"].astype(np.float64), d["y_true"].astype(np.int64))
    return out


def _orig_pred_file(preds_root: Path, dataset: str, bb: str, shot: int, run: int) -> Path:
    return preds_root / dataset / bb / "head_only_basic" / f"{shot}_shot_run_{run}_predictions.npz"


def _modern_pred_file(preds_root: Path, dataset: str, bb: str, shot: int, run: int) -> Path:
    return preds_root / bb / dataset / f"{shot}_shot_run_{run}_headpreds.npz"


def load_predictions(orig_root: Path, modern_root: Path, dataset: str, shot: int,
                     runs: range) -> Tuple[Dict[str, Dict[int, np.ndarray]], Optional[np.ndarray]]:
    """Return {backbone: {run: pred_argmax[N]}} and a reference y_true (or None)."""
    preds: Dict[str, Dict[int, np.ndarray]] = {}
    y_ref = None
    for bb in ORIGINAL:
        for r in runs:
            f = _orig_pred_file(orig_root, dataset, bb, shot, r)
            if not f.is_file():
                continue
            d = np.load(f)
            pred = d["y_prob"].argmax(1).astype(np.int64)
            preds.setdefault(bb, {})[r] = pred
            if y_ref is None:
                y_ref = d["y_true"].astype(np.int64)
    for bb in MODERN:
        for r in runs:
            f = _modern_pred_file(modern_root, dataset, bb, shot, r)
            if not f.is_file():
                continue
            d = np.load(f)
            preds.setdefault(bb, {})[r] = d["head_pred"].astype(np.int64)
            if y_ref is None:
                y_ref = d["y_true"].astype(np.int64)
    return preds, y_ref


# ----------------------------------------------------------------- analyses
def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two [n, d] representations (column-centered)."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xty = x.T @ y
    hsic_xy = float(np.sum(xty * xty))
    xtx = x.T @ x
    yty = y.T @ y
    denom = float(np.sqrt(np.sum(xtx * xtx) * np.sum(yty * yty)))
    return hsic_xy / denom if denom > 0 else 0.0


def cka_matrix(embs: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[List[str], np.ndarray]:
    names = [n for n in POOL if n in embs]
    # Align on common sample count (same test set, same order assumed).
    n_min = min(embs[n][0].shape[0] for n in names)
    M = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            xi = embs[names[i]][0][:n_min]
            xj = embs[names[j]][0][:n_min]
            c = linear_cka(xi, xj)
            M[i, j] = M[j, i] = c
    return names, M


def pairwise_disagreement(preds: Dict[str, Dict[int, np.ndarray]],
                          y_ref: np.ndarray) -> Tuple[List[str], np.ndarray, Dict[str, float]]:
    names = [n for n in POOL if n in preds and preds[n]]
    n = len(names)
    D = np.zeros((n, n))
    acc = {}
    # accuracy per backbone (avg over its runs)
    for nm in names:
        accs = [float((p == y_ref[:len(p)]).mean()) for p in preds[nm].values()]
        acc[nm] = float(np.mean(accs))
    for i in range(n):
        for j in range(i + 1, n):
            common = sorted(set(preds[names[i]]) & set(preds[names[j]]))
            vals = []
            for r in common:
                a, b = preds[names[i]][r], preds[names[j]][r]
                m = min(len(a), len(b))
                vals.append(float((a[:m] != b[:m]).mean()))
            d = float(np.mean(vals)) if vals else np.nan
            D[i, j] = D[j, i] = d
    return names, D, acc


def oracle_coverage(preds: Dict[str, Dict[int, np.ndarray]], y_ref: np.ndarray,
                    subset: List[str]) -> Optional[float]:
    subset = [s for s in subset if s in preds and preds[s]]
    if not subset:
        return None
    common = None
    for s in subset:
        rs = set(preds[s])
        common = rs if common is None else (common & rs)
    common = sorted(common or [])
    if not common:
        return None
    per_run = []
    for r in common:
        arrs = [preds[s][r] for s in subset]
        m = min(min(len(a) for a in arrs), len(y_ref))
        y = y_ref[:m]
        any_correct = np.zeros(m, dtype=bool)
        for a in arrs:
            any_correct |= (a[:m] == y)
        per_run.append(float(any_correct.mean()))
    return float(np.mean(per_run))


# ----------------------------------------------------------------- figures
_BG = "white"


def fig_cost_system(sys_cost: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ["DINOv2\n+ linear probe", "DD-CoBE\n(5 backbones)"]
    colors = ["#4C78A8", "#E45756"]
    params = [sys_cost["dinov2_linear"]["params_M"], sys_cost["ddcobe_5bb"]["params_M"]]
    lat = [sys_cost["dinov2_linear"]["latency_ms_b32"], sys_cost["ddcobe_5bb"]["latency_ms_b32"]]
    for ax, vals, title, unit in (
        (axes[0], params, "Total parameters", "M params"),
        (axes[1], lat, "Total inference latency (batch=32)", "ms / image"),
    ):
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
    fig.suptitle("System-level cost: single foundation backbone vs. full DD-CoBE ensemble",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out)


def fig_cost_per_backbone(costs: Dict[str, dict], out: Path):
    names = [n for n in POOL if n in costs]
    params = [costs[n]["n_params"] / 1e6 for n in names]
    lat = [costs[n]["latency_ms_per_img_b32"] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = ["#54A24B" if n in ORIGINAL else "#B279A2" for n in names]
    ax.scatter(params, lat, c=cols, s=90, edgecolor="black", zorder=3)
    for n, x, y in zip(names, params, lat):
        ax.annotate(PRETTY[n], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Latency (ms / image, batch=32)")
    ax.set_title("Per-backbone cost: parameters vs. inference latency", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#54A24B", label="Original (required 5)"),
                       Patch(color="#B279A2", label="Modern candidates")], loc="best")
    fig.tight_layout()
    _save(fig, out)


def fig_cka(names: List[str], M: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    labs = [PRETTY[n] for n in names]
    ax.set_xticks(range(len(names))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(labs, fontsize=8.5)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="linear CKA (1 = identical)")
    ax.set_title("Latent-space redundancy (linear CKA)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


def fig_backbone_map(names: List[str], M: np.ndarray, out: Path):
    try:
        from sklearn.manifold import MDS
    except Exception:
        print("[map] sklearn unavailable, skipping backbone map")
        return
    dist = 1.0 - M
    np.fill_diagonal(dist, 0.0)
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                 normalized_stress="auto").fit_transform(dist)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cols = ["#54A24B" if n in ORIGINAL else "#B279A2" for n in names]
    ax.scatter(coords[:, 0], coords[:, 1], c=cols, s=120, edgecolor="black", zorder=3)
    for n, (x, y) in zip(names, coords):
        ax.annotate(PRETTY[n], (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_title("Map of backbones (MDS on 1 − CKA): far apart = complementary",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2"); ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    handles = []
    if any(n in ORIGINAL for n in names):
        handles.append(Patch(color="#54A24B", label="Original (required 5)"))
    if any(n not in ORIGINAL for n in names):
        handles.append(Patch(color="#B279A2", label="Modern candidates"))
    if len(handles) > 1:
        ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    _save(fig, out)


def fig_disagreement(names: List[str], D: np.ndarray, shot: int, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(D, cmap="magma")
    labs = [PRETTY[n] for n in names]
    ax.set_xticks(range(len(names))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(labs, fontsize=8.5)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(D[i, j]):
                ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center",
                        color="white" if D[i, j] < D.max() * 0.6 else "black", fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="prediction disagreement rate")
    ax.set_title(f"Pairwise prediction disagreement ({shot}-shot)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


def fig_synthesis(costs, acc, M, names_cka, ref="dinov2_vitb14", out: Path = None):
    """cost (x) vs complementarity-to-DINOv2 (y=1-CKA) vs accuracy (size/color)."""
    if ref not in names_cka:
        print("[synth] DINOv2 not in CKA matrix, skipping synthesis")
        return
    ridx = names_cka.index(ref)
    pts = []
    for n in names_cka:
        if n == ref or n not in costs or n not in acc:
            continue
        j = names_cka.index(n)
        pts.append((n, costs[n]["n_params"] / 1e6, 1.0 - M[ridx, j], acc[n]))
    if not pts:
        print("[synth] no points for synthesis")
        return
    fig, ax = plt.subplots(figsize=(8.5, 6))
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]; accs = [p[3] for p in pts]
    sc = ax.scatter(xs, ys, c=accs, s=[200 * a for a in accs], cmap="plasma",
                    edgecolor="black", zorder=3)
    for n, x, y, a in pts:
        ax.annotate(PRETTY[n], (x, y), textcoords="offset points", xytext=(7, 4), fontsize=8.5)
    fig.colorbar(sc, ax=ax, label="single-backbone accuracy")
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Complementarity to DINOv2  (1 − CKA)")
    ax.set_title("Synthesis: which candidates are cheap, accurate AND complementary to DINOv2",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=_BG)
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"[fig] {out.with_suffix('.png')}")


def leave_one_out(preds, y_ref, group) -> Tuple[List[str], Optional[float], Dict[str, float]]:
    """Oracle of the full group, and oracle when each member is removed."""
    members = [g for g in group if g in preds and preds[g]]
    full = oracle_coverage(preds, y_ref, members)
    without = {g: oracle_coverage(preds, y_ref, [x for x in members if x != g]) for g in members}
    return members, full, without


def fig_leave_one_out(members, full, without, acc, best_single, shot, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labs = [PRETTY[m] for m in members]
    vals = [without[m] for m in members]
    bars = ax.bar(labs, vals, color="#4C78A8", width=0.62, zorder=3)
    ax.axhline(full, color="#E45756", lw=2, ls="--", zorder=4,
               label=f"full set oracle = {full:.3f}")
    if best_single is not None:
        ax.axhline(best_single, color="#54A24B", lw=2, ls=":", zorder=4,
                   label=f"best single backbone = {best_single:.3f}")
    for b, v, m in zip(bars, vals, members):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}\nΔ{v-full:+.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("oracle coverage (≥1 correct)")
    lo = min(min(vals), best_single or full) - 0.03
    ax.set_ylim(lo, max(full, max(vals)) + 0.06)
    ax.set_title(f"Leave-one-out oracle ({shot}-shot): how much each backbone adds",
                 fontsize=11.5, fontweight="bold")
    ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=_BG)
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"[fig] {out.with_suffix('.png')}")


GROUPS = {"original5": ORIGINAL, "modern5": MODERN, "all": POOL,
          "dino_cnx": ["dinov2_vitb14", "convnextv2_tiny"]}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", default="kshot/backbone_study")
    ap.add_argument("--orig-preds-root", default="kshot/results_single_backbone_original_preds")
    ap.add_argument("--modern-preds-root", default="kshot/results_modern_singles_preds")
    ap.add_argument("--dataset", default="deepweeds_no_bg_mergeval")
    ap.add_argument("--group", default="all", choices=list(GROUPS),
                    help="which backbone set to analyse")
    ap.add_argument("--shots", default="1,5,10,20")
    ap.add_argument("--runs", default="1-20")
    ap.add_argument("--out-root", default="kshot/backbone_study/report")
    args = ap.parse_args()

    def rp(p):
        q = Path(p)
        return q if q.is_absolute() else (REPO / q)

    study_root = rp(args.study_root)
    group = GROUPS[args.group]
    out_root = rp(args.out_root) / args.group
    (out_root / "cost").mkdir(parents=True, exist_ok=True)
    (out_root / "redundancy").mkdir(parents=True, exist_ok=True)
    a, b = (args.runs.split("-") + [args.runs])[:2] if "-" in args.runs else (args.runs, args.runs)
    runs = range(int(a), int(b) + 1)
    shots = [int(s) for s in args.shots.split(",")]
    import csv

    rep = [f"# Backbone study — group: {args.group} ({len(group)} backbones)\n",
           f"Dataset DeepWeeds; runs {args.runs}.\n\n"]

    # ---- COST (shot-independent) -----------------------------------------
    costs_all = load_costs(study_root)
    costs = {n: costs_all[n] for n in group if n in costs_all}
    if costs:
        with open(out_root / "cost" / "cost_per_backbone.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["backbone", "group", "n_params_M", "embedding_dim",
                        "latency_ms_b1", "latency_ms_b32"])
            for n in group:
                if n in costs:
                    c = costs[n]
                    w.writerow([PRETTY[n], "original" if n in ORIGINAL else "modern",
                                f"{c['n_params']/1e6:.2f}", c["embedding_dim"],
                                f"{c['latency_ms_per_img_b1']:.3f}", f"{c['latency_ms_per_img_b32']:.4f}"])
        fig_cost_per_backbone(costs, out_root / "cost" / "fig_cost_per_backbone")
        # system-level reference always uses the canonical DD-CoBE(5 original) vs DINOv2.
        sys_cost = {}
        if all(n in costs_all for n in ORIGINAL):
            sys_cost["ddcobe_5bb"] = {
                "params_M": sum(costs_all[n]["n_params"] for n in ORIGINAL) / 1e6,
                "latency_ms_b32": sum(costs_all[n]["latency_ms_per_img_b32"] for n in ORIGINAL)}
        if "dinov2_vitb14" in costs_all:
            c = costs_all["dinov2_vitb14"]
            sys_cost["dinov2_linear"] = {"params_M": c["n_params"] / 1e6,
                                         "latency_ms_b32": c["latency_ms_per_img_b32"]}
        (out_root / "cost" / "cost_system_level.json").write_text(json.dumps(sys_cost, indent=2))
        if "ddcobe_5bb" in sys_cost and "dinov2_linear" in sys_cost:
            fig_cost_system(sys_cost, out_root / "cost" / "fig_cost_system")
            d, e = sys_cost["dinov2_linear"], sys_cost["ddcobe_5bb"]
            rep.append("## Cost (system-level first)\n")
            rep.append(f"- DINOv2 + linear probe: **{d['params_M']:.1f}M** params, "
                       f"**{d['latency_ms_b32']:.2f} ms/img** (b=32).\n")
            rep.append(f"- Full DD-CoBE (5 original): **{e['params_M']:.1f}M** params, "
                       f"**{e['latency_ms_b32']:.2f} ms/img** — {e['params_M']/d['params_M']:.1f}× params, "
                       f"{e['latency_ms_b32']/d['latency_ms_b32']:.1f}× latency.\n\n")
        rep.append("Per-backbone cost: `cost/cost_per_backbone.csv`.\n\n")

    # ---- REDUNDANCY / CKA (shot-independent) -----------------------------
    embs_all = load_embeddings(study_root)
    embs = {n: embs_all[n] for n in group if n in embs_all}
    names_cka, M = [], np.zeros((0, 0))
    if len(embs) >= 2:
        names_cka, M = cka_matrix(embs)
        np.savetxt(out_root / "redundancy" / "cka_matrix.csv", M, delimiter=",",
                   header=",".join(names_cka), comments="")
        fig_cka(names_cka, M, out_root / "redundancy" / "fig_cka_matrix")
        fig_backbone_map(names_cka, M, out_root / "redundancy" / "fig_backbone_map")
        rep.append("## Latent-space redundancy (linear CKA)\n"
                   "`redundancy/`: matrix + map. Lower CKA = more complementary.\n\n")

    # ---- PER-SHOT: disagreement, oracle, leave-one-out -------------------
    rep.append("## Per-shot diversity (disagreement, oracle, leave-one-out)\n")
    for shot in shots:
        sdir = out_root / f"shot_{shot}"
        sdir.mkdir(parents=True, exist_ok=True)
        preds_all, y_ref = load_predictions(rp(args.orig_preds_root), rp(args.modern_preds_root),
                                            args.dataset, shot, runs)
        preds = {n: preds_all[n] for n in group if n in preds_all and preds_all[n]}
        if not preds or y_ref is None:
            rep.append(f"- {shot}-shot: no predictions found.\n")
            continue
        names_d, D, acc = pairwise_disagreement(preds, y_ref)
        np.savetxt(sdir / "disagreement.csv", D, delimiter=",", header=",".join(names_d), comments="")
        fig_disagreement(names_d, D, shot, sdir / "fig_disagreement")
        members, full, without = leave_one_out(preds, y_ref, group)
        best_single = max(acc.values()) if acc else None
        fig_leave_one_out(members, full, without, acc, best_single, shot, sdir / "fig_leave_one_out")
        oracle = {"group_oracle": full, "best_single_acc": best_single,
                  "per_backbone_acc": acc, "leave_one_out_oracle": without,
                  "mean_disagreement": float(np.nanmean(D[np.triu_indices(len(names_d), 1)]))
                  if len(names_d) > 1 else None}
        (sdir / "oracle.json").write_text(json.dumps(oracle, indent=2))
        rep.append(f"- **{shot}-shot**: oracle={_fmt(full)}, best single={_fmt(best_single)}, "
                   f"mean disagreement={_fmt(oracle['mean_disagreement'])} "
                   f"(`shot_{shot}/`).\n")

    (out_root / "REPORT.md").write_text("".join(rep))
    print(f"\n[done] {args.group}: {out_root / 'REPORT.md'}")


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "n/a"


if __name__ == "__main__":
    main()
