# src/uncertainty/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def _to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().flatten().cpu().numpy()


@torch.no_grad()
def plot_certainty_correct_vs_incorrect_hist(
    certainty: torch.Tensor,      # [N] values in [0,1] or [0,100]
    incorrect: torch.Tensor,      # [N] float/bool (1=incorrect)
    save_path: str,
    title: Optional[str] = None,
    bins: int = 80,
    certainty_is_percent: bool = False,
    fixed_y_max: Optional[float] = None,
    fixed_y_step: Optional[float] = None,
    x_padding: float = 0.0,
):
    """
    Paper-style certainty distribution:
    Overlaid histograms (density) for correct vs incorrect predictions.
    x-axis: Certainty (%), 0..100
    """
    incorrect_np = _to_numpy_1d(incorrect).astype(bool)

    c = _to_numpy_1d(certainty)
    if not certainty_is_percent:
        c = 100.0 * c  # [0,1] -> [0,100]

    c = np.clip(c, 0.0, 100.0)

    c_correct = c[~incorrect_np]
    c_incorrect = c[incorrect_np]

    plt.figure(figsize=(7, 4))
    edges = np.linspace(0.0, 100.0, bins + 1)

    plt.hist(c_correct, bins=edges, density=True, alpha=0.85, label="Correct Predictions")
    plt.hist(c_incorrect, bins=edges, density=True, alpha=0.55, label="Incorrect Predictions")

    plt.xlabel("Certainty (%)")
    plt.ylabel("Density")
    plt.title(title or "Certainty distribution")
    if x_padding > 0:
        plt.xlim(-x_padding, 100 + x_padding)
    else:
        plt.xlim(0, 100)
    if fixed_y_max is not None:
        plt.ylim(0.0, fixed_y_max)
        if fixed_y_step is not None and fixed_y_step > 0:
            # Include upper endpoint for clean tick labels, e.g. 0.0, 0.2, ..., 0.8
            plt.yticks(np.arange(0.0, fixed_y_max + 1e-9, fixed_y_step))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()




@torch.no_grad()
def plot_risk_coverage_curve(
    uncertainties: Dict[str, torch.Tensor],
    correct: torch.Tensor,      # [N] bool or {0,1} (1=correct)
    save_path: str,
    title: Optional[str] = None,
    n_points: int = 50,
):
    """
    Risk-coverage / selective prediction curve.
    For each method:
      - sort by uncertainty ascending
      - keep the lowest-uncertainty k samples (coverage k/N)
      - compute accuracy on kept subset
    """
    correct_np = _to_numpy_1d(correct).astype(bool)

    plt.figure(figsize=(6, 4))

    for name, u in uncertainties.items():
        u_np = _to_numpy_1d(u)
        order = np.argsort(u_np)  # low uncertainty first
        correct_sorted = correct_np[order]

        N = len(correct_sorted)
        coverages = np.linspace(0.05, 1.0, n_points)
        accs = []
        for cov in coverages:
            k = max(1, int(cov * N))
            accs.append(correct_sorted[:k].mean())

        plt.plot(coverages, accs, label=name)

    plt.xlabel("Coverage (fraction kept, least-uncertain)")
    plt.ylabel("Accuracy on kept subset")
    plt.title(title or "Risk–coverage (selective accuracy)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_risk_coverage_error_curve(
    curve: dict,
    save_path: str,
    title: str = "Risk-Coverage Curve (Predictive Entropy)",
):
    """
    Plot risk (error rate on kept samples) versus coverage.

    Expects curve dict with:
      - coverage: [N] in [0,1]
      - risk: [N] in [0,1]
      - base_error_rate: scalar in [0,1] (optional)
    """
    coverage = 100.0 * curve["coverage"].numpy()
    risk = 100.0 * curve["risk"].numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(coverage, risk, linewidth=2.0, label="Risk")

    if "base_error_rate" in curve:
        base_error = float(curve["base_error_rate"].item()) * 100.0
        plt.axhline(
            base_error,
            linestyle="--",
            linewidth=1.5,
            label=f"Base error ({base_error:.2f}%)",
        )

    plt.xlabel("Coverage (%)")
    plt.ylabel("Risk (%)")
    plt.title(title)
    plt.xlim(0.0, 100.0)
    plt.ylim(0.0, 100.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_reliability_diagram(
    logits: torch.Tensor,       # [N,C]
    labels: torch.Tensor,       # [N]
    save_path: str,
    n_bins: int = 15,
    title: Optional[str] = None,
):
    """
    Reliability diagram (calibration curve).
    Uses max probability as confidence.
    """
    probs = torch.softmax(logits, dim=-1)     # needed: logits -> probabilities
    conf, preds = probs.max(dim=-1)
    correct = (preds == labels).float()

    conf_np = conf.detach().cpu().numpy()
    corr_np = correct.detach().cpu().numpy()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf_np > lo) & (conf_np <= hi)
        if mask.any():
            accs[i] = corr_np[mask].mean()
            confs[i] = conf_np[mask].mean()
        else:
            accs[i] = np.nan
            confs[i] = np.nan

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title or "Reliability diagram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
@torch.no_grad()
def plot_p_u_given_i_curve(sweep: dict, save_path: str, title: str = "P(U|I) vs certainty threshold"):
    from pathlib import Path
    import matplotlib.pyplot as plt

    t = sweep["thresholds"].numpy()
    y = sweep["P_U_given_I"].numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(t, y)
    plt.xlabel("Certainty threshold t")
    plt.ylabel("P(U | Incorrect)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_p_u_given_i_uncertainty_curve(
    sweep: dict,
    save_path: str,
    title: str = "P(U|I) vs uncertainty threshold",
):
    """
    Expects sweep dict with:
      - thresholds_uncertainty: [T] in [0,1]
      - P_U_given_I: [T]
    """
    t = 100.0 * sweep["thresholds_uncertainty"].numpy()
    y = sweep["P_U_given_I"].numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(t, y, linewidth=2.0)
    plt.xlabel("Uncertainty Thresholds (%)")
    plt.ylabel("P_UI")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_p_i_given_u_uncertainty_curve(
    sweep: dict,
    save_path: str,
    title: str = "P(I|U) vs uncertainty threshold",
):
    """
    Expects sweep dict with:
      - thresholds_uncertainty: [T] in [0,1]
      - P_I_given_U: [T]
    """
    t = 100.0 * sweep["thresholds_uncertainty"].numpy()
    y = sweep["P_I_given_U"].numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(t, y, linewidth=2.0)
    plt.xlabel("Uncertainty Thresholds (%)")
    plt.ylabel("P_IU")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_certainty_proportion_curve(sweep: dict, save_path: str, title: str = "Certainty proportion vs threshold"):
    from pathlib import Path
    import matplotlib.pyplot as plt

    t = sweep["thresholds"].numpy()
    y = sweep["certainty_prop"].numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(t, y)
    plt.xlabel("Certainty threshold t")
    plt.ylabel("Certainty proportion P(C)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_certainty_proportion_bar(
    certainty: torch.Tensor,  # [N] in [0,1] or [0,100]
    save_path: str,
    title: str = "Certainty Proportion",
    certainty_is_percent: bool = False,
):
    """
    Bar chart of prediction proportions per certainty decile bucket:
      10, 20, ..., 100 (%)
    Y-axis is percentage of predictions in each bucket.
    """
    c = _to_numpy_1d(certainty)
    if not certainty_is_percent:
        c = 100.0 * c
    c = np.clip(c, 0.0, 100.0)

    # Map each certainty to decile upper-bound labels: 10..100.
    # Example: 91.2 -> 100, 67.0 -> 70, 0.0 -> 10.
    bucket_labels = np.ceil(np.maximum(c, 1e-9) / 10.0) * 10.0
    bucket_labels = np.clip(bucket_labels, 10.0, 100.0).astype(int)

    x = np.arange(10, 101, 10, dtype=int)
    counts = np.array([(bucket_labels == b).sum() for b in x], dtype=float)
    total = counts.sum()
    proportions = (100.0 * counts / total) if total > 0 else np.zeros_like(counts)

    plt.figure(figsize=(6, 4))
    plt.bar(x, proportions, width=7.5)
    plt.xlabel("Certainty (%)")
    plt.ylabel("Proportion of Predictions (%)")
    plt.title(title)
    plt.xlim(-5, 105)
    plt.xticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_calibration_error_by_certainty_bar(
    calib: dict,
    save_path: str,
    title: str = "Calibration Error by Certainty",
):
    """
    Bar chart:
      x-axis: certainty bins in percent (bin upper edge)
      y-axis: calibration error per bin in percent
    """
    x = calib["bin_upper_percent"].numpy()
    y = calib["calibration_error_percent"].numpy()

    plt.figure(figsize=(6, 4))
    plt.bar(x, y, width=7.5)
    plt.xlabel("Certainty (%)")
    plt.ylabel("Calibration Error (%)")
    plt.title(title)
    plt.xlim(-5, 105)
    plt.xticks(np.arange(0, 101, 10))
    plt.ylim(0, 50)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_ece_on_certain_curve(sweep: dict, save_path: str, title: str = "ECE on certain subset vs threshold"):
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np

    t = sweep["thresholds"].numpy()
    y = sweep["ece_on_certain"].numpy()

    # keep NaNs (when no samples are certain) — matplotlib will break the line there
    plt.figure(figsize=(6, 4))
    plt.plot(t, y)
    plt.xlabel("Certainty threshold t")
    plt.ylabel("ECE (on samples with certainty ≥ t)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    
@torch.no_grad()
def plot_roc_curve(
    roc: dict,
    save_path: str,
    title: str = "ROC curve",
):
    from pathlib import Path
    import matplotlib.pyplot as plt

    fpr = roc["fpr"].numpy()
    tpr = roc["tpr"].numpy()

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
