# src/uncertainty/metrics.py
"""
Uncertainty & calibration metrics (Torch-first).

Implements:
- NLL (cross-entropy)
- ECE (Expected Calibration Error)
- Binary AUROC (e.g., error detection AUROC)

All functions are safe to call in eval with torch.no_grad().
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import math
from typing import Optional, Dict

@torch.no_grad()
def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Negative log likelihood (mean cross entropy).
    logits: [N, C]
    labels: [N]
    """
    return F.cross_entropy(logits, labels, reduction="mean").item()


@torch.no_grad()
def ece_from_logits(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) using confidence binning.
    logits: [N, C]
    labels: [N]
    """
    probs = torch.softmax(logits, dim=-1)  # [N, C]
    conf, preds = probs.max(dim=-1)        # [N]
    correct = (preds == labels).float()    # [N]

    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            bin_prop = mask.float().mean()
            ece += torch.abs(bin_acc - bin_conf) * bin_prop

    return ece.item()


@torch.no_grad()
def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    AUROC for binary labels using the Mann–Whitney U statistic.
    scores: [N] higher -> more positive
    labels: [N] values {0,1}
    """
    scores = scores.detach().flatten()
    labels = labels.detach().flatten().float()

    pos = labels == 1
    neg = labels == 0
    n_pos = pos.sum().item()
    n_neg = neg.sum().item()

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # ranks with tie handling (average rank for ties)
    sorted_scores, order = torch.sort(scores)
    sorted_labels = labels[order]

    ranks = torch.zeros_like(sorted_scores)
    i = 0
    rank = 1.0
    N = scores.numel()

    while i < N:
        j = i
        while j + 1 < N and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1

        # average rank for tie group i..j
        avg_rank = (rank + (rank + (j - i))) / 2.0
        ranks[i:j + 1] = avg_rank

        rank += (j - i + 1)
        i = j + 1

    # sum ranks of positives
    sum_ranks_pos = ranks[sorted_labels == 1].sum()
    U = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = U / (n_pos * n_neg)
    return float(auc)
    
@torch.no_grad()
def uncertainty_predictive_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Uncertainty score: predictive entropy of the softmax distribution.

    H(p) = -sum_c p_c * log(p_c)

    logits: [N, C]
    returns: [N] (higher = more uncertain)
    """
    probs = torch.softmax(logits, dim=-1)      # [N, C]
    p = probs.clamp_min(eps)                   # avoid log(0)
    entropy = -(p * p.log()).sum(dim=-1)       # [N]
    return entropy


@torch.no_grad()
def risk_coverage_curve_from_uncertainty(
    uncertainty: torch.Tensor,
    incorrect: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Build risk-coverage curve for selective prediction.

    Inputs:
      uncertainty: [N] (higher = more uncertain)
      incorrect:   [N] bool/{0,1} (1 means incorrect prediction)

    Procedure:
      - sort by uncertainty ascending (keep least-uncertain first)
      - for each prefix k (coverage k/N), compute risk_k = (#errors in prefix)/k
    """
    u = uncertainty.detach().flatten().float()
    err = incorrect.detach().flatten().bool()
    if u.numel() != err.numel():
        raise ValueError(f"uncertainty and incorrect must have same N, got {u.numel()} vs {err.numel()}")
    if u.numel() == 0:
        raise ValueError("uncertainty and incorrect must be non-empty")

    order = torch.argsort(u, descending=False)
    err_sorted = err[order].float()

    n = err_sorted.numel()
    k = torch.arange(1, n + 1, device=err_sorted.device, dtype=torch.float32)
    cumulative_err = torch.cumsum(err_sorted, dim=0)
    risk = cumulative_err / k
    coverage = k / float(n)
    base_error_rate = err_sorted.mean()

    return {
        "coverage": coverage.detach().cpu(),
        "risk": risk.detach().cpu(),
        "base_error_rate": base_error_rate.detach().cpu(),
    }


@torch.no_grad()
def aurc_eaurc_from_risk_coverage(curve: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute AURC and E-AURC from a risk-coverage curve.

    AURC  = mean risk over coverage steps.
    E-AURC = AURC - AURC*, where:
      AURC* = e + (1 - e) * log(1 - e), and e is base error rate.
    """
    risk = curve["risk"].detach().flatten().float()
    if risk.numel() == 0:
        raise ValueError("curve['risk'] must be non-empty")

    e = float(curve["base_error_rate"].item())
    aurc = float(risk.mean().item())
    one_minus_e = max(1.0 - e, 1e-12)
    aurc_opt = float(e + one_minus_e * math.log(one_minus_e))
    eaurc = float(aurc - aurc_opt)

    return {
        "aurc": aurc,
        "eaurc": eaurc,
        "aurc_opt": aurc_opt,
        "base_error_rate": e,
    }


@torch.no_grad()
def risk_coverage_curve_from_predictive_entropy_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Risk-coverage curve using predictive entropy as uncertainty score.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,C], got shape {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be [N], got shape {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels must share batch size N, got {logits.shape[0]} vs {labels.shape[0]}"
        )

    uncertainty = uncertainty_predictive_entropy_from_logits(logits, eps=eps)
    incorrect = logits.argmax(dim=1).ne(labels)
    return risk_coverage_curve_from_uncertainty(uncertainty=uncertainty, incorrect=incorrect)


@torch.no_grad()
def aurc_eaurc_from_predictive_entropy_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    AURC and E-AURC using predictive entropy as uncertainty score.
    """
    curve = risk_coverage_curve_from_predictive_entropy_logits(logits=logits, labels=labels, eps=eps)
    return aurc_eaurc_from_risk_coverage(curve)


@torch.no_grad()
def certainty_from_predictive_entropy_logits(
    logits: torch.Tensor,
    num_classes: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns certainty in [0,1] from predictive entropy:
        certainty = 1 - H(p)/log(C)
    logits: [N,C]
    """
    probs = torch.softmax(logits, dim=-1)
    p = probs.clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1)  # [N]
    cert = 1.0 - entropy / (math.log(num_classes) + 1e-12)
    return cert.clamp(0.0, 1.0)


@torch.no_grad()
def accuracy_vs_uncertainty_metrics(
    certainty: torch.Tensor,     # [N] in [0,1]
    correct: torch.Tensor,       # [N] bool OR {0,1}
    thresholds: torch.Tensor | None = None,  # e.g. torch.linspace(0,1,101)
) -> dict:
    """
    Computes paper metrics over thresholds:
      P(A|C), P(U|I), P(I|U), AvU
    certainty: higher = more certain
    correct: True/1 if prediction correct
    """
    cert = certainty.detach().flatten().float()
    corr = correct.detach().flatten().bool()

    if thresholds is None:
        thresholds = torch.linspace(0.0, 1.0, 101, device=cert.device)

    pac_list = []
    pui_list = []
    piu_list = []
    avu_list = []

    for t in thresholds:
        certain = cert >= t
        uncertain = ~certain

        accurate = corr
        inaccurate = ~corr

        nac = (accurate & certain).sum().float()
        nau = (accurate & uncertain).sum().float()
        nic = (inaccurate & certain).sum().float()
        niu = (inaccurate & uncertain).sum().float()

        # P(A|C) = nac / (nac + nic)
        denom_pac = nac + nic
        pac = nac / (denom_pac + 1e-12)

        # P(U|I) = niu / (nic + niu)
        denom_pui = nic + niu
        pui = niu / (denom_pui + 1e-12)

        # P(I|U) = niu / (nau + niu)
        denom_piu = nau + niu
        piu = niu / (denom_piu + 1e-12)

        # AvU = (nac + niu) / N
        N = nac + nau + nic + niu
        avu = (nac + niu) / (N + 1e-12)

        pac_list.append(pac)
        pui_list.append(pui)
        piu_list.append(piu)
        avu_list.append(avu)

    return {
        "thresholds": thresholds.detach().cpu(),
        "P_A_given_C": torch.stack(pac_list).detach().cpu(),  # [T]
        "P_U_given_I": torch.stack(pui_list).detach().cpu(),  # [T]
        "P_I_given_U": torch.stack(piu_list).detach().cpu(),  # [T]
        "AvU": torch.stack(avu_list).detach().cpu(),          # [T]
    }
@torch.no_grad()
def certainty_threshold_sweep(
    certainty: torch.Tensor,         # [N] in [0,1] higher=more certain
    logits: torch.Tensor,            # [N,C]
    labels: torch.Tensor,            # [N]
    n_thresholds: int = 101,
    ece_bins: int = 15,
) -> dict:
    """
    Sweep threshold t in [0,1] and compute:
      - P_U_given_I(t)   = P(U | Incorrect)
      - certainty_prop(t)= P(C) = fraction certain
      - ece_on_certain(t)= ECE computed on subset certainty>=t

    Returns CPU tensors:
      thresholds [T], P_U_given_I [T], certainty_prop [T], ece_on_certain [T]
    """
    cert = certainty.detach().flatten().float()
    preds = logits.argmax(dim=1)
    correct = (preds == labels).bool()
    incorrect = ~correct

    thresholds = torch.linspace(0.0, 1.0, n_thresholds, device=logits.device)

    pui_list = []
    prop_list = []
    ece_list = []

    for t in thresholds:
        certain = cert >= t
        uncertain = ~certain

        # P(U|I) = #incorrect & uncertain / #incorrect
        num = (incorrect & uncertain).sum().float()
        den = incorrect.sum().float()
        pui = num / (den + 1e-12)
        pui_list.append(pui)

        # certainty proportion P(C)
        prop_list.append(certain.float().mean())

        # ECE on "certain" subset
        if certain.any():
            ece_val = ece_from_logits(logits[certain], labels[certain], n_bins=ece_bins)
            ece_list.append(torch.tensor(ece_val, device=logits.device))
        else:
            ece_list.append(torch.tensor(float("nan"), device=logits.device))

    return {
        "thresholds": thresholds.detach().cpu(),
        "P_U_given_I": torch.stack(pui_list).detach().cpu(),
        "certainty_prop": torch.stack(prop_list).detach().cpu(),
        "ece_on_certain": torch.stack(ece_list).detach().cpu(),
    }


@torch.no_grad()
def p_u_given_i_uncertainty_sweep_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 101,
    eps: float = 1e-12,
) -> dict:
    """
    Sweep uncertainty thresholds u_t in [0,1] and compute:
      P(U|I)(u_t) = # (incorrect AND uncertain) / # incorrect

    Uncertainty is normalized predictive entropy:
      u = H(p) / log(C), where p = softmax(logits)

    Returns CPU tensors:
      - thresholds_uncertainty: [T] in [0,1]
      - P_U_given_I: [T]
      - P_I_given_U: [T]
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,C], got shape {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be [N], got shape {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels must share batch size N, got {logits.shape[0]} vs {labels.shape[0]}"
        )
    if n_thresholds < 2:
        raise ValueError(f"n_thresholds must be >= 2, got {n_thresholds}")

    num_classes = int(logits.shape[1])
    probs = torch.softmax(logits, dim=-1)
    p = probs.clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1)
    uncertainty = (entropy / (math.log(num_classes) + eps)).clamp(0.0, 1.0)

    preds = logits.argmax(dim=1)
    incorrect = preds.ne(labels)
    correct = ~incorrect
    denom_err = incorrect.sum().float()

    thresholds_u = torch.linspace(0.0, 1.0, n_thresholds, device=logits.device)
    pui_vals = []
    piu_vals = []
    for t in thresholds_u:
        uncertain = uncertainty >= t
        niu = (incorrect & uncertain).sum().float()
        nau = (correct & uncertain).sum().float()
        pui_vals.append(niu / (denom_err + eps))
        piu_vals.append(niu / (nau + niu + eps))

    return {
        "thresholds_uncertainty": thresholds_u.detach().cpu(),
        "P_U_given_I": torch.stack(pui_vals).detach().cpu(),
        "P_I_given_U": torch.stack(piu_vals).detach().cpu(),
    }


@torch.no_grad()
def avu_means_from_predictive_entropy_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 101,
) -> Dict[str, float]:
    """
    Computes mean Accuracy-vs-Uncertainty metrics from predictive-entropy certainty.

    Inputs:
      logits: [N, C]
      labels: [N]

    Outputs:
      {
        "p_a_given_c_mean": float,
        "p_u_given_i_mean": float,
        "p_i_given_u_mean": float,
        "avu_mean": float,
      }
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,C], got shape {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be [N], got shape {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels must share batch size N, got {logits.shape[0]} vs {labels.shape[0]}"
        )
    if n_thresholds < 2:
        raise ValueError(f"n_thresholds must be >= 2, got {n_thresholds}")

    num_classes = int(logits.shape[1])
    certainty = certainty_from_predictive_entropy_logits(logits, num_classes=num_classes)
    preds = logits.argmax(dim=1)
    correct = preds.eq(labels)
    thresholds = torch.linspace(0.0, 1.0, n_thresholds, device=certainty.device)
    curves = accuracy_vs_uncertainty_metrics(
        certainty=certainty,
        correct=correct,
        thresholds=thresholds,
    )
    return {
        "p_a_given_c_mean": float(curves["P_A_given_C"].mean().item()),
        "p_u_given_i_mean": float(curves["P_U_given_I"].mean().item()),
        "p_i_given_u_mean": float(curves["P_I_given_U"].mean().item()),
        "avu_mean": float(curves["AvU"].mean().item()),
    }


@torch.no_grad()
def calibration_error_by_certainty_bins(
    certainty: torch.Tensor,   # [N] in [0,1]
    correct: torch.Tensor,     # [N] bool/{0,1} (1 means correct)
    n_bins: int = 10,
) -> Dict[str, torch.Tensor | float]:
    """
    Calibration gaps over certainty bins.

    Per bin i:
      gap_i = |acc_i - cert_i|
      where
        acc_i  = mean(correct in bin i)
        cert_i = mean(certainty in bin i)

    Returns:
      - bin_upper_percent: [B] (10,20,...,100 for B=10)
      - calibration_error_percent: [B] (100 * gap_i)
      - counts: [B]
      - ece: unweighted mean gap over bins (paper-style, 0..1)
      - mce: max gap over bins (0..1)
    """
    cert = certainty.detach().flatten().float().clamp(0.0, 1.0)
    corr = correct.detach().flatten().bool()
    if cert.numel() != corr.numel():
        raise ValueError(f"certainty and correct must have same N, got {cert.numel()} vs {corr.numel()}")
    if cert.numel() == 0:
        raise ValueError("certainty and correct must be non-empty")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=cert.device)
    gaps = []
    counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (cert >= lo) & (cert <= hi)
        else:
            mask = (cert > lo) & (cert <= hi)
        cnt = int(mask.sum().item())
        counts.append(cnt)
        if cnt > 0:
            acc_i = corr[mask].float().mean()
            cert_i = cert[mask].mean()
            gaps.append(torch.abs(acc_i - cert_i))
        else:
            gaps.append(torch.tensor(0.0, device=cert.device))

    gaps_t = torch.stack(gaps)
    counts_t = torch.tensor(counts, device=cert.device, dtype=torch.long)
    # Paper-style ECE: unweighted mean over fixed bins.
    ece = torch.mean(gaps_t)
    mce = torch.max(gaps_t)

    return {
        "bin_upper_percent": (bins[1:] * 100.0).detach().cpu(),
        "calibration_error_percent": (gaps_t * 100.0).detach().cpu(),
        "counts": counts_t.detach().cpu(),
        "ece": float(ece.item()),
        "mce": float(mce.item()),
    }


@torch.no_grad()
def calibration_error_by_certainty_bins_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    n_bins: int = 10,
) -> Dict[str, torch.Tensor | float]:
    """
    Convenience wrapper:
      certainty = 1 - predictive_entropy/log(C)
      correct   = argmax(logits) == labels
      then computes per-bin calibration gaps + ECE/MCE.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,C], got shape {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be [N], got shape {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels must share batch size N, got {logits.shape[0]} vs {labels.shape[0]}"
        )
    certainty = certainty_from_predictive_entropy_logits(logits, num_classes=num_classes)
    correct = logits.argmax(dim=1).eq(labels)
    return calibration_error_by_certainty_bins(
        certainty=certainty,
        correct=correct,
        n_bins=n_bins,
    )


@torch.no_grad()
def mce_from_logits(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Maximum Calibration Error (MCE) using confidence binning.
    Same bins as ECE, but returns max |acc - conf| across bins.
    """
    probs = torch.softmax(logits, dim=-1)
    conf, preds = probs.max(dim=-1)
    correct = (preds == labels).float()

    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    max_gap = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            gap = torch.abs(bin_acc - bin_conf)
            max_gap = torch.maximum(max_gap, gap)

    return max_gap.item()
@torch.no_grad()
def roc_curve_binary(scores: torch.Tensor, labels: torch.Tensor, n_thresholds: int = 400) -> dict:
    """
    Compute ROC curve for binary labels.
    scores: [N] higher => more positive
    labels: [N] in {0,1}
    Returns dict with thresholds, fpr, tpr (CPU tensors).
    """
    s = scores.detach().flatten().float()
    y = labels.detach().flatten().float()

    # thresholds from high to low so TPR/FPR increase
    t = torch.linspace(s.max(), s.min(), n_thresholds, device=s.device)

    P = (y == 1).sum().float()
    N = (y == 0).sum().float()
    if P == 0 or N == 0:
        return {"thresholds": t.cpu(), "fpr": torch.full((n_thresholds,), float("nan")).cpu(),
                "tpr": torch.full((n_thresholds,), float("nan")).cpu()}

    tpr = []
    fpr = []
    for thr in t:
        pred_pos = s >= thr
        TP = ((pred_pos) & (y == 1)).sum().float()
        FP = ((pred_pos) & (y == 0)).sum().float()
        tpr.append(TP / (P + 1e-12))
        fpr.append(FP / (N + 1e-12))

    return {"thresholds": t.detach().cpu(), "fpr": torch.stack(fpr).detach().cpu(), "tpr": torch.stack(tpr).detach().cpu()}


@torch.no_grad()
def fpr_at_tpr(scores: torch.Tensor, labels: torch.Tensor, target_tpr: float = 0.95, n_thresholds: int = 400) -> float:
    """
    Returns the minimum FPR achieved at TPR >= target_tpr (standard FPR@95%TPR).
    """
    roc = roc_curve_binary(scores, labels, n_thresholds=n_thresholds)
    fpr = roc["fpr"]
    tpr = roc["tpr"]

    mask = tpr >= target_tpr
    if mask.any():
        return float(fpr[mask].min().item())
    return float("nan")
