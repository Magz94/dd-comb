"""Eval-time W&B logging and plot rendering for linear fusion runs."""
from __future__ import annotations

import os
from typing import Dict

import torch
import wandb

from ddcomb.config import FusionPipelineConfig, _suffix
from ddcomb.evaluation.metrics import (
    avu_means_from_predictive_entropy_logits,
    binary_auroc,
    calibration_error_by_certainty_bins_from_logits,
    certainty_from_predictive_entropy_logits,
    ece_from_logits,
    fpr_at_tpr,
    mce_from_logits,
    nll_from_logits,
    p_u_given_i_uncertainty_sweep_from_logits,
    roc_curve_binary,
    uncertainty_predictive_entropy_from_logits,
)
from ddcomb.evaluation.plots import (
    plot_calibration_error_by_certainty_bar,
    plot_certainty_correct_vs_incorrect_hist,
    plot_certainty_proportion_bar,
    plot_p_i_given_u_uncertainty_curve,
    plot_reliability_diagram,
    plot_risk_coverage_curve,
    plot_roc_curve,
)
from ddcomb.models.fusion import FusionClassifier
from ddcomb.models.plots import (
    plot_attention_weights,
    plot_fusion_embeddings,
    plot_self_attention,
)


ROC_N_THRESHOLDS = 400


def _render_roc_image(scores, incorrect, plot_path: str, title: str):
    roc = roc_curve_binary(scores, incorrect, n_thresholds=ROC_N_THRESHOLDS)
    plot_roc_curve(roc, plot_path, title=title)
    return wandb.Image(plot_path)


def _pred_entropy_core_metrics(
    metric_logits, labels_all, incorrect, n_bins: int = 15, include_nll: bool = False
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if include_nll:
        out["test_nll"] = nll_from_logits(metric_logits, labels_all)
    out["test_ece"] = ece_from_logits(metric_logits, labels_all, n_bins=n_bins)
    u_entropy = uncertainty_predictive_entropy_from_logits(metric_logits)
    out["auroc_err_pred_entropy"] = binary_auroc(u_entropy, incorrect)
    out["fpr@95_err_pred_entropy"] = fpr_at_tpr(u_entropy, incorrect, target_tpr=0.95)
    out["_u_entropy"] = u_entropy
    return out


def _append_dgmobe_diagnostics(eval_out, wandb_log: Dict[str, float]) -> None:
    router = eval_out.get("router_probs")
    if router is not None:
        router = router.float().clamp_min(1e-12)
        router = router / router.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(router * router.log()).sum(dim=1)
        if router.shape[1] > 1:
            entropy = entropy / torch.log(router.new_tensor(float(router.shape[1]))).clamp_min(1e-12)
        uniform = router.new_full(router.shape, 1.0 / float(router.shape[1]))
        wandb_log["dgmobe/router_entropy_mean"] = float(entropy.mean().item())
        wandb_log["dgmobe/router_max_mean"] = float(router.max(dim=1).values.mean().item())
        wandb_log["dgmobe/router_l1_from_uniform_mean"] = float(
            (router - uniform).abs().sum(dim=1).mean().item()
        )

    expert_logits = eval_out.get("expert_logits")
    if expert_logits is not None:
        expert_logits = expert_logits.float()
        centered = expert_logits - expert_logits.mean(dim=-1, keepdim=True)
        normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        num_experts = similarity.shape[1]
        if num_experts > 1:
            mask = ~torch.eye(num_experts, device=similarity.device, dtype=torch.bool).unsqueeze(0)
            wandb_log["dgmobe/expert_logit_cos_mean"] = float(similarity.masked_select(mask).mean().item())
        wandb_log["dgmobe/expert_logit_std_mean"] = float(expert_logits.std(dim=1, unbiased=False).mean().item())


def _append_eval_uncertainty_metrics(
    metric_logits,
    labels_all,
    incorrect,
    cfg: FusionPipelineConfig,
    wandb_log: Dict[str, float],
    plot_path,
) -> Dict[str, "wandb.Image"]:
    if not cfg.enable_uncertainty_metrics:
        return {}

    metrics = avu_means_from_predictive_entropy_logits(
        logits=metric_logits, labels=labels_all, n_thresholds=101,
    )
    calib_by_cert = calibration_error_by_certainty_bins_from_logits(
        logits=metric_logits, labels=labels_all, num_classes=cfg.num_classes, n_bins=10,
    )
    pui_sweep = p_u_given_i_uncertainty_sweep_from_logits(
        logits=metric_logits, labels=labels_all, n_thresholds=101,
    )

    piu_plot_path = plot_path("p_i_given_u_uncertainty")
    plot_p_i_given_u_uncertainty_curve(
        pui_sweep, save_path=piu_plot_path, title="P(I|U) vs uncertainty threshold",
    )
    certainty = certainty_from_predictive_entropy_logits(
        logits=metric_logits, num_classes=cfg.num_classes,
    )
    cert_plot_path = plot_path("certainty_distribution")
    plot_certainty_correct_vs_incorrect_hist(
        certainty=certainty, incorrect=incorrect, save_path=cert_plot_path,
        title="Certainty Distribution", bins=80, certainty_is_percent=False,
    )
    cert_prop_plot_path = plot_path("certainty_proportion")
    plot_certainty_proportion_bar(
        certainty=certainty, save_path=cert_prop_plot_path,
        title="Certainty Proportion", certainty_is_percent=False,
    )
    calib_plot_path = plot_path("calibration_error_by_certainty")
    plot_calibration_error_by_certainty_bar(
        calib=calib_by_cert, save_path=calib_plot_path, title="Calibration Error by Certainty",
    )

    wandb_log["uncertainty/p_a_given_c_mean"] = metrics["p_a_given_c_mean"]
    wandb_log["uncertainty/p_u_given_i_mean"] = metrics["p_u_given_i_mean"]
    wandb_log["uncertainty/p_i_given_u_mean"] = metrics["p_i_given_u_mean"]
    wandb_log["uncertainty/avu_mean"] = metrics["avu_mean"]
    wandb_log["uncertainty/ece_certainty_binned"] = float(calib_by_cert["ece"])
    wandb_log["uncertainty/mce_certainty_binned"] = float(calib_by_cert["mce"])

    return {
        "plot/p_i_given_u_uncertainty": wandb.Image(piu_plot_path),
        "plot/certainty_distribution": wandb.Image(cert_plot_path),
        "plot/certainty_proportion": wandb.Image(cert_prop_plot_path),
        "plot/calibration_error_by_certainty": wandb.Image(calib_plot_path),
    }


def _log_dirichlet_attention_branch(
    eval_out, cfg, _plot_path, labels_all, logits_all, incorrect, wandb_log,
) -> Dict[str, float]:
    core = _pred_entropy_core_metrics(logits_all, labels_all, incorrect)
    u_entropy = core.pop("_u_entropy")
    wandb_log.update(core)

    plot_logs = {
        "plot/roc_err_pred_entropy": _render_roc_image(
            u_entropy, incorrect, _plot_path("roc_err_pred_entropy"),
            "ROC: Error detection (Predictive Entropy)",
        )
    }

    if "dirichlet_strength" in eval_out:
        dir_strength = eval_out["dirichlet_strength"]
        dir_strength_flat = dir_strength.flatten()
        wandb_log["dir_strength_p50"] = float(torch.quantile(dir_strength_flat, 0.50).item())
        wandb_log["dir_strength_p95"] = float(torch.quantile(dir_strength_flat, 0.95).item())

        if "dirichlet_uncertainty" in eval_out:
            dir_u = eval_out["dirichlet_uncertainty"]
        else:
            token_count = float(dir_strength.shape[1])
            dir_u = token_count / dir_strength.clamp_min(1e-12)

        dir_u_flat = dir_u.flatten()
        wandb_log["dir_u_p50"] = float(torch.quantile(dir_u_flat, 0.50).item())
        wandb_log["dir_u_p95"] = float(torch.quantile(dir_u_flat, 0.95).item())

        dir_u_sample = dir_u.mean(dim=1)
        wandb_log["auroc_err_dir_u"] = binary_auroc(dir_u_sample, incorrect)
        wandb_log["fpr@95_err_dir_u"] = fpr_at_tpr(dir_u_sample, incorrect, target_tpr=0.95)
        plot_logs["plot/roc_err_dir_u"] = _render_roc_image(
            dir_u_sample, incorrect, _plot_path("roc_err_dir_u"),
            "ROC: Error detection (Dirichlet U)",
        )

    plot_logs.update(_append_eval_uncertainty_metrics(
        logits_all, labels_all, incorrect, cfg, wandb_log, _plot_path,
    ))

    if cfg.use_wandb:
        wandb.log(wandb_log)
        wandb.log(plot_logs)
    return wandb_log


def _log_baseline_minimal_branch(
    cfg, _plot_path, labels_all, logits_all, incorrect, wandb_log,
) -> Dict[str, float]:
    core = _pred_entropy_core_metrics(logits_all, labels_all, incorrect)
    u_entropy = core.pop("_u_entropy")
    wandb_log.update(core)

    roc_image = _render_roc_image(
        u_entropy, incorrect, _plot_path("roc_err_pred_entropy"),
        "ROC: Error detection (Predictive Entropy)",
    )

    if cfg.use_wandb:
        wandb.log(wandb_log)
        wandb.log({"plot/roc_err_pred_entropy": roc_image})
    return wandb_log


def _log_full_diagnostic_branch(
    eval_out, cfg, models_list, _plot_path, suffix,
    labels_all, logits_all, preds, incorrect, wandb_log,
) -> Dict[str, float]:
    wandb_log["test_nll"] = nll_from_logits(logits_all, labels_all)
    wandb_log["test_ece"] = ece_from_logits(logits_all, labels_all, n_bins=15)
    wandb_log["test_mce"] = mce_from_logits(logits_all, labels_all, n_bins=15)

    u_entropy = uncertainty_predictive_entropy_from_logits(logits_all)
    wandb_log["auroc_err_pred_entropy"] = binary_auroc(u_entropy, incorrect)
    wandb_log["fpr@95_err_pred_entropy"] = fpr_at_tpr(u_entropy, incorrect, target_tpr=0.95)

    plot_logs = {
        "plot/roc_err_pred_entropy": _render_roc_image(
            u_entropy, incorrect, _plot_path("roc_err_pred_entropy"),
            "ROC: Error detection (Predictive Entropy)",
        )
    }
    plot_logs.update(_append_eval_uncertainty_metrics(
        logits_all, labels_all, incorrect, cfg, wandb_log, _plot_path,
    ))

    uncertainties = {"pred_entropy": u_entropy}

    if "attn_entropy" in eval_out and "pool_weights" in eval_out:
        ent_all = eval_out["attn_entropy"]
        pool_all = eval_out["pool_weights"].squeeze(-1)
        u_ent_mean = ent_all.mean(dim=1)
        u_ent_pooled = (ent_all * pool_all).sum(dim=1)
        wandb_log["auroc_err_entropy_mean"] = binary_auroc(u_ent_mean, incorrect)
        wandb_log["auroc_err_attnent_pooled"] = binary_auroc(u_ent_pooled, incorrect)

    if cfg.enable_diagnostic_plots:
        correct = (preds == labels_all).float()
        reliability_path = _plot_path("reliability")
        plot_reliability_diagram(logits_all, labels_all, n_bins=15, save_path=reliability_path)
        riskcov_path = _plot_path("risk_coverage")
        plot_risk_coverage_curve(
            uncertainties=uncertainties, correct=correct, save_path=riskcov_path,
            title="Risk-coverage: pred-entropy",
        )
        cert_entropy = certainty_from_predictive_entropy_logits(logits_all, num_classes=cfg.num_classes)
        cert_entropy_path = _plot_path("certainty_entropy")
        plot_certainty_correct_vs_incorrect_hist(
            certainty=cert_entropy, incorrect=incorrect, save_path=cert_entropy_path,
            title="Certainty Distribution (Predictive Entropy)", bins=80, certainty_is_percent=False,
        )
        plot_logs["plot/reliability"] = wandb.Image(reliability_path)
        plot_logs["plot/risk_coverage"] = wandb.Image(riskcov_path)
        plot_logs["plot/certainty_entropy"] = wandb.Image(cert_entropy_path)

    if cfg.enable_diagnostic_plots and "attn_weights_mean" in eval_out:
        plot_self_attention(
            eval_out["attn_weights_mean"], models=models_list,
            save_path=_plot_path("self_attention", ext="svg"),
        )
    if cfg.enable_diagnostic_plots and "pool_weights_mean" in eval_out:
        plot_attention_weights(
            eval_out["pool_weights_mean"], models=models_list,
            save_path=_plot_path("pool_weights", ext="svg"),
        )
    if cfg.enable_diagnostic_plots and "embeddings" in eval_out and "emb_labels" in eval_out:
        embedding_2d = plot_fusion_embeddings(
            eval_out["embeddings"], eval_out["emb_labels"], method="umap",
            n_neighbors=15, min_dist=0.1, metric="cosine", title=None,
            save_path=_plot_path("umap_fused", ext="svg"),
        )
        if cfg.use_wandb:
            table = wandb.Table(columns=["x", "y", "class"])
            labels_np = eval_out["emb_labels"].numpy()
            for x, y, label in zip(embedding_2d[:, 0], embedding_2d[:, 1], labels_np):
                table.add_data(float(x), float(y), int(label))
            wandb.log(
                {
                    "fusion_embeddings_umap": wandb.plot.scatter(
                        table=table, x="x", y="y",
                        title=f"Fusion Embeddings UMAP {suffix}",
                    )
                }
            )

    if cfg.use_wandb:
        wandb.log(wandb_log)
        wandb.log(plot_logs)
    return wandb_log


def log_evaluation_artifacts(
    eval_out: Dict[str, torch.Tensor],
    classifier_model: FusionClassifier,
    models_list,
    cfg: FusionPipelineConfig,
    postfix: str,
) -> Dict[str, float]:
    logits_all = eval_out["logits"]
    probs_all = eval_out.get("probs")
    labels_all = eval_out["labels"]

    preds = probs_all.argmax(dim=1) if probs_all is not None else logits_all.argmax(dim=1)
    incorrect = 1.0 - (preds == labels_all).float()

    wandb_log: Dict[str, float] = {
        "test_accuracy": float(eval_out["accuracy"]),
        "test_precision": float(eval_out["precision"]),
        "test_recall": float(eval_out["recall"]),
    }
    _append_dgmobe_diagnostics(eval_out, wandb_log)

    os.makedirs(cfg.img_plot_path, exist_ok=True)
    suffix = _suffix(cfg.run_tag, postfix, cfg.added_postfix)

    def _plot_path(name: str, ext: str = "png") -> str:
        return os.path.join(cfg.img_plot_path, f"{name}_{suffix}.{ext}")

    if cfg.attention_method == "dirichlet":
        return _log_dirichlet_attention_branch(
            eval_out, cfg, _plot_path, labels_all, logits_all, incorrect, wandb_log,
        )
    if not cfg.enable_uncertainty_metrics:
        return _log_baseline_minimal_branch(
            cfg, _plot_path, labels_all, logits_all, incorrect, wandb_log,
        )
    return _log_full_diagnostic_branch(
        eval_out, cfg, models_list, _plot_path, suffix,
        labels_all, logits_all, preds, incorrect, wandb_log,
    )
