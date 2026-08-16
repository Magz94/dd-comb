from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import wandb

from ddcomb.config import FusionPipelineConfig, _suffix
from ddcomb.data.dataloaders import load_models_and_dataloaders_for_postfix
from ddcomb.evaluation.evaluate import evaluate_fusion_classifier
from ddcomb.evaluation.logging import log_evaluation_artifacts
from ddcomb.runtime import (
    FUSION_CLASSIFIER_CKPT_PREFIX,
    FUSION_MODEL_CKPT_PREFIX,
    checkpoint_path,
    init_runtime,
    resolve_model_order,
)
from ddcomb.training.build import build_fusion_model, load_fusion_state_for_transfer
from ddcomb.training.train import (
    calibrate_dirichlet_logit_moe_support_router,
    train_fusion_classifier,
    train_fusion_classifier_with_val_selection,
)


def run_single_postfix(
    cfg: FusionPipelineConfig,
    postfix: str,
    models_list,
    train_loader_emb,
    val_loader_emb,
    test_loader_emb,
    device: str,
    mode: str,
    frozen_fusion_tag: Optional[str] = None,
    frozen_fusion_added_postfix: Optional[str] = None,
    skip_train: bool = False,
    return_full_metrics: bool = False,
) -> Dict[str, float]:
    if mode not in {"stage_a", "stage_b", "stage_d"}:
        raise ValueError("mode must be 'stage_a', 'stage_b', or 'stage_d'.")

    freeze_fusion = mode == "stage_b"
    active_weight_decay = cfg.weight_decay_stage_a if mode in {"stage_a", "stage_d"} else cfg.weight_decay_stage_b
    frozen_fusion_ckpt = None
    if freeze_fusion:
        if not frozen_fusion_tag:
            raise ValueError("Stage B requires frozen_fusion_tag from Stage A.")
        frozen_fusion_ckpt = checkpoint_path(
            cfg.model_save_path,
            FUSION_MODEL_CKPT_PREFIX,
            frozen_fusion_tag,
            postfix,
            frozen_fusion_added_postfix,
        )

    fusion_model, classifier_model = build_fusion_model(
        models_list,
        cfg=cfg,
        freeze_fusion=freeze_fusion,
        frozen_fusion_ckpt=frozen_fusion_ckpt,
    )

    # Stage-D warm-start from a frozen Stage-A checkpoint. Skipped when skip_train=True
    # because the skip_train branch below loads current_fusion_ckpt + current_clf_ckpt
    # directly, which would overwrite anything initialised here.
    if mode == "stage_d" and not skip_train:
        if not frozen_fusion_tag:
            raise ValueError("Stage D requires frozen_fusion_tag as the initialization fusion checkpoint tag.")
        init_fusion_ckpt = checkpoint_path(
            cfg.model_save_path,
            FUSION_MODEL_CKPT_PREFIX,
            frozen_fusion_tag,
            postfix,
            frozen_fusion_added_postfix,
        )
        fusion_state = torch.load(init_fusion_ckpt, map_location="cpu")
        probe_keys = [k for k in fusion_state.keys() if k.startswith("token_probe.")]
        if probe_keys:
            print(f"[INFO] Dropping legacy probe keys from fusion checkpoint: {probe_keys}")
            fusion_state = {k: v for k, v in fusion_state.items() if not k.startswith("token_probe.")}
        load_fusion_state_for_transfer(fusion_model, fusion_state)

        if cfg.stage_d_load_classifier:
            if not cfg.stage_d_classifier_tag:
                raise ValueError("stage_d_load_classifier=True requires stage_d_classifier_tag.")
            init_clf_ckpt = checkpoint_path(
                cfg.model_save_path,
                FUSION_CLASSIFIER_CKPT_PREFIX,
                cfg.stage_d_classifier_tag,
                postfix,
                cfg.stage_d_classifier_added_postfix,
            )
            init_clf_state = torch.load(init_clf_ckpt, map_location="cpu")
            incompatible = classifier_model.load_state_dict(init_clf_state, strict=False)
            missing = [k for k in incompatible.missing_keys if not k.startswith("fusion_model.")]
            unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("fusion_model.")]
            if missing:
                print(f"[WARN] Stage-D classifier init missing keys: {missing}")
            if unexpected:
                print(f"[WARN] Stage-D classifier init unexpected keys: {unexpected}")
            if cfg.stage_d_classifier_strict and (missing or unexpected):
                raise RuntimeError(
                    "stage_d_classifier_strict=True and classifier checkpoint load was not exact.\n"
                    f"Missing: {missing}\nUnexpected: {unexpected}"
                )

    current_fusion_ckpt = checkpoint_path(
        cfg.model_save_path,
        FUSION_MODEL_CKPT_PREFIX,
        cfg.run_tag,
        postfix,
        cfg.added_postfix,
    )
    current_clf_ckpt = checkpoint_path(
        cfg.model_save_path,
        FUSION_CLASSIFIER_CKPT_PREFIX,
        cfg.run_tag,
        postfix,
        cfg.added_postfix,
    )

    if cfg.use_wandb:
        if mode == "stage_a":
            project = "fusion_model_training"
        elif mode == "stage_b":
            project = "fusion_model_classifier_only"
        else:
            project = "fusion_model_finetune_stage_d"
        run_name = f"{mode}_{_suffix(cfg.run_tag, postfix, cfg.added_postfix)}"
        wandb.init(project=project, name=run_name, reinit="finish_previous")
        # All FusionPipelineConfig fields plus the run-time additions that
        # aren't part of the config dataclass.
        wandb_cfg = asdict(cfg)
        wandb_cfg.update(
            {
                "mode": mode,
                "postfix": postfix,
                "model_order": [name for name, _, _ in models_list],
                "weight_decay_active": active_weight_decay,
                "frozen_fusion_tag": frozen_fusion_tag,
                "has_val_loader": val_loader_emb is not None,
            }
        )
        wandb.config.update(wandb_cfg, allow_val_change=True)

    if skip_train:
        if mode in {"stage_a", "stage_d"}:
            fusion_model.load_state_dict(torch.load(current_fusion_ckpt, map_location=device))
        classifier_model.load_state_dict(torch.load(current_clf_ckpt, map_location=device))
    else:
        os.makedirs(cfg.model_save_path, exist_ok=True)
        if val_loader_emb is not None:
            # Best-val-accuracy checkpoint selection benefits any stage that
            # has a validation loader, not just Stage-B. Stage-D in particular
            # is the most likely place to overfit and was previously skipped.
            classifier_model = train_fusion_classifier_with_val_selection(
                model=classifier_model,
                train_dataloader=train_loader_emb,
                val_dataloader=val_loader_emb,
                cfg=cfg,
                classifier_ckpt_path=current_clf_ckpt,
                device=device,
                weight_decay=active_weight_decay,
            )
        else:
            classifier_model = train_fusion_classifier(
                classifier_model,
                train_loader_emb,
                cfg=cfg,
                classifier_ckpt_path=current_clf_ckpt,
                device=device,
                weight_decay=active_weight_decay,
            )

        if mode in {"stage_a", "stage_d"}:
            torch.save(fusion_model.state_dict(), current_fusion_ckpt)

    calibrate_dirichlet_logit_moe_support_router(
        model=classifier_model,
        dataloader=train_loader_emb,
        cfg=cfg,
        device=device,
    )

    need_dirichlet_outputs = cfg.attention_method == "dirichlet"
    need_prefusion_outputs = cfg.use_pre_fusion_uncertainty
    need_uncertainty_outputs = (
        (cfg.enable_uncertainty_metrics and cfg.attention_method != "dirichlet")
        or need_dirichlet_outputs
        or need_prefusion_outputs
    )
    need_plot_outputs = cfg.enable_diagnostic_plots and cfg.attention_method != "dirichlet"

    eval_out = evaluate_fusion_classifier(
        classifier_model,
        test_loader_emb,
        num_classes=cfg.num_classes,
        device=device,
        return_embeddings=need_plot_outputs,
        return_attention=need_plot_outputs or (need_uncertainty_outputs and cfg.attention_method != "dirichlet"),
        return_extras=need_uncertainty_outputs,
    )

    if cfg.prediction_output_path:
        prediction_path = Path(cfg.prediction_output_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            y_true=eval_out["labels"].detach().cpu().numpy(),
            y_prob=eval_out["probs"].detach().cpu().numpy(),
            logits=eval_out["logits"].detach().cpu().numpy(),
            num_classes=np.array([cfg.num_classes], dtype=np.int64),
            model_type=np.array(["fusion"]),
            run_tag=np.array([cfg.run_tag]),
            postfix=np.array([postfix]),
        )
        print(f"[PRED] Wrote prediction probabilities: {prediction_path}")

    metrics = log_evaluation_artifacts(
        eval_out=eval_out,
        classifier_model=classifier_model,
        models_list=models_list,
        cfg=cfg,
        postfix=postfix,
    )

    if cfg.use_wandb:
        wandb.finish()

    # Always return a dict so callers don't have to branch on return_full_metrics.
    # When return_full_metrics is False, the dict carries only the headline accuracy;
    # when True, it carries every scalar metric produced by log_evaluation_artifacts.
    if return_full_metrics:
        return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    return {"test_accuracy": float(metrics["test_accuracy"])}


def run_experiment(
    cfg: FusionPipelineConfig,
    mode: str,
    frozen_fusion_tag: Optional[str] = None,
    frozen_fusion_added_postfix: Optional[str] = None,
    skip_train: bool = False,
    return_full_metrics: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Train + eval per postfix; return ``{postfix: metrics_dict}``.

    ``return_full_metrics`` controls dict richness only: False yields a
    minimal ``{"test_accuracy": float}`` per postfix; True yields every
    scalar metric produced by log_evaluation_artifacts.
    """
    device = init_runtime(cfg)
    model_order = resolve_model_order(cfg)
    print(f"Model order: {model_order}")

    results: Dict[str, Dict[str, float]] = {}
    for postfix in cfg.subfolders:
        models_list, train_loader_emb, val_loader_emb, test_loader_emb, _ = load_models_and_dataloaders_for_postfix(
            cfg=cfg,
            postfix=postfix,
            model_order=model_order,
            device=device,
        )
        postfix_metrics = run_single_postfix(
            cfg=cfg,
            postfix=postfix,
            models_list=models_list,
            train_loader_emb=train_loader_emb,
            val_loader_emb=val_loader_emb,
            test_loader_emb=test_loader_emb,
            device=device,
            mode=mode,
            frozen_fusion_tag=frozen_fusion_tag,
            frozen_fusion_added_postfix=frozen_fusion_added_postfix,
            skip_train=skip_train,
            return_full_metrics=return_full_metrics,
        )
        results[postfix] = postfix_metrics
        print(f"Test Accuracy @ {postfix}: {postfix_metrics.get('test_accuracy', float('nan')):.4f}")

    return results
