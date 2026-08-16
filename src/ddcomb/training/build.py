from __future__ import annotations

from typing import Optional

import torch

from ddcomb.config import FusionPipelineConfig
from ddcomb.models.embedding_models import get_embedding_shapes
from ddcomb.models.fusion import FusionClassifier, FusionModel


def load_fusion_state_for_transfer(fusion_model: FusionModel, fusion_state: dict[str, torch.Tensor]) -> None:
    """Load a fusion checkpoint, reinitializing target-class pre-fusion modal heads if needed."""
    current_state = fusion_model.state_dict()
    dropped_resize_keys = []
    for key, value in list(fusion_state.items()):
        if key not in current_state:
            continue
        if current_state[key].shape == value.shape:
            continue
        if key.startswith("prefusion_modal_heads."):
            dropped_resize_keys.append(key)
            fusion_state.pop(key)
            continue
        raise RuntimeError(
            "Fusion checkpoint shape mismatch outside the pre-fusion auxiliary modal heads.\n"
            f"Key: {key}\n"
            f"Checkpoint shape: {tuple(value.shape)}\n"
            f"Current shape: {tuple(current_state[key].shape)}"
        )

    if dropped_resize_keys:
        print(
            "[Stage D] Reinitializing pre-fusion auxiliary modal heads for the target "
            f"dataset classes. Dropped checkpoint keys: {dropped_resize_keys}"
        )
        incompatible = fusion_model.load_state_dict(fusion_state, strict=False)
        missing = [key for key in incompatible.missing_keys if key not in dropped_resize_keys]
        if missing:
            raise RuntimeError(f"Unexpected missing fusion keys while loading checkpoint: {missing}")
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected fusion checkpoint keys: {incompatible.unexpected_keys}")
    else:
        fusion_model.load_state_dict(fusion_state, strict=True)


def build_fusion_model(
    models,
    cfg: FusionPipelineConfig,
    freeze_fusion: bool = False,
    frozen_fusion_ckpt: Optional[str] = None,
):
    emb_shapes = get_embedding_shapes(models)
    input_dims = [s[-1] for s in emb_shapes]
    default_fusion_dim = min(input_dims)
    hidden_dim = fusion_dim = int(cfg.fusion_projection_dim or default_fusion_dim)
    if hidden_dim % cfg.n_heads != 0:
        raise ValueError(
            f"fusion_projection_dim={hidden_dim} must be divisible by n_heads={cfg.n_heads}."
        )

    fusion_model = FusionModel(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        fusion_dim=fusion_dim,
        n_heads=cfg.n_heads,
        attention_method=cfg.attention_method,
        fused_l2_normalize=cfg.fused_l2_normalize,
        use_pre_fusion_uncertainty=cfg.use_pre_fusion_uncertainty,
        pre_fusion_uncertainty_hidden_dim=cfg.pre_fusion_uncertainty_hidden_dim,
        pre_fusion_uncertainty_eps=cfg.pre_fusion_uncertainty_eps,
        pre_fusion_num_classes=cfg.num_classes,
    )

    if freeze_fusion:
        if not frozen_fusion_ckpt:
            raise ValueError("freeze_fusion=True but no frozen_fusion_ckpt provided.")
        fusion_state = torch.load(frozen_fusion_ckpt, map_location="cpu")
        probe_keys = [k for k in fusion_state.keys() if k.startswith("token_probe.")]
        if probe_keys:
            print(f"[INFO] Dropping legacy probe keys from fusion checkpoint: {probe_keys}")
            fusion_state = {k: v for k, v in fusion_state.items() if not k.startswith("token_probe.")}
        load_fusion_state_for_transfer(fusion_model, fusion_state)
        for p in fusion_model.parameters():
            p.requires_grad = False

    classifier_model = FusionClassifier(
        fusion_model,
        fusion_dim=fusion_dim,
        n_classes=cfg.num_classes,
        classifier_method=cfg.classifier_method,
        dirichlet_kl_regularization=cfg.dirichlet_kl_regularization,
        dirichlet_kl_lambda=cfg.dirichlet_kl_lambda,
        dirichlet_kl_warmup_epochs=cfg.dirichlet_kl_warmup_epochs,
        pre_fusion_aux_loss_weight=cfg.pre_fusion_aux_loss_weight if cfg.use_pre_fusion_uncertainty else 0.0,
        pre_fusion_cali_loss_weight=cfg.pre_fusion_cali_loss_weight if cfg.use_pre_fusion_uncertainty else 0.0,
        pre_fusion_cali_eps=cfg.pre_fusion_cali_eps,
        attention_loss_mode=cfg.attention_loss_mode,
        attention_loss_warmup_epochs=cfg.attention_loss_warmup_epochs,
        attention_entropy_sample_lambda=cfg.attention_entropy_sample_lambda,
        attention_entropy_batch_lambda=cfg.attention_entropy_batch_lambda,
        attention_ce_weighted_easy_lambda=cfg.attention_ce_weighted_easy_lambda,
        attention_ce_weighted_hard_lambda=cfg.attention_ce_weighted_hard_lambda,
        attention_ce_weighted_tau=cfg.attention_ce_weighted_tau,
        attention_head_diversity_lambda=cfg.attention_head_diversity_lambda,
        classification_loss_mode=cfg.classification_loss_mode,
        classification_kl_lambda=cfg.classification_kl_lambda,
        classification_loss_warmup_epochs=cfg.classification_loss_warmup_epochs,
        classification_fisher_power=cfg.classification_fisher_power,
        class_evidential_loss_mode=cfg.class_evidential_loss_mode,
        class_evidential_wrong_lambda=cfg.class_evidential_wrong_lambda,
        class_evidential_margin_lambda=cfg.class_evidential_margin_lambda,
        class_evidential_margin=cfg.class_evidential_margin,
        class_evidential_nll_lambda=cfg.class_evidential_nll_lambda,
        class_evidential_kl_lambda=cfg.class_evidential_kl_lambda,
        class_evidential_warmup_epochs=cfg.class_evidential_warmup_epochs,
        class_evidential_eps=cfg.class_evidential_eps,
        use_dirichlet_logit_moe=cfg.use_dirichlet_logit_moe,
        dirichlet_logit_moe_gamma=cfg.dirichlet_logit_moe_gamma,
        dirichlet_logit_moe_aux_lambda=cfg.dirichlet_logit_moe_aux_lambda,
        dirichlet_logit_moe_resp_lambda=cfg.dirichlet_logit_moe_resp_lambda,
        dirichlet_logit_moe_tau=cfg.dirichlet_logit_moe_tau,
        dirichlet_logit_moe_warmup_epochs=cfg.dirichlet_logit_moe_warmup_epochs,
        dirichlet_logit_moe_router=cfg.dirichlet_logit_moe_router,
        dirichlet_logit_moe_router_beta=cfg.dirichlet_logit_moe_router_beta,
        dirichlet_logit_moe_router_temperature=cfg.dirichlet_logit_moe_router_temperature,
        dirichlet_logit_moe_combine_mode=cfg.dirichlet_logit_moe_combine_mode,
        dirichlet_logit_moe_class_router_kl_lambda=cfg.dirichlet_logit_moe_class_router_kl_lambda,
        dirichlet_logit_moe_dempster_fused_view=cfg.dirichlet_logit_moe_dempster_fused_view,
        dirichlet_logit_moe_dempster_rule=cfg.dirichlet_logit_moe_dempster_rule,
        dirichlet_logit_moe_dempster_loss_mode=cfg.dirichlet_logit_moe_dempster_loss_mode,
        dirichlet_logit_moe_dempster_kl_lambda=cfg.dirichlet_logit_moe_dempster_kl_lambda,
        dirichlet_logit_moe_dempster_edl_lambda=cfg.dirichlet_logit_moe_dempster_edl_lambda,
        use_mean_preserving_evidential_moe=cfg.use_mean_preserving_evidential_moe,
        mp_evidential_moe_objective=cfg.mp_evidential_moe_objective,
        mp_evidential_moe_edl_lambda=cfg.mp_evidential_moe_edl_lambda,
        mp_evidential_moe_kl_lambda=cfg.mp_evidential_moe_kl_lambda,
        mp_evidential_moe_strength_lambda=cfg.mp_evidential_moe_strength_lambda,
        mp_evidential_moe_strength_scale=cfg.mp_evidential_moe_strength_scale,
        mp_evidential_moe_strength_tau=cfg.mp_evidential_moe_strength_tau,
        mp_evidential_moe_warmup_epochs=cfg.mp_evidential_moe_warmup_epochs,
        mp_evidential_moe_hidden_dim=cfg.mp_evidential_moe_hidden_dim,
        mp_evidential_moe_eps=cfg.mp_evidential_moe_eps,
        use_evidential_fusion=cfg.use_evidential_fusion,
        evidential_fusion_use_experts=cfg.evidential_fusion_use_experts,
        evidential_fusion_loss_mode=cfg.evidential_fusion_loss_mode,
        evidential_fusion_objective=cfg.evidential_fusion_objective,
        evidential_fusion_beta=cfg.evidential_fusion_beta,
        evidential_fusion_gamma=cfg.evidential_fusion_gamma,
        evidential_fusion_edl_lambda=cfg.evidential_fusion_edl_lambda,
        evidential_fusion_kl_lambda=cfg.evidential_fusion_kl_lambda,
        evidential_fusion_aux_lambda=cfg.evidential_fusion_aux_lambda,
        evidential_fusion_resp_lambda=cfg.evidential_fusion_resp_lambda,
        evidential_fusion_tau=cfg.evidential_fusion_tau,
        evidential_fusion_warmup_epochs=cfg.evidential_fusion_warmup_epochs,
        evidential_fusion_eps=cfg.evidential_fusion_eps,
    )
    return fusion_model, classifier_model
