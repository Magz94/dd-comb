from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


CANONICAL_ATTENTION_METHODS = frozenset({"dirichlet", "softmax"})
CANONICAL_CLASSIFIER_METHODS = frozenset({"linear"})
CANONICAL_CLASSIFICATION_LOSS_MODES = frozenset(
    {"cross_entropy", "edl_nll_kl", "dirichlet_expected_ce_kl", "fisher_edl"}
)
CANONICAL_ATTENTION_LOSS_MODES = frozenset(
    {"none", "sparse_balanced", "ce_weighted_uncertainty", "head_diversity"}
)
CANONICAL_CLASS_EVIDENTIAL_LOSS_MODES = frozenset(
    {"none", "wrong_evidence", "evidential_margin", "evidential_nll_kl"}
)
CANONICAL_EVIDENTIAL_FUSION_LOSS_MODES = frozenset({"ce_anchored", "pure"})
CANONICAL_EVIDENTIAL_FUSION_OBJECTIVES = frozenset(
    {"expected_ce_kl", "nll_kl", "brier_mse_kl"}
)
CANONICAL_DIRICHLET_LOGIT_MOE_ROUTERS = frozenset(
    {
        "dirichlet",
        "uniform",
        "mixed",
        "concentration",
        "variance",
        "support_ce",
        "support_ce_mixed",
        "class_stacking",
        "class_conditional_stacking",
    }
)
CANONICAL_DIRICHLET_LOGIT_MOE_COMBINE_MODES = frozenset({"residual", "moe_only", "fused_only", "dempster"})
CANONICAL_DIRICHLET_LOGIT_MOE_DEMPSTER_LOSS_MODES = frozenset({"pure", "ce_anchored"})
CANONICAL_DIRICHLET_LOGIT_MOE_DEMPSTER_RULES = frozenset({"dempster", "mean_evidence"})


@dataclass
class FusionPipelineConfig:
    seed: int = 42
    use_wandb: bool = True

    attention_method: str = "dirichlet"  # one of CANONICAL_ATTENTION_METHODS
    classifier_method: str = "linear"    # one of CANONICAL_CLASSIFIER_METHODS
    fused_l2_normalize: bool = True
    # Optional Dirichlet attention regularizer:
    #   L = L_CE + lambda * KL(Dir(alpha) || Dir(1))
    # Disabled by default to preserve historical runs.
    dirichlet_kl_regularization: bool = False
    dirichlet_kl_lambda: float = 0.0
    # Optional linear warm-up for the active KL weight. 0 means no warm-up.
    dirichlet_kl_warmup_epochs: int = 0
    # Optional static-image pre-fusion uncertainty/reliability weighting.
    use_pre_fusion_uncertainty: bool = False
    pre_fusion_uncertainty_hidden_dim: int = 128
    pre_fusion_uncertainty_eps: float = 1e-6
    pre_fusion_aux_loss_weight: float = 0.1
    pre_fusion_cali_loss_weight: float = 0.05
    pre_fusion_cali_eps: float = 1e-8
    # Optional Dirichlet-attention auxiliary losses. They operate on attention
    # alpha / attention weights, not on classifier evidence.
    attention_loss_mode: str = "none"
    attention_loss_warmup_epochs: int = 0
    attention_entropy_sample_lambda: float = 0.0
    attention_entropy_batch_lambda: float = 0.0
    attention_ce_weighted_easy_lambda: float = 0.0
    attention_ce_weighted_hard_lambda: float = 0.0
    attention_ce_weighted_tau: float = 1.0
    attention_head_diversity_lambda: float = 0.0
    # Primary classification objective. cross_entropy preserves the standard
    # setup; the other modes replace CE with class-level Dirichlet evidence.
    classification_loss_mode: str = "cross_entropy"
    classification_kl_lambda: float = 0.0
    classification_loss_warmup_epochs: int = 0
    classification_fisher_power: float = 1.0
    # Optional class-level evidential losses. They derive
    # alpha_class = softplus(logits) + 1 from classifier outputs.
    class_evidential_loss_mode: str = "none"
    class_evidential_wrong_lambda: float = 0.0
    class_evidential_margin_lambda: float = 0.0
    class_evidential_margin: float = 0.2
    class_evidential_nll_lambda: float = 0.0
    class_evidential_kl_lambda: float = 0.0
    class_evidential_warmup_epochs: int = 0
    class_evidential_eps: float = 1e-8
    # Optional Dirichlet-gated logit mixture of per-backbone expert heads.
    # This keeps the fused classifier and adds a sample-wise expert-logit route:
    # logits = logits_fused + gamma * sum_m p_m logits_m.
    use_dirichlet_logit_moe: bool = False
    dirichlet_logit_moe_gamma: float = 0.5
    dirichlet_logit_moe_aux_lambda: float = 0.2
    dirichlet_logit_moe_resp_lambda: float = 0.1
    dirichlet_logit_moe_tau: float = 0.5
    dirichlet_logit_moe_warmup_epochs: int = 0
    dirichlet_logit_moe_router: str = "dirichlet"
    dirichlet_logit_moe_router_beta: float = 0.5
    dirichlet_logit_moe_router_temperature: float = 1.0
    dirichlet_logit_moe_combine_mode: str = "residual"
    dirichlet_logit_moe_class_router_kl_lambda: float = 1e-3
    # Dempster combine mode (combine_mode="dempster"): TMC-style reduced
    # Dempster-Shafer combination of per-expert Dirichlet opinions, with the
    # fused head optionally included as an additional view.
    dirichlet_logit_moe_dempster_fused_view: bool = True
    # "dempster" = reduced Dempster-Shafer rule; "mean_evidence" = uniform
    # control that averages the per-view evidence under identical training.
    dirichlet_logit_moe_dempster_rule: str = "dempster"
    dirichlet_logit_moe_dempster_loss_mode: str = "pure"
    dirichlet_logit_moe_dempster_kl_lambda: float = 1.0
    dirichlet_logit_moe_dempster_edl_lambda: float = 1.0
    # Mean-preserving evidential extension of Dirichlet Logit-MoE.
    # The Logit-MoE predictive mean is kept as softmax(logits), while a
    # separate strength head learns the Dirichlet concentration.
    use_mean_preserving_evidential_moe: bool = False
    mp_evidential_moe_objective: str = "expected_ce_kl"
    mp_evidential_moe_edl_lambda: float = 0.03
    mp_evidential_moe_kl_lambda: float = 1e-3
    mp_evidential_moe_strength_lambda: float = 0.05
    mp_evidential_moe_strength_scale: float = 50.0
    mp_evidential_moe_strength_tau: float = 1.0
    mp_evidential_moe_warmup_epochs: int = 3
    mp_evidential_moe_hidden_dim: int = 128
    mp_evidential_moe_eps: float = 1e-8
    # Fully evidential fusion variants. They convert classifier evidence
    # into class Dirichlet parameters and couple that evidence with the
    # Dirichlet attention uncertainty. Optional expert heads add per-backbone
    # evidence before the same uncertainty gate.
    use_evidential_fusion: bool = False
    evidential_fusion_use_experts: bool = False
    evidential_fusion_loss_mode: str = "ce_anchored"
    evidential_fusion_objective: str = "expected_ce_kl"
    evidential_fusion_beta: float = 0.5
    evidential_fusion_gamma: float = 0.5
    evidential_fusion_edl_lambda: float = 0.05
    evidential_fusion_kl_lambda: float = 1e-3
    evidential_fusion_aux_lambda: float = 0.05
    evidential_fusion_resp_lambda: float = 0.1
    evidential_fusion_tau: float = 0.5
    evidential_fusion_warmup_epochs: int = 3
    evidential_fusion_eps: float = 1e-8
    # Shared adapter/attention/fused embedding dimension. If None, use the
    # historical default min(input_dims), which is 512 for the 5-backbone setup.
    fusion_projection_dim: Optional[int] = None
    n_heads: int = 8
    # Attention-free DD-CoBE ensemble (models/ddcobe_ensemble.py, run via
    # scripts/sweep_kshot_ddcobe.py). Ignored by the fusion-classifier path.
    ddcobe_combine: str = "dempster"  # dempster | mean_evidence | softmax_ensemble
    ddcobe_kl_lambda: float = 0.1
    ddcobe_kl_warmup_epochs: int = 10
    ddcobe_hidden_dim: int = 512
    ddcobe_head_type: str = "mlp"  # mlp (adapter+LN) | linear_ln (LayerNorm+Linear probe)
    ddcobe_head_pred_export_path: Optional[str] = None  # if set, save per-backbone-head test predictions (.npz)
    # Standard single-backbone uncertainty baselines (models/baselines_single.py,
    # run via scripts/sweep_kshot_baselines.py). Operate on ONE frozen backbone
    # embedding selected by baseline_backbone. Ignored by all other paths.
    baseline_method: str = "sensoy"       # sensoy | mc_dropout
    baseline_backbone: str = "ResNet18"   # which backbone embedding to use
    baseline_hidden_dim: int = 0          # 0 = single linear head; >0 = 1-hidden MLP
    mc_dropout_p: float = 0.5             # Gal & Ghahramani (2016) dropout rate
    mc_dropout_samples: int = 50          # T stochastic forward passes at test
    sensoy_evidence: str = "relu"         # relu | softplus | exp  (paper: relu)
    sensoy_loss: str = "mse"              # mse (Brier, paper default) | ce | digamma
    num_epochs: int = 10
    lr: float = 1e-3
    weight_decay_stage_a: float = 0.0
    weight_decay_stage_b: float = 0.0
    batch_size: int = 32
    pretrained: bool = False
    use_jkk: bool = True
    use_deepweeds: bool = False
    use_ssl_patches: bool = False
    use_imagenet1k_parquet: bool = False
    use_deepweeds_without_label8: bool = False
    use_deepweeds_merge_val_into_train: bool = False
    # Stage-D (fine-tune): optionally warm-start classifier from an existing checkpoint tag.
    # Fusion warm-start tag is passed via run_experiment(..., frozen_fusion_tag=...).
    stage_d_load_classifier: bool = False
    stage_d_classifier_tag: Optional[str] = None
    stage_d_classifier_added_postfix: Optional[str] = None
    stage_d_classifier_strict: bool = True
    num_classes: int = 2
    grad_clip_norm: float = 1.0
    embedding_num_workers: int = 0
    embedding_pin_memory: bool = True
    embedding_persistent_workers: bool = True
    embedding_prefetch_factor: int = 2
    train_log_every_steps: int = 1000

    dataset_root: str = "/home/magz94/ddcomb/data/ImageNet100/ImageNet100"
    ssl_patches_root: str = "/home/magz94/Uncertainty/ssl_patches"
    ssl_patches_train_split: str = "train"
    ssl_patches_eval_split: str = "val"
    ssl_patches_merge_split: str = "val"
    use_ssl_patches_merge_val_into_train: bool = False
    imagenet1k_parquet_root: str = "/datasets/imagenet-1k"
    imagenet1k_parquet_train_split: str = "train"
    imagenet1k_parquet_eval_split: str = "validation"
    yaml_path: str = "/datasets/ragweeds_final_dataset/split1_labeled.yaml"
    dataset_root_jkk: str = "/datasets/ragweeds_final_dataset"
    deepweeds_root: str = "DeepWeeds"
    deepweeds_subset: str = "subset0"
    base_models_path: Optional[str] = "models/base"
    emb_cache_root: str = "embedding_cache"
    img_plot_root: str = "imgs/plots"
    model_save_path: str = "models"

    subfolders: List[str] = field(default_factory=lambda: ["acc85"])
    model_names: List[str] = field(
        default_factory=lambda: ["ResNet18", "VGG16", "DenseNet121", "EfficientNetB0", "ViT-B16"]
    )
    shuffle_model_order: bool = False

    run_tag: str = "fusion_run"
    added_postfix: Optional[str] = None
    # Optional k-shot manifest path. When set, train split is filtered to listed samples.
    kshot_manifest_path: Optional[str] = None
    # Optional path for saving held-out labels/probabilities/logits after eval.
    # Used by k-shot PR-curve plotting; kept opt-in to avoid large extra files.
    prediction_output_path: Optional[str] = None
    enable_uncertainty_metrics: bool = False
    enable_diagnostic_plots: bool = False

    def __post_init__(self) -> None:
        if self.attention_method not in CANONICAL_ATTENTION_METHODS:
            raise ValueError(
                f"FusionPipelineConfig: attention_method={self.attention_method!r} is not canonical. "
                f"Allowed: {sorted(CANONICAL_ATTENTION_METHODS)}."
            )
        if self.classifier_method not in CANONICAL_CLASSIFIER_METHODS:
            raise ValueError(
                f"FusionPipelineConfig: classifier_method={self.classifier_method!r} is not canonical. "
                f"Allowed: {sorted(CANONICAL_CLASSIFIER_METHODS)}."
            )
        if self.classification_loss_mode not in CANONICAL_CLASSIFICATION_LOSS_MODES:
            raise ValueError(
                f"FusionPipelineConfig: classification_loss_mode={self.classification_loss_mode!r} "
                f"is not valid. Allowed: {sorted(CANONICAL_CLASSIFICATION_LOSS_MODES)}."
            )
        if self.classification_kl_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: classification_kl_lambda must be non-negative.")
        if self.classification_loss_warmup_epochs < 0:
            raise ValueError("FusionPipelineConfig: classification_loss_warmup_epochs must be non-negative.")
        if self.classification_fisher_power < 0.0:
            raise ValueError("FusionPipelineConfig: classification_fisher_power must be non-negative.")
        if self.dirichlet_kl_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: dirichlet_kl_lambda must be non-negative.")
        if self.dirichlet_kl_warmup_epochs < 0:
            raise ValueError("FusionPipelineConfig: dirichlet_kl_warmup_epochs must be non-negative.")
        if self.dirichlet_kl_regularization and self.attention_method != "dirichlet":
            raise ValueError(
                "FusionPipelineConfig: dirichlet_kl_regularization=True requires "
                "attention_method='dirichlet'."
            )
        if self.pre_fusion_uncertainty_hidden_dim <= 0:
            raise ValueError("FusionPipelineConfig: pre_fusion_uncertainty_hidden_dim must be positive.")
        if self.pre_fusion_uncertainty_eps <= 0.0:
            raise ValueError("FusionPipelineConfig: pre_fusion_uncertainty_eps must be positive.")
        if self.pre_fusion_aux_loss_weight < 0.0:
            raise ValueError("FusionPipelineConfig: pre_fusion_aux_loss_weight must be non-negative.")
        if self.pre_fusion_cali_loss_weight < 0.0:
            raise ValueError("FusionPipelineConfig: pre_fusion_cali_loss_weight must be non-negative.")
        if self.pre_fusion_cali_eps <= 0.0:
            raise ValueError("FusionPipelineConfig: pre_fusion_cali_eps must be positive.")
        if self.attention_loss_mode not in CANONICAL_ATTENTION_LOSS_MODES:
            raise ValueError(
                f"FusionPipelineConfig: attention_loss_mode={self.attention_loss_mode!r} is not valid. "
                f"Allowed: {sorted(CANONICAL_ATTENTION_LOSS_MODES)}."
            )
        if self.attention_loss_mode != "none" and self.attention_method != "dirichlet":
            raise ValueError("FusionPipelineConfig: attention_loss_mode requires attention_method='dirichlet'.")
        if self.attention_loss_warmup_epochs < 0:
            raise ValueError("FusionPipelineConfig: attention_loss_warmup_epochs must be non-negative.")
        if self.attention_entropy_sample_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: attention_entropy_sample_lambda must be non-negative.")
        if self.attention_entropy_batch_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: attention_entropy_batch_lambda must be non-negative.")
        if self.attention_ce_weighted_easy_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: attention_ce_weighted_easy_lambda must be non-negative.")
        if self.attention_ce_weighted_hard_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: attention_ce_weighted_hard_lambda must be non-negative.")
        if self.attention_ce_weighted_tau <= 0.0:
            raise ValueError("FusionPipelineConfig: attention_ce_weighted_tau must be positive.")
        if self.attention_head_diversity_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: attention_head_diversity_lambda must be non-negative.")
        if self.class_evidential_loss_mode not in CANONICAL_CLASS_EVIDENTIAL_LOSS_MODES:
            raise ValueError(
                "FusionPipelineConfig: "
                f"class_evidential_loss_mode={self.class_evidential_loss_mode!r} is not valid. "
                f"Allowed: {sorted(CANONICAL_CLASS_EVIDENTIAL_LOSS_MODES)}."
            )
        if self.class_evidential_wrong_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: class_evidential_wrong_lambda must be non-negative.")
        if self.class_evidential_margin_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: class_evidential_margin_lambda must be non-negative.")
        if self.class_evidential_margin < 0.0:
            raise ValueError("FusionPipelineConfig: class_evidential_margin must be non-negative.")
        if self.class_evidential_nll_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: class_evidential_nll_lambda must be non-negative.")
        if self.class_evidential_kl_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: class_evidential_kl_lambda must be non-negative.")
        if self.class_evidential_warmup_epochs < 0:
            raise ValueError("FusionPipelineConfig: class_evidential_warmup_epochs must be non-negative.")
        if self.class_evidential_eps <= 0.0:
            raise ValueError("FusionPipelineConfig: class_evidential_eps must be positive.")
        if self.use_dirichlet_logit_moe and self.attention_method != "dirichlet":
            raise ValueError("FusionPipelineConfig: use_dirichlet_logit_moe requires attention_method='dirichlet'.")
        if self.dirichlet_logit_moe_gamma < 0.0:
            raise ValueError("FusionPipelineConfig: dirichlet_logit_moe_gamma must be non-negative.")
        if self.dirichlet_logit_moe_aux_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: dirichlet_logit_moe_aux_lambda must be non-negative.")
        if self.dirichlet_logit_moe_resp_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: dirichlet_logit_moe_resp_lambda must be non-negative.")
        if self.dirichlet_logit_moe_tau <= 0.0:
            raise ValueError("FusionPipelineConfig: dirichlet_logit_moe_tau must be positive.")
        if self.dirichlet_logit_moe_warmup_epochs < 0:
            raise ValueError("FusionPipelineConfig: dirichlet_logit_moe_warmup_epochs must be non-negative.")
        if self.dirichlet_logit_moe_router not in CANONICAL_DIRICHLET_LOGIT_MOE_ROUTERS:
            raise ValueError(
                "FusionPipelineConfig: "
                f"dirichlet_logit_moe_router={self.dirichlet_logit_moe_router!r} is not valid. "
                f"Allowed: {sorted(CANONICAL_DIRICHLET_LOGIT_MOE_ROUTERS)}."
            )
        if self.dirichlet_logit_moe_router_beta < 0.0:
            raise ValueError("FusionPipelineConfig: dirichlet_logit_moe_router_beta must be non-negative.")
        if self.dirichlet_logit_moe_router in {"mixed", "support_ce_mixed"} and self.dirichlet_logit_moe_router_beta > 1.0:
            raise ValueError(
                "FusionPipelineConfig: dirichlet_logit_moe_router_beta must be in [0, 1] "
                "when dirichlet_logit_moe_router is mixed or support_ce_mixed."
            )
        if self.dirichlet_logit_moe_router_temperature <= 0.0:
            raise ValueError("FusionPipelineConfig: dirichlet_logit_moe_router_temperature must be positive.")
        if self.dirichlet_logit_moe_combine_mode not in CANONICAL_DIRICHLET_LOGIT_MOE_COMBINE_MODES:
            raise ValueError(
                "FusionPipelineConfig: "
                f"dirichlet_logit_moe_combine_mode={self.dirichlet_logit_moe_combine_mode!r} is not valid. "
                f"Allowed: {sorted(CANONICAL_DIRICHLET_LOGIT_MOE_COMBINE_MODES)}."
            )
        if self.dirichlet_logit_moe_dempster_rule not in CANONICAL_DIRICHLET_LOGIT_MOE_DEMPSTER_RULES:
            raise ValueError(
                "FusionPipelineConfig: "
                f"dirichlet_logit_moe_dempster_rule={self.dirichlet_logit_moe_dempster_rule!r} "
                f"is not valid. Allowed: {sorted(CANONICAL_DIRICHLET_LOGIT_MOE_DEMPSTER_RULES)}."
            )
        if self.dirichlet_logit_moe_dempster_loss_mode not in CANONICAL_DIRICHLET_LOGIT_MOE_DEMPSTER_LOSS_MODES:
            raise ValueError(
                "FusionPipelineConfig: "
                f"dirichlet_logit_moe_dempster_loss_mode={self.dirichlet_logit_moe_dempster_loss_mode!r} "
                f"is not valid. Allowed: {sorted(CANONICAL_DIRICHLET_LOGIT_MOE_DEMPSTER_LOSS_MODES)}."
            )
        if self.dirichlet_logit_moe_dempster_kl_lambda < 0.0:
            raise ValueError(
                "FusionPipelineConfig: dirichlet_logit_moe_dempster_kl_lambda must be non-negative."
            )
        if self.dirichlet_logit_moe_dempster_edl_lambda < 0.0:
            raise ValueError(
                "FusionPipelineConfig: dirichlet_logit_moe_dempster_edl_lambda must be non-negative."
            )
        if self.dirichlet_logit_moe_class_router_kl_lambda < 0.0:
            raise ValueError(
                "FusionPipelineConfig: dirichlet_logit_moe_class_router_kl_lambda must be non-negative."
            )
        if self.ddcobe_combine not in {"dempster", "mean_evidence", "softmax_ensemble"}:
            raise ValueError(
                "FusionPipelineConfig: ddcobe_combine must be one of: "
                "dempster, mean_evidence, softmax_ensemble."
            )
        if self.ddcobe_kl_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: ddcobe_kl_lambda must be non-negative.")
        if self.ddcobe_kl_warmup_epochs < 0:
            raise ValueError("FusionPipelineConfig: ddcobe_kl_warmup_epochs must be non-negative.")
        if self.ddcobe_head_type not in {"mlp", "linear_ln"}:
            raise ValueError("FusionPipelineConfig: ddcobe_head_type must be one of: mlp, linear_ln.")
        if self.baseline_method not in {"sensoy", "mc_dropout"}:
            raise ValueError("FusionPipelineConfig: baseline_method must be one of: sensoy, mc_dropout.")
        if not (0.0 <= self.mc_dropout_p < 1.0):
            raise ValueError("FusionPipelineConfig: mc_dropout_p must be in [0, 1).")
        if self.mc_dropout_samples < 1:
            raise ValueError("FusionPipelineConfig: mc_dropout_samples must be >= 1.")
        if self.sensoy_evidence not in {"relu", "softplus", "exp"}:
            raise ValueError("FusionPipelineConfig: sensoy_evidence must be one of: relu, softplus, exp.")
        if self.sensoy_loss not in {"mse", "ce", "digamma"}:
            raise ValueError("FusionPipelineConfig: sensoy_loss must be one of: mse, ce, digamma.")
        if self.use_mean_preserving_evidential_moe and not self.use_dirichlet_logit_moe:
            raise ValueError(
                "FusionPipelineConfig: use_mean_preserving_evidential_moe=True requires "
                "use_dirichlet_logit_moe=True."
            )
        if self.mp_evidential_moe_objective not in CANONICAL_EVIDENTIAL_FUSION_OBJECTIVES:
            raise ValueError(
                "FusionPipelineConfig: "
                f"mp_evidential_moe_objective={self.mp_evidential_moe_objective!r} is not valid. "
                f"Allowed: {sorted(CANONICAL_EVIDENTIAL_FUSION_OBJECTIVES)}."
            )
        if self.mp_evidential_moe_edl_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: mp_evidential_moe_edl_lambda must be non-negative.")
        if self.mp_evidential_moe_kl_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: mp_evidential_moe_kl_lambda must be non-negative.")
        if self.mp_evidential_moe_strength_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: mp_evidential_moe_strength_lambda must be non-negative.")
        if self.mp_evidential_moe_strength_scale <= 0.0:
            raise ValueError("FusionPipelineConfig: mp_evidential_moe_strength_scale must be positive.")
        if self.mp_evidential_moe_strength_tau <= 0.0:
            raise ValueError("FusionPipelineConfig: mp_evidential_moe_strength_tau must be positive.")
        if self.mp_evidential_moe_warmup_epochs < 0:
            raise ValueError("FusionPipelineConfig: mp_evidential_moe_warmup_epochs must be non-negative.")
        if self.mp_evidential_moe_hidden_dim <= 0:
            raise ValueError("FusionPipelineConfig: mp_evidential_moe_hidden_dim must be positive.")
        if self.mp_evidential_moe_eps <= 0.0:
            raise ValueError("FusionPipelineConfig: mp_evidential_moe_eps must be positive.")
        if self.use_evidential_fusion and self.attention_method != "dirichlet":
            raise ValueError("FusionPipelineConfig: use_evidential_fusion requires attention_method='dirichlet'.")
        if self.use_evidential_fusion and self.use_dirichlet_logit_moe:
            raise ValueError(
                "FusionPipelineConfig: use_evidential_fusion and use_dirichlet_logit_moe are mutually exclusive."
            )
        if self.use_evidential_fusion and self.classification_loss_mode != "cross_entropy":
            raise ValueError(
                "FusionPipelineConfig: use_evidential_fusion owns the primary evidential objective; "
                "keep classification_loss_mode='cross_entropy'."
            )
        if self.use_evidential_fusion and self.class_evidential_loss_mode != "none":
            raise ValueError(
                "FusionPipelineConfig: use_evidential_fusion cannot be combined with class_evidential_loss_mode."
            )
        if self.evidential_fusion_use_experts and not self.use_evidential_fusion:
            raise ValueError(
                "FusionPipelineConfig: evidential_fusion_use_experts requires use_evidential_fusion=True."
            )
        if self.evidential_fusion_loss_mode not in CANONICAL_EVIDENTIAL_FUSION_LOSS_MODES:
            raise ValueError(
                "FusionPipelineConfig: "
                f"evidential_fusion_loss_mode={self.evidential_fusion_loss_mode!r} is not valid. "
                f"Allowed: {sorted(CANONICAL_EVIDENTIAL_FUSION_LOSS_MODES)}."
            )
        if self.evidential_fusion_objective not in CANONICAL_EVIDENTIAL_FUSION_OBJECTIVES:
            raise ValueError(
                "FusionPipelineConfig: "
                f"evidential_fusion_objective={self.evidential_fusion_objective!r} is not valid. "
                f"Allowed: {sorted(CANONICAL_EVIDENTIAL_FUSION_OBJECTIVES)}."
            )
        if self.evidential_fusion_beta < 0.0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_beta must be non-negative.")
        if self.evidential_fusion_gamma < 0.0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_gamma must be non-negative.")
        if self.evidential_fusion_edl_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_edl_lambda must be non-negative.")
        if self.evidential_fusion_kl_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_kl_lambda must be non-negative.")
        if self.evidential_fusion_aux_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_aux_lambda must be non-negative.")
        if self.evidential_fusion_resp_lambda < 0.0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_resp_lambda must be non-negative.")
        if self.evidential_fusion_tau <= 0.0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_tau must be positive.")
        if self.evidential_fusion_warmup_epochs < 0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_warmup_epochs must be non-negative.")
        if self.evidential_fusion_eps <= 0.0:
            raise ValueError("FusionPipelineConfig: evidential_fusion_eps must be positive.")
        dataset_flags = {
            "use_deepweeds": self.use_deepweeds,
            "use_ssl_patches": self.use_ssl_patches,
            "use_imagenet1k_parquet": self.use_imagenet1k_parquet,
            "use_jkk": self.use_jkk,
        }
        active = [name for name, value in dataset_flags.items() if value]
        if len(active) > 1:
            raise ValueError(
                "FusionPipelineConfig: at most one dataset flag may be True, "
                f"got {active}. (Default ImageNet path is selected when all are False.)"
            )

    @property
    def emb_cache_path(self) -> str:
        return os.path.join(self.emb_cache_root, self.run_tag)

    @property
    def img_plot_path(self) -> str:
        base = os.path.join(self.img_plot_root, "jkk") if self.use_jkk else self.img_plot_root
        return os.path.join(base, self.run_tag)

    @property
    def deepweeds_images_dir(self) -> str:
        return os.path.join(self.deepweeds_root, "images")

    @property
    def deepweeds_labels_dir(self) -> str:
        return os.path.join(self.deepweeds_root, "labels")


def _suffix(tag: str, postfix: str, added_postfix: Optional[str] = None) -> str:
    if added_postfix:
        return f"{tag}_{added_postfix}_{postfix}"
    return f"{tag}_{postfix}"


VALID_FUSION_MODES = frozenset({"stage_a", "stage_b", "stage_d"})

VALID_SINGLE_BACKBONE_MODELS = frozenset(
    {"resnet18", "vgg16", "densenet121", "efficientnet_b0", "vit_b16", "convnext_tiny"}
)
VALID_SINGLE_BACKBONE_DATASETS = frozenset(
    {
        "imagenet100",
        "deepweeds_no_bg",
        "plant_vs_meadow",
        "ninco_ood_split",
        "pathmnist_224",
        "bloodmnist_224",
        "aptos2019",
        "plantvillage",
        "cassava_leaf_disease",
        "leafsugar_disease",
        "plant_seedlings_v2",
        "sar_cld_original",
    }
)
VALID_SINGLE_BACKBONE_TRAIN_MODES = frozenset({"full_finetune", "head_only"})
VALID_SINGLE_BACKBONE_PREPROCESS = frozenset({"basic", "weights"})
VALID_SINGLE_BACKBONE_OPTIMIZERS = frozenset({"adam", "adamw"})

VALID_MANIFEST_DATASET_TYPES = frozenset({"deepweeds_csv", "imagefolder"})


@dataclass
class TrainFusionConfig:
    """Top-level Hydra config for scripts/train_fusion.py.

    Wraps a FusionPipelineConfig with the orchestration arguments accepted by
    pipeline.run_experiment (mode, frozen_fusion_tag, skip_train, ...).
    """

    pipeline: FusionPipelineConfig = field(default_factory=FusionPipelineConfig)
    mode: str = "stage_a"
    frozen_fusion_tag: Optional[str] = None
    frozen_fusion_added_postfix: Optional[str] = None
    skip_train: bool = False
    return_full_metrics: bool = False

    def __post_init__(self) -> None:
        if self.mode not in VALID_FUSION_MODES:
            raise ValueError(
                f"TrainFusionConfig: mode={self.mode!r} is not valid. "
                f"Allowed: {sorted(VALID_FUSION_MODES)}."
            )


@dataclass
class SingleBackboneConfig:
    """Top-level Hydra config for scripts/train_single_backbone.py.

    Mirrors training.single_backbone.parse_args. Path-typed fields are stored as
    strings and converted at the call site.
    """

    model: str = MISSING
    # Avoid Hydra's `dataset` config-group name used by fusion experiments.
    dataset_name: str = MISSING
    mode: str = MISSING
    preprocess: str = "basic"
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    lr: Optional[float] = None
    weight_decay: float = 0.0
    optimizer: str = "adam"
    label_smoothing: float = 0.0
    amp: bool = False
    merge_val_into_train: bool = False

    deepweeds_root: str = "/datasets/DeepWeeds"
    deepweeds_subset: str = "subset0"
    imagenet100_root: str = "/home/magz94/ddcomb/data/ImageNet100/ImageNet100"
    ssl_patches_root: str = "/home/magz94/Uncertainty/ssl_patches"
    ninco_ood_root: str = "/home/magz94/ddcomb/data/ninco_ood_split"
    pathmnist_root: str = "/home/magz94/ddcomb/data/pathmnist_224_imagefolder"
    bloodmnist_root: str = "/home/magz94/ddcomb/data/bloodmnist_224_imagefolder"
    aptos_root: str = "/home/magz94/ddcomb/data/aptos2019_split"
    plantvillage_root: str = "/home/magz94/ddcomb/data/plantvillage_color_split"
    cassava_root: str = "/home/magz94/ddcomb/data/cassava_leaf_disease_split"
    leafsugar_root: str = "/home/magz94/ddcomb/data/leafsugar_disease_split"
    plant_seedlings_root: str = "/home/magz94/ddcomb/data/plant_seedlings_v2_split"
    sar_cld_root: str = "/home/magz94/ddcomb/data/sar_cld_original_split"

    output_dir: str = "models/single_backbone"
    export_base_model_path: Optional[str] = None
    # Optional ImageNet-100 EmbeddingModel checkpoint initialisation for
    # comparing single backbones against the same /models/base sources as fusion.
    init_base_model_path: Optional[str] = None
    init_base_models_path: Optional[str] = None
    init_base_subfolder: Optional[str] = None
    run_name: Optional[str] = None
    kshot_manifest_path: Optional[str] = None
    prediction_output_path: Optional[str] = None
    enable_uncertainty_metrics: bool = False

    use_wandb: bool = False
    wandb_project: str = "single_backbone_finetune"
    wandb_entity: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model not in VALID_SINGLE_BACKBONE_MODELS:
            raise ValueError(
                f"SingleBackboneConfig: model={self.model!r} is not valid. "
                f"Allowed: {sorted(VALID_SINGLE_BACKBONE_MODELS)}."
            )
        if self.dataset_name not in VALID_SINGLE_BACKBONE_DATASETS:
            raise ValueError(
                f"SingleBackboneConfig: dataset_name={self.dataset_name!r} is not valid. "
                f"Allowed: {sorted(VALID_SINGLE_BACKBONE_DATASETS)}."
            )
        if self.mode not in VALID_SINGLE_BACKBONE_TRAIN_MODES:
            raise ValueError(
                f"SingleBackboneConfig: mode={self.mode!r} is not valid. "
                f"Allowed: {sorted(VALID_SINGLE_BACKBONE_TRAIN_MODES)}."
            )
        if self.preprocess not in VALID_SINGLE_BACKBONE_PREPROCESS:
            raise ValueError(
                f"SingleBackboneConfig: preprocess={self.preprocess!r} is not valid. "
                f"Allowed: {sorted(VALID_SINGLE_BACKBONE_PREPROCESS)}."
            )
        if self.optimizer not in VALID_SINGLE_BACKBONE_OPTIMIZERS:
            raise ValueError(
                f"SingleBackboneConfig: optimizer={self.optimizer!r} is not valid. "
                f"Allowed: {sorted(VALID_SINGLE_BACKBONE_OPTIMIZERS)}."
            )


@dataclass
class SingleBackboneKShotSweepConfig(SingleBackboneConfig):
    """Hydra config for scripts/sweep_single_backbone_kshot.py.

    Extends the regular single-backbone config with the sweep dimensions
    needed to select one shared k-shot manifest per Hydra job.
    """

    output_dir: str = "models/single_backbone_kshot"
    wandb_project: str = "single_backbone_pretrained_kshot"

    kshot_root: str = "kshot/manifests"
    kshot_results_root: str = "kshot/results_single_backbone"
    kshot_dataset_key: str = MISSING
    kshot_shot: int = MISSING
    kshot_run_idx: int = MISSING
    kshot_base_seed: int = 42
    kshot_expected_runs: int = 20
    skip_existing_kshot_results: bool = True
    export_kshot_predictions: bool = False


@dataclass
class KShotSweepConfig:
    """Hydra config for scripts/sweep_kshot.py — multirun k-shot fusion training.

    Wraps a FusionPipelineConfig with the run_experiment orchestration knobs
    plus the k-shot sweep dimensions (kshot_root, kshot_dataset_key, kshot_shot,
    kshot_run_idx) that drive a Hydra ``-m`` multirun. ``conf/sweep_kshot.yaml``
    interpolates ``pipeline.kshot_manifest_path`` and ``pipeline.run_tag`` from
    these fields, so each per-job config is fully determined by
    ``kshot_dataset_key``, ``kshot_shot``, and ``kshot_run_idx``.
    """

    pipeline: FusionPipelineConfig = field(default_factory=FusionPipelineConfig)
    mode: str = "stage_d"
    frozen_fusion_tag: Optional[str] = None
    frozen_fusion_added_postfix: Optional[str] = None
    return_full_metrics: bool = True

    kshot_root: str = "kshot/manifests"
    kshot_results_root: str = "kshot/results"
    kshot_dataset_key: str = MISSING
    kshot_shot: int = MISSING
    kshot_run_idx: int = MISSING
    kshot_base_seed: int = 42
    kshot_expected_runs: int = 20
    skip_existing_kshot_results: bool = True
    export_kshot_predictions: bool = False

    def __post_init__(self) -> None:
        if self.mode not in VALID_FUSION_MODES:
            raise ValueError(
                f"KShotSweepConfig: mode={self.mode!r} is not valid. "
                f"Allowed: {sorted(VALID_FUSION_MODES)}."
            )


@dataclass
class KShotManifestConfig:
    """Hydra config for scripts/build_kshot_manifests.py.

    Drives a single call to ``data.kshot.ensure_kshot_manifests``. Dataset-specific
    fields are grouped under ``deepweeds_*`` / ``imagefolder_*`` prefixes; only
    the prefix matching ``dataset_type`` is used.
    """

    dataset_key: str = MISSING
    dataset_type: str = MISSING
    shots: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    repeats: int = 20
    base_seed: int = 42
    manifest_root: str = "kshot/manifests"
    force_resample: bool = False

    # When dataset_type == "deepweeds_csv":
    deepweeds_labels_dir: Optional[str] = None
    deepweeds_subset: str = "subset0"
    deepweeds_drop_label8: bool = True
    deepweeds_merge_val_into_train: bool = True

    # When dataset_type == "imagefolder":
    imagefolder_root: Optional[str] = None
    imagefolder_train_split: str = "train"
    imagefolder_merge_split: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dataset_type not in VALID_MANIFEST_DATASET_TYPES:
            raise ValueError(
                f"KShotManifestConfig: dataset_type={self.dataset_type!r} is not valid. "
                f"Allowed: {sorted(VALID_MANIFEST_DATASET_TYPES)}."
            )
        if self.dataset_type == "deepweeds_csv" and self.deepweeds_labels_dir is None:
            raise ValueError(
                "KShotManifestConfig: dataset_type='deepweeds_csv' requires deepweeds_labels_dir."
            )
        if self.dataset_type == "imagefolder" and self.imagefolder_root is None:
            raise ValueError(
                "KShotManifestConfig: dataset_type='imagefolder' requires imagefolder_root."
            )


def register_configs() -> None:
    """Register the structured configs with Hydra's ConfigStore.

    Idempotent — call once at the top of each @hydra.main entry-point module.
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="train_fusion_schema", node=TrainFusionConfig)
    cs.store(name="train_single_backbone_schema", node=SingleBackboneConfig)
    cs.store(name="single_backbone_kshot_sweep_schema", node=SingleBackboneKShotSweepConfig)
    cs.store(name="kshot_sweep_schema", node=KShotSweepConfig)
    cs.store(name="kshot_manifest_schema", node=KShotManifestConfig)
    cs.store(group="pipeline", name="fusion_pipeline_schema", node=FusionPipelineConfig)
