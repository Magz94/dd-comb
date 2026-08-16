"""ddcomb: multi-backbone uncertainty-aware attention fusion."""

from ddcomb.config import FusionPipelineConfig
from ddcomb.data.datasets import DeepWeedsCSVDataset
from ddcomb.data.dataloaders import load_models_and_dataloaders_for_postfix
from ddcomb.evaluation.evaluate import evaluate_fusion_classifier
from ddcomb.pipeline import run_experiment, run_single_postfix
from ddcomb.runtime import _checkpoint_path, checkpoint_path, init_runtime
from ddcomb.training.build import build_fusion_model

__all__ = [
    "FusionPipelineConfig",
    "DeepWeedsCSVDataset",
    "build_fusion_model",
    "checkpoint_path",
    "evaluate_fusion_classifier",
    "init_runtime",
    "load_models_and_dataloaders_for_postfix",
    "run_experiment",
    "run_single_postfix",
    "_checkpoint_path",  # alias of checkpoint_path; preserved for legacy imports
]
