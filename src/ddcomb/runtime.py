from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from ddcomb.config import FusionPipelineConfig, _suffix


# On-disk checkpoint filename prefixes. The full filename is
#   f"{prefix}_{_suffix(tag, postfix, added_postfix)}.pt"
# (see _checkpoint_path / checkpoint_path below).
FUSION_MODEL_CKPT_PREFIX = "attn_fusion_model"
FUSION_CLASSIFIER_CKPT_PREFIX = "attn_fusion_classifier"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using {seed} as seed")


def checkpoint_path(
    model_save_path: str,
    prefix: str,
    tag: str,
    postfix: str,
    added_postfix: Optional[str] = None,
) -> str:
    """Return the full on-disk path of a fusion model or classifier checkpoint."""
    return os.path.join(model_save_path, f"{prefix}_{_suffix(tag, postfix, added_postfix)}.pt")


# Backwards-compatible alias for external code (e.g. evaluation/eval_uncertainty.py)
# that imported the underscore-prefixed name. The leading underscore was misleading:
# it was always part of the public surface.
_checkpoint_path = checkpoint_path


def init_runtime(cfg: FusionPipelineConfig) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} as device")
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        n_cuda = torch.cuda.device_count()
        print(f"[CUDA] CUDA_VISIBLE_DEVICES={visible}")
        print(f"[CUDA] Detected {n_cuda} CUDA device(s).")
        for idx in range(n_cuda):
            print(f"[CUDA] cuda:{idx} -> {torch.cuda.get_device_name(idx)}")
    set_seed(cfg.seed)

    if cfg.use_wandb:
        wandb_key = os.getenv("WANDB_API_KEY")
        if not wandb_key:
            raise ValueError("WANDB_API_KEY is not set.")
        wandb.login(key=wandb_key)

    return device


def resolve_model_order(cfg: FusionPipelineConfig) -> List[str]:
    ordered = list(cfg.model_names)
    if cfg.shuffle_model_order:
        rng = random.Random(cfg.seed)
        rng.shuffle(ordered)
    return ordered


def _build_embedding_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    workers = max(int(num_workers), 0)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)
