from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import wandb

from ddcomb.config import FusionPipelineConfig
from ddcomb.models.fusion import FusionClassifier


def _update_confusion_matrix(
    cm: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int,
) -> None:
    idx = labels.long() * num_classes + preds.long()
    bincount = torch.bincount(idx.detach().cpu(), minlength=num_classes * num_classes)
    cm += bincount.reshape(num_classes, num_classes)


def _macro_precision_recall(cm: torch.Tensor) -> tuple[float, float]:
    tp = cm.diag().float()
    pred_pos = cm.sum(dim=0).float().clamp(min=1.0)
    gt_pos = cm.sum(dim=1).float().clamp(min=1.0)
    precision_macro = (tp / pred_pos).mean().item()
    recall_macro = (tp / gt_pos).mean().item()
    return precision_macro, recall_macro


def _needs_fusion_extras(cfg: FusionPipelineConfig) -> bool:
    return (
        (cfg.dirichlet_kl_regularization and cfg.dirichlet_kl_lambda > 0.0)
        or cfg.attention_loss_mode != "none"
        or cfg.use_pre_fusion_uncertainty
        or cfg.use_dirichlet_logit_moe
        or cfg.use_evidential_fusion
    )


@torch.no_grad()
def calibrate_dirichlet_logit_moe_support_router(
    model: FusionClassifier,
    dataloader,
    cfg: FusionPipelineConfig,
    device: str = "cpu",
) -> None:
    router_mode = cfg.dirichlet_logit_moe_router
    if not (model.use_dirichlet_logit_moe and router_mode in {"support_ce", "support_ce_mixed"}):
        return

    model.to(device)
    model.eval()
    ce_sum = None
    total = 0
    num_experts = len(model.expert_heads)
    if num_experts <= 0:
        raise RuntimeError("Support-based DG-MoBE router requires expert heads.")

    for batch_embeddings, batch_labels in dataloader:
        batch_embeddings = [e.to(device) for e in batch_embeddings]
        batch_labels = batch_labels.to(device)
        fused, _, _, extras = model.fusion_model(batch_embeddings, return_extras=True)
        expert_tokens = getattr(extras, "expert_tokens", None)
        if expert_tokens is None:
            raise RuntimeError("Support-based router calibration requires expert_tokens in fusion extras.")
        expert_logits = torch.stack([
            head(expert_tokens[:, idx, :])
            for idx, head in enumerate(model.expert_heads)
        ], dim=1)
        batch_size, _, num_classes = expert_logits.shape
        expanded_targets = batch_labels.unsqueeze(1).expand(-1, num_experts).reshape(-1)
        expert_ce = F.cross_entropy(
            expert_logits.reshape(batch_size * num_experts, num_classes),
            expanded_targets,
            reduction="none",
        ).view(batch_size, num_experts)
        batch_ce_sum = expert_ce.sum(dim=0)
        ce_sum = batch_ce_sum if ce_sum is None else ce_sum + batch_ce_sum
        total += batch_size

    if ce_sum is None or total <= 0:
        raise RuntimeError("Cannot calibrate support-based router on an empty dataloader.")

    mean_ce = ce_sum / float(total)
    temperature = max(float(cfg.dirichlet_logit_moe_router_temperature), 1e-12)
    router = torch.softmax(-mean_ce / temperature, dim=0)
    if router_mode == "support_ce_mixed":
        beta = float(cfg.dirichlet_logit_moe_router_beta)
        uniform = router.new_full(router.shape, 1.0 / float(num_experts))
        router = (1.0 - beta) * uniform + beta * router
    model.set_dirichlet_logit_moe_support_router(router.detach().cpu())
    print(
        "[DG-MoBE] Support-CE router calibrated: "
        f"mode={router_mode}, "
        f"temperature={temperature:g}, beta={cfg.dirichlet_logit_moe_router_beta:g}, "
        f"mean_ce={[round(float(v), 6) for v in mean_ce.detach().cpu()]}, "
        f"router={[round(float(v), 6) for v in router.detach().cpu()]}",
        flush=True,
    )


def train_fusion_classifier(
    model: FusionClassifier,
    dataloader,
    cfg: FusionPipelineConfig,
    classifier_ckpt_path: Optional[str] = None,
    device: str = "cpu",
    weight_decay: float = 0.0,
):
    model.to(device)
    model.train()
    fusion_forward_model = model.fusion_model

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=weight_decay,
    )

    for epoch in range(cfg.num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(dataloader)
        cm = torch.zeros((cfg.num_classes, cfg.num_classes), dtype=torch.long)

        for step, (batch_embeddings, batch_labels) in enumerate(dataloader, start=1):
            batch_embeddings = [e.to(device) for e in batch_embeddings]
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            if _needs_fusion_extras(cfg):
                fused, _, _, extras = fusion_forward_model(batch_embeddings, return_extras=True)
                cls_out = model.classify_fused(fused, extras=extras)
            else:
                fused, _, _ = fusion_forward_model(batch_embeddings)
                cls_out = model.classify_fused(fused)
            probs = cls_out["probs"]
            loss = model.compute_loss(cls_out, batch_labels, is_train=True, epoch_idx=epoch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            preds = probs.argmax(dim=1)
            running_loss += loss.item() * batch_labels.size(0)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            _update_confusion_matrix(cm, batch_labels, preds, cfg.num_classes)

            if cfg.train_log_every_steps > 0 and (step % cfg.train_log_every_steps == 0 or step == num_batches):
                print(
                    f"[Train] epoch {epoch + 1}/{cfg.num_epochs} "
                    f"step {step}/{num_batches} "
                    f"running_loss={(running_loss / max(total, 1)):.6f} "
                    f"running_acc={(correct / max(total, 1)):.4f}",
                    flush=True,
                )

        epoch_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        precision, recall = _macro_precision_recall(cm)
        if cfg.use_wandb:
            log_payload = {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
            }
            log_payload.update({f"train_{key}": value for key, value in model.last_loss_terms.items()})
            wandb.log(log_payload)

    if classifier_ckpt_path is not None:
        torch.save(model.state_dict(), classifier_ckpt_path)
    return model


def evaluate_classifier_basic(
    model: FusionClassifier,
    dataloader,
    num_classes: int,
    device: str = "cpu",
    epoch_idx: int = 0,
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    fusion_forward_model = model.fusion_model

    running_loss = 0.0
    correct = 0
    total = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings = [e.to(device) for e in batch_embeddings]
            batch_labels = batch_labels.to(device)

            if model.use_dirichlet_logit_moe or model.use_evidential_fusion:
                fused, _, _, extras = fusion_forward_model(batch_embeddings, return_extras=True)
                cls_out = model.classify_fused(fused, extras=extras)
            else:
                fused, _, _ = fusion_forward_model(batch_embeddings)
                cls_out = model.classify_fused(fused)
            probs = cls_out["probs"]
            loss = model.compute_loss(cls_out, batch_labels, is_train=False, epoch_idx=epoch_idx)

            preds = probs.argmax(dim=1)
            running_loss += loss.item() * batch_labels.size(0)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            _update_confusion_matrix(cm, batch_labels, preds, num_classes)

    precision, recall = _macro_precision_recall(cm)

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
    }


def train_fusion_classifier_with_val_selection(
    model: FusionClassifier,
    train_dataloader,
    val_dataloader,
    cfg: FusionPipelineConfig,
    classifier_ckpt_path: str,
    device: str = "cpu",
    weight_decay: float = 0.0,
):
    """Train with best-val checkpoint selection.

    Originally Stage-B-only (when fusion_model is frozen and only the
    classifier head trains); also valid for Stage-A and Stage-D when a
    val_loader_emb is available, since the classifier_ckpt_path stores the
    full FusionClassifier state including the fusion_model that's currently
    being trained.
    """
    model.to(device)
    fusion_forward_model = model.fusion_model
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=weight_decay,
    )

    best_val_accuracy = float("-inf")
    best_epoch = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        cm = torch.zeros((cfg.num_classes, cfg.num_classes), dtype=torch.long)

        for batch_embeddings, batch_labels in train_dataloader:
            batch_embeddings = [e.to(device) for e in batch_embeddings]
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            if _needs_fusion_extras(cfg):
                fused, _, _, extras = fusion_forward_model(batch_embeddings, return_extras=True)
                cls_out = model.classify_fused(fused, extras=extras)
            else:
                fused, _, _ = fusion_forward_model(batch_embeddings)
                cls_out = model.classify_fused(fused)
            probs = cls_out["probs"]
            loss = model.compute_loss(cls_out, batch_labels, is_train=True, epoch_idx=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            preds = probs.argmax(dim=1)
            running_loss += loss.item() * batch_labels.size(0)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            _update_confusion_matrix(cm, batch_labels, preds, cfg.num_classes)

        train_loss = running_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)
        train_precision, train_recall = _macro_precision_recall(cm)

        val_metrics = evaluate_classifier_basic(
            model=model,
            dataloader=val_dataloader,
            num_classes=cfg.num_classes,
            device=device,
            epoch_idx=epoch,
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), classifier_ckpt_path)

        if cfg.use_wandb:
            log_payload = {
                "epoch": epoch + 1,
                "loss": train_loss,
                "accuracy": train_accuracy,
                "precision": train_precision,
                "recall": train_recall,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
            }
            log_payload.update({f"train_{key}": value for key, value in model.last_loss_terms.items()})
            wandb.log(log_payload)

    print(f"[Stage B] Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(classifier_ckpt_path, map_location=device))
    return model
