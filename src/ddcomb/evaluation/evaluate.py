from __future__ import annotations

import torch

from ddcomb.models.fusion import FusionClassifier


def _macro_precision_recall(labels: torch.Tensor, probs: torch.Tensor, num_classes: int) -> tuple[float, float]:
    preds = probs.argmax(dim=1).detach().cpu()
    labels = labels.detach().cpu()
    idx = labels.long() * num_classes + preds.long()
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    tp = cm.diag().float()
    pred_pos = cm.sum(dim=0).float().clamp(min=1.0)
    gt_pos = cm.sum(dim=1).float().clamp(min=1.0)
    precision_macro = (tp / pred_pos).mean().item()
    recall_macro = (tp / gt_pos).mean().item()
    return precision_macro, recall_macro


def _append_extra_tensor(values, extras, name: str) -> None:
    tensor = getattr(extras, name, None)
    if tensor is not None:
        values.append(tensor.detach().cpu())


def evaluate_fusion_classifier(
    model: FusionClassifier,
    dataloader,
    num_classes: int,
    device: str = "cpu",
    return_embeddings: bool = False,
    return_attention: bool = False,
    return_extras: bool = False,
):
    model.to(device)
    model.eval()
    return_extras = return_extras or model.use_dirichlet_logit_moe or model.use_evidential_fusion

    embeddings_list, labels_list = [], []
    attn_list, pool_list = [], []
    logits_all, probs_all, labels_all = [], [], []
    ent_all, dir_all, dir_u_all = [], [], []
    pref_u_all, pref_w_all = [], []
    expert_logits_all, router_probs_all = [], []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings = [e.to(device) for e in batch_embeddings]
            batch_labels = batch_labels.to(device)

            extras = None
            if return_attention or return_extras:
                if return_extras:
                    fused, attn_weights, pool_weights, extras = model.fusion_model(batch_embeddings, return_extras=True)
                else:
                    fused, attn_weights, pool_weights = model.fusion_model(batch_embeddings)

                if return_attention:
                    attn_list.append(attn_weights.detach().cpu())
                    pool_list.append(pool_weights.detach().cpu())

                if return_extras and extras is not None:
                    _append_extra_tensor(ent_all, extras, "attn_entropy")
                    _append_extra_tensor(dir_all, extras, "dirichlet_strength")
                    _append_extra_tensor(dir_u_all, extras, "dirichlet_uncertainty")
                    _append_extra_tensor(pref_u_all, extras, "prefusion_uncertainty")
                    _append_extra_tensor(pref_w_all, extras, "prefusion_weights")
            else:
                fused, _, _ = model.fusion_model(batch_embeddings)

            cls_out = model.classify_fused(fused, extras=extras)
            logits = cls_out["logits"]
            probs = cls_out["probs"]
            if cls_out.get("expert_logits") is not None:
                expert_logits_all.append(cls_out["expert_logits"].detach().cpu())
            if cls_out.get("router_probs") is not None:
                router_probs_all.append(cls_out["router_probs"].detach().cpu())

            correct += (probs.argmax(dim=1) == batch_labels).sum().item()
            total += batch_labels.size(0)

            logits_all.append(logits.detach().cpu())
            probs_all.append(probs.detach().cpu())
            labels_all.append(batch_labels.detach().cpu())

            if return_embeddings:
                embeddings_list.append(fused.detach().cpu())
                labels_list.append(batch_labels.detach().cpu())

    logits_all_tensor = torch.cat(logits_all, dim=0)
    probs_all_tensor = torch.cat(probs_all, dim=0)
    labels_all_tensor = torch.cat(labels_all, dim=0)
    precision_macro, recall_macro = _macro_precision_recall(
        labels=labels_all_tensor,
        probs=probs_all_tensor,
        num_classes=num_classes,
    )

    results = {
        "accuracy": correct / max(total, 1),
        "precision": precision_macro,
        "recall": recall_macro,
        "logits": logits_all_tensor,
        "probs": probs_all_tensor,
        "labels": labels_all_tensor,
    }

    if return_embeddings:
        results["embeddings"] = torch.cat(embeddings_list, dim=0)
        results["emb_labels"] = torch.cat(labels_list, dim=0)

    if return_attention:
        pool_all = torch.cat(pool_list, dim=0)
        results["attn_weights_mean"] = torch.cat(attn_list, dim=0).mean(dim=0, keepdim=True)
        results["pool_weights_mean"] = pool_all.mean(dim=0, keepdim=True)
        results["pool_weights"] = pool_all

    if return_extras and len(ent_all) > 0:
        results["attn_entropy"] = torch.cat(ent_all, dim=0)
    if return_extras and len(dir_all) > 0:
        results["dirichlet_strength"] = torch.cat(dir_all, dim=0)
    if return_extras and len(dir_u_all) > 0:
        results["dirichlet_uncertainty"] = torch.cat(dir_u_all, dim=0)
    if return_extras and len(pref_u_all) > 0:
        results["prefusion_uncertainty"] = torch.cat(pref_u_all, dim=0)
    if return_extras and len(pref_w_all) > 0:
        results["prefusion_weights"] = torch.cat(pref_w_all, dim=0)
    if len(expert_logits_all) > 0:
        results["expert_logits"] = torch.cat(expert_logits_all, dim=0)
    if len(router_probs_all) > 0:
        results["router_probs"] = torch.cat(router_probs_all, dim=0)

    return results
