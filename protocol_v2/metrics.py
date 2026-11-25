"""Classification and calibration metrics used by Protocol V2."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        if in_bin.any():
            weight = in_bin.float().mean()
            accuracy = correct[in_bin].float().mean()
            confidence = confidences[in_bin].mean()
            ece = ece + weight * (accuracy - confidence).abs()
    return float(ece.item())


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    probs = F.softmax(logits, dim=1)
    predictions = probs.argmax(dim=1)
    labels_np = labels.numpy()
    predictions_np = predictions.numpy()

    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
    nll = F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels)
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean()
    error_score = 1.0 - probs.max(dim=1).values
    errors = predictions.ne(labels).numpy().astype(np.int8)
    if np.unique(errors).size == 2:
        error_auroc = float(roc_auc_score(errors, error_score.numpy()))
    else:
        error_auroc = float("nan")

    return {
        "accuracy": float(predictions.eq(labels).float().mean().item()),
        "f1_macro": float(f1_score(labels_np, predictions_np, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels_np, predictions_np, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(labels_np, predictions_np)),
        "nll": float(nll.item()),
        "brier": float(brier.item()),
        "ece": expected_calibration_error(probs, labels),
        "error_auroc": error_auroc,
    }
