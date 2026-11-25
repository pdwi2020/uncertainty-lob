"""Model construction and architecture-specific loss handling."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.models.deeplob import DeepLOB
from src.models.evlob import (
    EvLOB,
    edl_bayes_risk_loss,
    edl_ce_warmup_bayes_loss,
    edl_ce_warmup_loss,
    edl_uce_loss,
)
from src.models.lob_sde import LOB_SDE, lob_sde_loss


SUPPORTED_MODELS = {
    "cnn_lstm_attention",
    "cnn_lstm_no_attention",
    "deeplob",
    "evlob",
    "lob_sde",
}


def build_model(config: Dict[str, Any], n_micro: int) -> nn.Module:
    name = str(config["model"])
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {name!r}; choose from {sorted(SUPPORTED_MODELS)}")
    dropout = float(config.get("dropout", 0.2))
    hidden = int(config.get("hidden", 64))

    if name in {"cnn_lstm_attention", "cnn_lstm_no_attention"}:
        return CNNLSTMAttention(
            input_channels=40,
            cnn_filters=[32, 64, 64],
            lstm_hidden=hidden,
            lstm_layers=2,
            dropout=dropout,
            n_classes=3,
            use_attention=name == "cnn_lstm_attention",
        )
    if name == "deeplob":
        return DeepLOB(
            input_channels=40,
            cnn_out_channels=96,
            lstm_hidden=hidden,
            n_classes=3,
            dropout=dropout,
        )
    if name == "evlob":
        return EvLOB(
            input_channels=40,
            lstm_hidden=hidden,
            n_classes=3,
            n_micro=n_micro,
            gate_hidden=int(config.get("gate_hidden", 32)),
            dropout=dropout,
        )
    return LOB_SDE(
        input_channels=40,
        lstm_hidden=hidden,
        n_classes=3,
        n_micro=n_micro,
        latent_dim=int(config.get("latent_dim", 64)),
        sde_hidden=int(config.get("sde_hidden", 64)),
        dropout=dropout,
        n_rollouts=int(config.get("n_rollouts", 20)),
        dt=float(config.get("dt", 0.05)),
    )


def forward_for_training(
    model: nn.Module,
    model_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    micro: Optional[torch.Tensor],
    epoch: int,
    epochs: int,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model_name in {"cnn_lstm_attention", "cnn_lstm_no_attention", "deeplob"}:
        logits = model(x)
        return logits, F.cross_entropy(logits, y)

    if model_name == "evlob":
        alpha, _, _, logits = model(x, micro)
        loss_name = str(config.get("loss_fn", "bayes_risk"))
        lambda_kl = float(config.get("lambda_kl", 1.0))
        warm_epochs = int(config.get("warm_epochs", 20))
        if loss_name == "bayes_risk":
            loss = edl_bayes_risk_loss(alpha, y, epoch, epochs, lambda_kl)
        elif loss_name == "ce_warmup_bayes":
            loss = edl_ce_warmup_bayes_loss(alpha, y, epoch, epochs, lambda_kl, warm_epochs)
        elif loss_name == "uce":
            loss = edl_uce_loss(alpha, y, epoch, epochs, lambda_kl)
        elif loss_name == "ce_warmup":
            loss = edl_ce_warmup_loss(alpha, y, epoch, epochs, lambda_kl, warm_epochs)
        else:
            raise ValueError(f"Unsupported EvLOB loss {loss_name!r}")
        return logits, loss

    logits, path_energy = model(x, micro)
    beta = float(config.get("path_energy_beta", 1e-4))
    return logits, lob_sde_loss(logits, path_energy, y, beta=beta)


@torch.no_grad()
def forward_for_evaluation(
    model: nn.Module,
    model_name: str,
    x: torch.Tensor,
    micro: Optional[torch.Tensor],
    use_rollouts: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if model_name in {"cnn_lstm_attention", "cnn_lstm_no_attention", "deeplob"}:
        return model(x), {}
    if model_name == "evlob":
        alpha, strength, probabilities, logits = model(x, micro)
        return logits, {
            "alpha": alpha,
            "strength": strength.squeeze(1),
            "vacuity": model.n_classes / strength.squeeze(1),
            "probabilities": probabilities,
        }
    if use_rollouts:
        prediction = model.predict(x, micro)
        return prediction["logits"], {
            "aleatoric": prediction["aleatoric"],
            "epistemic": prediction["epistemic"],
            "h1_energy": prediction["h1_energy"],
        }
    logits, path_energy = model(x, micro)
    return logits, {"h1_energy": path_energy}
