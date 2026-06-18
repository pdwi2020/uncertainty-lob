"""MC Dropout for epistemic uncertainty estimation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class MCDropout:
    """MC Dropout wrapper: enable dropout at inference for T forward passes.

    Uncertainty estimates:
      - Predictive entropy: H[p̄] = -Σ p̄ log p̄
      - Epistemic uncertainty (mutual information): H[p̄] - E[H[p_t]]
    """

    def __init__(self, model: nn.Module, n_samples: int = 20):
        self.model = model
        self.n_samples = n_samples

    def predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MC Dropout inference.

        Returns:
            mean_probs: (B, n_classes)
            predictive_entropy: (B,)
            epistemic_uncertainty: (B,) mutual information
        """
        self.model.eval()
        self.model.enable_dropout()

        probs_samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs_samples.append(torch.softmax(logits, dim=-1))

        probs_stack = torch.stack(probs_samples)           # (S, B, C)
        mean_probs = probs_stack.mean(dim=0)               # (B, C)

        eps = 1e-8
        pred_entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)  # (B,)
        per_sample_h = -(probs_stack * (probs_stack + eps).log()).sum(dim=-1)  # (S, B)
        epistemic = pred_entropy - per_sample_h.mean(dim=0)                    # (B,)

        return mean_probs, pred_entropy, epistemic

    def predict_class(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predicted class and predictive entropy."""
        mean_probs, entropy, _ = self.predict(x)
        return mean_probs.argmax(dim=-1), entropy
