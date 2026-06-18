"""Temperature scaling for post-hoc calibration of neural network classifiers."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class TemperatureScaling(nn.Module):
    """Post-hoc calibration via a single scalar temperature T.

    Calibrated probabilities: p = softmax(logits / T)
    T is fitted by minimizing NLL on validation set (LBFGS).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Fit temperature on validation logits. Returns fitted temperature."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self.temperature.item()

    def compute_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 10,
        apply_temperature: bool = True,
    ) -> float:
        """Expected Calibration Error (lower is better).

        Returns:
            ECE value in [0, 1].
        """
        with torch.no_grad():
            scaled = self.forward(logits) if apply_temperature else logits
            probs = torch.softmax(scaled, dim=-1)
            confidences, predictions = probs.max(dim=-1)
            correct = predictions.eq(labels).float()

        confidences = confidences.cpu().numpy()
        correct = correct.cpu().numpy()
        N = len(confidences)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / N) * abs(bin_acc - bin_conf)
        return float(ece)
