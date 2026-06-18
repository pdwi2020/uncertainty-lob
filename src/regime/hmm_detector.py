"""HMM-based market regime detection for LOB data."""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import f1_score, accuracy_score
from typing import Tuple, Optional


class HMMRegimeDetector:
    """Gaussian HMM with 2 components for low-vol / high-vol regime labeling.

    Features: realized volatility (1h rolling), spread, order flow imbalance.
    Fitted on FI-2010 train; applied to test and Bybit.
    Convention: regime 0 = low-vol, regime 1 = high-vol.
    """

    def __init__(self, n_components: int = 2, n_iter: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        self.high_vol_component: Optional[int] = None
        self.regime_labels: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> "HMMRegimeDetector":
        """Fit HMM on training features.

        Args:
            features: (T, 3) with columns [realized_vol, spread, ofi].
        """
        self.model.fit(features)
        # Identify which component has higher mean realized vol → that's regime 1
        means_vol = self.model.means_[:, 0]
        self.high_vol_component = int(np.argmax(means_vol))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Infer regime labels (0=low-vol, 1=high-vol).

        Returns:
            Integer array (T,).
        """
        raw = self.model.predict(features)
        # Remap: high-vol component → 1, other → 0
        remapped = np.where(raw == self.high_vol_component, 1, 0)
        self.regime_labels = remapped
        return remapped

    def regime_conditional_performance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        regimes: np.ndarray,
    ) -> pd.DataFrame:
        """Compute F1 and accuracy per regime.

        Args:
            predictions: Model predictions (N,).
            labels: True class labels (N,).
            regimes: Regime label per sample (N,).

        Returns:
            DataFrame with columns [regime, accuracy, f1_macro, f1_weighted, n_samples].
        """
        rows = []
        for regime_id in sorted(np.unique(regimes)):
            mask = regimes == regime_id
            if mask.sum() == 0:
                continue
            y_true = labels[mask]
            y_pred = predictions[mask]
            rows.append({
                "regime": int(regime_id),
                "regime_name": "high-vol" if regime_id == 1 else "low-vol",
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "n_samples": int(mask.sum()),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def compute_realized_vol(mid_prices: np.ndarray, window: int = 3600) -> np.ndarray:
        """Compute rolling realized volatility (log-return std over window steps)."""
        log_ret = np.diff(np.log(mid_prices + 1e-10), prepend=np.nan)
        vol = pd.Series(log_ret).rolling(window, min_periods=1).std().to_numpy()
        return vol

    @staticmethod
    def compute_ofi(bid_vol: np.ndarray, ask_vol: np.ndarray) -> np.ndarray:
        """Order flow imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        denom = bid_vol + ask_vol
        return np.where(denom > 0, (bid_vol - ask_vol) / denom, 0.0)
