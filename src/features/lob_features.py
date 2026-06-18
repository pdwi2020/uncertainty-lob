"""LOB feature engineering: OFI, queue imbalance, depth-weighted mid, spread.

FI-2010 feature layout (40 features, 10 levels):
    col 4*(i-1)+0 : ask_price_level_i
    col 4*(i-1)+1 : ask_vol_level_i
    col 4*(i-1)+2 : bid_price_level_i
    col 4*(i-1)+3 : bid_vol_level_i
"""

import numpy as np

_N_LEVELS = 10


def _col(level: int, field: str) -> int:
    fields = ["ask_price", "ask_vol", "bid_price", "bid_vol"]
    return (level - 1) * 4 + fields.index(field)


def mid_price(features: np.ndarray) -> np.ndarray:
    return (features[..., _col(1, "bid_price")] + features[..., _col(1, "ask_price")]) / 2


def spread(features: np.ndarray) -> np.ndarray:
    return features[..., _col(1, "ask_price")] - features[..., _col(1, "bid_price")]


def relative_spread(features: np.ndarray) -> np.ndarray:
    mid = mid_price(features)
    return spread(features) / (np.abs(mid) + 1e-8)


def queue_imbalance(features: np.ndarray) -> np.ndarray:
    """Best-level signed volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).

    Clipped to [-1, 1] — raw ratio explodes on near-zero Z-scored volumes.
    """
    bv = features[..., _col(1, "bid_vol")]
    av = features[..., _col(1, "ask_vol")]
    raw = (bv - av) / (bv + av + 1e-8)
    return np.clip(raw, -1.0, 1.0)


def depth_weighted_mid(features: np.ndarray) -> np.ndarray:
    """Volume-weighted average price across all 10 bid/ask levels."""
    weighted_sum = np.zeros(features.shape[:-1])
    total_vol = np.zeros(features.shape[:-1])
    for i in range(1, _N_LEVELS + 1):
        bp = features[..., _col(i, "bid_price")]
        ap = features[..., _col(i, "ask_price")]
        bv = features[..., _col(i, "bid_vol")]
        av = features[..., _col(i, "ask_vol")]
        weighted_sum += bp * bv + ap * av
        total_vol += bv + av
    return weighted_sum / (total_vol + 1e-8)


def order_flow_imbalance(features: np.ndarray) -> np.ndarray:
    """OFI: first-differenced signed net volume imbalance across all levels.

    For a (T, 40) array returns (T-1,) OFI series.
    For a single (40,) snapshot returns scalar signed imbalance.
    """
    if features.ndim == 1:
        ofi = 0.0
        for i in range(1, _N_LEVELS + 1):
            ofi += features[_col(i, "bid_vol")] - features[_col(i, "ask_vol")]
        return np.float32(ofi)

    bid_vols = np.stack([features[:, _col(i, "bid_vol")] for i in range(1, _N_LEVELS + 1)], axis=1)
    ask_vols = np.stack([features[:, _col(i, "ask_vol")] for i in range(1, _N_LEVELS + 1)], axis=1)
    delta_bid = np.diff(bid_vols, axis=0).sum(axis=1)
    delta_ask = np.diff(ask_vols, axis=0).sum(axis=1)
    return delta_bid - delta_ask


def realized_volatility(mid: np.ndarray, window: int = 3600) -> np.ndarray:
    """Rolling realized volatility of log mid-price returns (NaN for first `window` steps)."""
    log_ret = np.diff(np.log(np.abs(mid) + 1e-8))
    rv = np.full(len(mid), np.nan)
    for t in range(window, len(mid)):
        rv[t] = log_ret[t - window:t].std()
    return rv


def extract_all(features: np.ndarray) -> dict:
    """Extract all engineered features from an (N, 40) array. Returns dict of (N,) arrays."""
    return {
        "mid_price": mid_price(features),
        "spread": spread(features),
        "relative_spread": relative_spread(features),
        "queue_imbalance": queue_imbalance(features),
        "depth_weighted_mid": depth_weighted_mid(features),
    }
