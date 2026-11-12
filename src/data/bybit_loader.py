"""Bybit BTCUSDT L2 order book data loader for domain shift evaluation."""

import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List


# ── Order book reconstruction ─────────────────────────────────────────────────

class _OrderBook:
    """Maintains live LOB state from Bybit snapshot+delta events."""

    def __init__(self):
        self.bids: dict = {}  # price → size
        self.asks: dict = {}  # price → size

    def apply(self, bids_raw: list, asks_raw: list, is_snapshot: bool = False) -> None:
        if is_snapshot:
            self.bids.clear()
            self.asks.clear()
        for price_str, size_str in bids_raw:
            p, s = float(price_str), float(size_str)
            if s == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = s
        for price_str, size_str in asks_raw:
            p, s = float(price_str), float(size_str)
            if s == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = s

    def top_n(self, n: int = 10) -> Optional[List[float]]:
        """Return [ask_p1, ask_v1, bid_p1, bid_v1, ..., ask_pN, ask_vN, bid_pN, bid_vN].

        Matches FI-2010 feature order (ask_price, ask_vol, bid_price, bid_vol per level).
        Returns None if fewer than n levels available on either side.
        """
        if len(self.bids) < n or len(self.asks) < n:
            return None
        top_bids = sorted(self.bids.items(), reverse=True)[:n]   # best bid first
        top_asks = sorted(self.asks.items())[:n]                  # best ask first
        row: List[float] = []
        for i in range(n):
            ap, av = top_asks[i]
            bp, bv = top_bids[i]
            row.extend([ap, av, bp, bv])
        return row


def _parse_levels(raw) -> list:
    """Parse bids/asks field which is a JSON-string list of [price, size] pairs."""
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


def _reconstruct_1s_snapshots(df: pd.DataFrame, n_levels: int = 10) -> pd.DataFrame:
    """Reconstruct 1-second LOB snapshots from snapshot+delta event stream.

    Args:
        df: Raw parquet DataFrame with columns [ts, type, bids, asks].
        n_levels: Number of price levels to extract (default 10).

    Returns:
        DataFrame indexed by second-level timestamp, 4*n_levels columns.
    """
    book = _OrderBook()
    records = []
    current_second = None
    last_valid_row: Optional[List[float]] = None

    # ts is in milliseconds
    for _, row in df.iterrows():
        ts_s = int(row['ts']) // 1000          # floor to second
        is_snap = row['type'] == 'snapshot'
        bids_raw = _parse_levels(row['bids'])
        asks_raw = _parse_levels(row['asks'])
        book.apply(bids_raw, asks_raw, is_snapshot=is_snap)

        if current_second is None:
            current_second = ts_s

        # When we cross a second boundary, emit last known book state for each elapsed second
        while ts_s > current_second:
            if last_valid_row is not None:
                records.append((current_second, last_valid_row))
            current_second += 1

        snapshot = book.top_n(n_levels)
        if snapshot is not None:
            last_valid_row = snapshot

    # Emit final second
    if last_valid_row is not None and current_second is not None:
        records.append((current_second, last_valid_row))

    if not records:
        return pd.DataFrame()

    n = n_levels
    cols = []
    for i in range(1, n + 1):
        cols += [f'ask_p{i}', f'ask_v{i}', f'bid_p{i}', f'bid_v{i}']

    timestamps, rows = zip(*records)
    result = pd.DataFrame(rows, index=pd.to_datetime(timestamps, unit='s', utc=True), columns=cols)
    result.index.name = 'timestamp'
    return result


# ── Label generation ──────────────────────────────────────────────────────────

def _compute_labels(df: pd.DataFrame, horizon: int = 10, alpha: float = 2e-4) -> pd.Series:
    """Generate 3-class mid-price movement labels following the FI-2010 labelling rule.

    Label at time t: based on smoothed future mid-price over next k seconds.
      up (2)        if (m_future - m_t) / m_t >  alpha
      down (0)      if (m_future - m_t) / m_t < -alpha
      stationary (1) otherwise

    Encoding {0=down, 1=stationary, 2=up} (0-indexed). This is the REVERSE of FI-2010
    (0=up, 2=down): map with y -> 2 - y before scoring FI-2010-trained models.
    """
    mid = (df['ask_p1'] + df['bid_p1']) / 2.0
    # Smoothed future mid: mean of next k seconds
    future_mid = mid[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-horizon)
    rel_change = (future_mid - mid) / mid
    labels = np.where(rel_change > alpha, 2, np.where(rel_change < -alpha, 0, 1))
    return pd.Series(labels, index=df.index, name='label')


# ── Dataset ───────────────────────────────────────────────────────────────────

class BybitDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 100):
        self.T = sequence_length
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.features) - self.T

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[idx : idx + self.T])
        y = torch.tensor(self.labels[idx + self.T - 1], dtype=torch.long)
        return x, y


# ── Main loader class ─────────────────────────────────────────────────────────

class BybitLoader:
    """Load and preprocess Bybit BTCUSDT L2 data for domain shift experiments.

    Data source layout:
        data_root/date=YYYY-MM-DD/orderbook.parquet

    Protocol: reconstruct LOB from snapshot+delta stream → 1s snapshots →
              extract 10-level features → z-score normalize (matching FI-2010 scale).
    Split: last calib_fraction of the time series = calibration holdout.
    """

    def __init__(
        self,
        data_root: str,
        n_levels: int = 10,
        calib_fraction: float = 0.10,
    ):
        self.data_root = Path(data_root)
        self.n_levels = n_levels
        self.calib_fraction = calib_fraction
        self._snapshots: Optional[pd.DataFrame] = None
        self._labels: Optional[pd.Series] = None

    def build_lob_snapshots(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Parse raw L2 parquets and build 1-second LOB snapshot DataFrame.

        Args:
            start_date: ISO date string e.g. '2025-05-01'.
            end_date:   ISO date string e.g. '2025-05-07' (inclusive).

        Returns:
            DataFrame with 40 columns (10 levels × 4 fields), 1-second index.
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        day_dfs = []
        for date in dates:
            day_str = date.strftime('%Y-%m-%d')
            parquet = self.data_root / f'date={day_str}' / 'orderbook.parquet'
            if not parquet.exists():
                print(f'  [skip] {parquet} not found')
                continue
            print(f'  Loading {day_str} ...', end=' ', flush=True)
            raw = pd.read_parquet(parquet, columns=['ts', 'type', 'bids', 'asks'])
            snap = _reconstruct_1s_snapshots(raw, self.n_levels)
            print(f'{len(snap):,} rows')
            if not snap.empty:
                day_dfs.append(snap)

        if not day_dfs:
            raise RuntimeError(f'No parquet files found in {self.data_root} for {start_date}–{end_date}')

        snapshots = pd.concat(day_dfs).sort_index()
        # Drop duplicate timestamps (day boundary overlaps)
        snapshots = snapshots[~snapshots.index.duplicated(keep='last')]
        self._snapshots = snapshots
        print(f'Total 1s snapshots: {len(snapshots):,}  ({start_date} → {end_date})')
        return snapshots

    def normalize(
        self,
        df: pd.DataFrame,
        stats: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """Z-score normalize per-feature.

        Args:
            df:    Raw LOB snapshot DataFrame (output of build_lob_snapshots).
            stats: Optional {'mean': Series, 'std': Series}. If None, computed from df.

        Returns:
            (normalized_df, stats_dict)
        """
        if stats is None:
            mean = df.mean()
            std  = df.std().replace(0, 1.0)   # avoid division by zero for flat cols
            stats = {'mean': mean, 'std': std}
        normalized = (df - stats['mean']) / stats['std']
        return normalized, stats

    def get_features_and_labels(
        self,
        snapshots: pd.DataFrame,
        horizon: int = 10,
        alpha: float = 2e-4,
        norm_stats: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Return normalized features array, label array, and norm stats.

        Returns:
            features:   (M, 40) float32
            labels:     (M,)    int64  — {0=down, 1=stationary, 2=up}
            norm_stats: dict for re-use on held-out data
        """
        labels_series = _compute_labels(snapshots, horizon=horizon, alpha=alpha)
        # Drop rows where label is NaN (last `horizon` rows)
        valid = labels_series.notna()
        snapshots = snapshots[valid]
        labels_series = labels_series[valid]

        normalized, norm_stats = self.normalize(snapshots, stats=norm_stats)
        features = normalized.values.astype(np.float32)
        labels   = labels_series.values.astype(np.int64)
        return features, labels, norm_stats

    def get_calib_dataset(
        self,
        snapshots: pd.DataFrame,
        horizon: int = 10,
        sequence_length: int = 100,
        batch_size: int = 64,
        norm_stats: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader, dict]:
        """Split into eval (90%) and calibration holdout (10%) DataLoaders.

        Returns:
            (eval_loader, calib_loader, norm_stats)
        """
        features, labels, norm_stats = self.get_features_and_labels(
            snapshots, horizon=horizon, norm_stats=norm_stats
        )
        split = int(len(features) * (1 - self.calib_fraction))
        eval_ds  = BybitDataset(features[:split], labels[:split],  sequence_length)
        calib_ds = BybitDataset(features[split:], labels[split:],  sequence_length)
        kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
        return (
            DataLoader(eval_ds,  shuffle=False, **kw),
            DataLoader(calib_ds, shuffle=False, **kw),
            norm_stats,
        )
