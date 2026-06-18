"""Market-making backtest with adverse-selection penalty (Avellaneda-Stoikov framework)."""

import numpy as np
import pandas as pd
from typing import Optional, Dict


class MarketMakingSimulator:
    """Simulate market-making P&L using directional signal and OFI for adverse selection.

    Quoting strategy:
      - Without signal: symmetric quotes ± half_spread around mid.
      - With bearish signal (0): skew bid down, ask down (reluctant to buy).
      - With bullish signal (2): skew bid up, ask up (reluctant to sell).
      - Adverse selection: flag fills where OFI predicts adverse mid move.

    Fill model: simplified — a bid fill occurs when the next mid drops below bid;
    an ask fill occurs when the next mid rises above ask.
    """

    def __init__(
        self,
        spread_bps: float = 2.0,
        skew_bps: float = 0.5,
        inventory_limit: int = 10,
        ofi_threshold: float = 0.5,
        adverse_lookback: int = 10,
        price_std: float = None,   # if set, spread/skew are scaled by price_std
    ):
        # When data is Z-score normalised, prices have std~1 and absolute BPS
        # (1e-4 scale) have no effect. Scale spread/skew by price_std when provided.
        scale = price_std if price_std is not None else 1.0 / 10_000
        self.half_spread = (spread_bps / 2) * scale
        self.skew = skew_bps * scale
        self.inventory_limit = inventory_limit
        self.ofi_threshold = ofi_threshold
        self.adverse_lookback = adverse_lookback

    def simulate(
        self,
        mid_prices: np.ndarray,
        signals: Optional[np.ndarray] = None,
        ofi: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Run market-making simulation.

        Args:
            mid_prices: (T,) mid-price time series.
            signals: (T,) directional predictions: 0=down, 1=flat, 2=up.
            ofi: (T,) order flow imbalance for adverse selection detection.

        Returns:
            DataFrame with per-timestep results.
        """
        T = len(mid_prices)
        if signals is None:
            signals = np.ones(T, dtype=int)
        if ofi is None:
            ofi = np.zeros(T)

        rows = []
        inventory = 0
        cum_pnl = 0.0

        for t in range(T - 1):
            mid = mid_prices[t]
            sig = int(signals[t])

            # Direction skew based on signal
            if sig == 0:      # bearish: skew quotes down
                skew_adj = -self.skew
            elif sig == 2:    # bullish: skew quotes up
                skew_adj = +self.skew
            else:
                skew_adj = 0.0

            bid = mid - self.half_spread + skew_adj
            ask = mid + self.half_spread + skew_adj

            # Inventory control: widen/suppress quotes near limits
            if inventory >= self.inventory_limit:
                bid = -np.inf  # don't buy more
            elif inventory <= -self.inventory_limit:
                ask = np.inf   # don't sell more

            next_mid = mid_prices[t + 1]
            fill_side = None
            fill_pnl = 0.0
            adverse = False

            # Fill check: simplified touch model
            if next_mid <= bid:
                # Bid hit → we bought at bid; mark-to-market with next mid
                fill_pnl = next_mid - bid
                inventory += 1
                fill_side = "buy"
                # Adverse: OFI strongly negative (sellers dominating) while we just bought
                adverse = bool(ofi[t] < -self.ofi_threshold)

            elif next_mid >= ask:
                # Ask lifted → we sold at ask
                fill_pnl = ask - next_mid
                inventory -= 1
                fill_side = "sell"
                adverse = bool(ofi[t] > self.ofi_threshold)

            # Running inventory PnL: mark existing position at mid change
            position_pnl = inventory * (next_mid - mid)
            step_pnl = fill_pnl + position_pnl
            cum_pnl += step_pnl

            rows.append({
                "t": t,
                "mid": mid,
                "bid": bid,
                "ask": ask,
                "signal": sig,
                "fill_side": fill_side,
                "fill_pnl": fill_pnl,
                "position_pnl": position_pnl,
                "step_pnl": step_pnl,
                "cum_pnl": cum_pnl,
                "inventory": inventory,
                "adverse_flag": adverse,
            })

        return pd.DataFrame(rows)

    def compute_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Compute Sharpe, fill rate, adverse selection rate."""
        T = len(results)
        fills = results["fill_side"].notna()
        n_fills = fills.sum()
        n_adverse = results["adverse_flag"].sum()

        # Daily P&L (assume ~23400 steps per trading day for equity LOB at 1s)
        steps_per_day = 23400
        daily_pnl = results["step_pnl"].values
        n_days = max(1, T // steps_per_day)
        daily_groups = np.array_split(daily_pnl, n_days)
        daily_totals = np.array([g.sum() for g in daily_groups])

        sharpe = (daily_totals.mean() / (daily_totals.std() + 1e-10)) * np.sqrt(252)

        return {
            "sharpe": float(sharpe),
            "total_pnl": float(results["cum_pnl"].iloc[-1]),
            "fill_rate": float(n_fills / T),
            "adverse_rate": float(n_adverse / max(1, n_fills)),
            "n_fills": int(n_fills),
            "n_adverse_fills": int(n_adverse),
        }
