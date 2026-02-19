"""
Sudden Activity Spike Detection Module.

Detects accounts with abnormal short-term transaction surges.

For each account:
  baseline_rate = total_transactions / total_active_days
  window_rate   = transactions_in_72h_window / 3  (normalized to daily)

If window_rate >= 5 × baseline_rate AND total_transactions > threshold → flag

Time Complexity: O(V × T) where T = avg transactions per account
Memory: O(V)
"""

from typing import Set, Tuple, Dict

import pandas as pd
from datetime import timedelta

from app.config import (
    SPIKE_MULTIPLIER,
    SPIKE_MIN_TRANSACTIONS,
    VELOCITY_WINDOW_HOURS,
)


def detect_activity_spikes(
    df: pd.DataFrame,
    min_txns_override: float | None = None,
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Detect accounts with sudden activity spikes using vectorized rolling windows.

    Returns:
        Tuple of (flagged_set, trigger_timestamps)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    flagged: Set[str] = set()
    trigger_times: Dict[str, str] = {}
    
    min_txns = min_txns_override or SPIKE_MIN_TRANSACTIONS
    window_delta_str = f"{VELOCITY_WINDOW_HOURS}h"
    
    # Pre-filter: only accounts with enough transactions
    all_senders = df.groupby("sender_id").size()
    all_receivers = df.groupby("receiver_id").size()
    
    potential_accounts = set(all_senders[all_senders >= min_txns].index) | \
                         set(all_receivers[all_receivers >= min_txns].index)

    for account in potential_accounts:
        acct_txns = df[
            (df["sender_id"] == account) | (df["receiver_id"] == account)
        ][["timestamp"]].sort_values("timestamp")

        total_txns = len(acct_txns)
        if total_txns < min_txns:
            continue

        # Baseline rate
        t_min = acct_txns["timestamp"].min()
        t_max = acct_txns["timestamp"].max()
        total_span_days = (t_max - t_min).total_seconds() / 86400
        total_span_days = max(total_span_days, 1.0)
        baseline_rate = total_txns / total_span_days

        # Skip consistently active accounts (not spiky by nature)
        if baseline_rate > 10:
            continue

        # Ensure a minimum baseline to avoid false positives
        effective_baseline = max(baseline_rate, 0.1)

        # Efficiently find max 72h window count using rolling
        acct_txns = acct_txns.set_index("timestamp")
        acct_txns["count"] = 1
        # Rolling count in the window
        rolling_counts = acct_txns["count"].rolling(window_delta_str).count()
        
        max_window_count = rolling_counts.max()
        peak_timestamp = rolling_counts.idxmax()

        window_rate = max_window_count / 3.0  # normalize to daily (72h = 3 days)

        if window_rate >= SPIKE_MULTIPLIER * effective_baseline:
            account_id = str(account)
            flagged.add(account_id)
            trigger_times[account_id] = str(peak_timestamp)

    return flagged, trigger_times
