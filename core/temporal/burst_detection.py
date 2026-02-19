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

from typing import Set

import pandas as pd
from datetime import timedelta

from app.config import (
    SPIKE_MULTIPLIER,
    SPIKE_MIN_TRANSACTIONS,
    VELOCITY_WINDOW_HOURS,
)


def detect_activity_spikes(df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts with sudden activity spikes.

    Returns:
        Set of flagged account IDs

    Time Complexity: O(V × T)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    flagged: Set[str] = set()
    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
    window_delta = timedelta(hours=VELOCITY_WINDOW_HOURS)

    for account in all_accounts:
        acct_txns = df[
            (df["sender_id"] == account) | (df["receiver_id"] == account)
        ].sort_values("timestamp")

        total_txns = len(acct_txns)
        if total_txns < SPIKE_MIN_TRANSACTIONS:
            continue

        # Baseline rate
        total_span_days = (
            acct_txns["timestamp"].max() - acct_txns["timestamp"].min()
        ).total_seconds() / 86400
        if total_span_days < 1:
            total_span_days = 1

        baseline_rate = total_txns / total_span_days

        # Skip consistently active accounts (not spiky by nature)
        if baseline_rate > 10:
            continue

        # Ensure a minimum baseline to avoid false positives on very sparse accounts
        effective_baseline = max(baseline_rate, 0.1)

        # Find max 72h window rate
        max_window_rate = 0
        for i in range(len(acct_txns)):
            window_end = acct_txns["timestamp"].iloc[i]
            window_start = window_end - window_delta
            window_count = len(
                acct_txns[
                    (acct_txns["timestamp"] >= window_start)
                    & (acct_txns["timestamp"] <= window_end)
                ]
            )
            window_rate = window_count / 3  # normalize to daily (72h = 3 days)
            max_window_rate = max(max_window_rate, window_rate)

        if max_window_rate >= SPIKE_MULTIPLIER * effective_baseline:
            flagged.add(account)

    return flagged
