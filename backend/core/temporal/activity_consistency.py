"""
Activity Consistency Variance Module.

Computes daily transaction count variance for each account.
Flags accounts with high variance (sharp spike behavior) combined
with long periods of inactivity.

Pattern: "irregular_activity_spike", Score: +20

Time Complexity: O(V × T)
Memory: O(V × D) where D = distinct active days
"""

import datetime
from typing import Set

import numpy as np
import pandas as pd


def detect_irregular_activity(df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts with highly irregular daily activity patterns.

    Flags accounts where:
      - variance of daily txn counts > 3× mean daily count
      - at least 50% of days in span have zero activity (inactivity periods)
      - minimum 10 total transactions

    Time Complexity: O(V × T)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    flagged: Set[str] = set()

    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())

    for account in all_accounts:
        acct_txns = df[
            (df["sender_id"] == account) | (df["receiver_id"] == account)
        ]

        if len(acct_txns) < 10:
            continue

        # Get daily counts
        dates = acct_txns["timestamp"].dt.date
        min_date = dates.min()
        max_date = dates.max()
        span_days = (max_date - min_date).days + 1

        if span_days < 7:
            continue

        # Count transactions per day (including zeros)
        daily_counts = dates.value_counts()
        active_days = len(daily_counts)
        inactive_days = span_days - active_days

        # Build full daily array including zeros
        full_counts = []
        for d in range(span_days):
            day = min_date + datetime.timedelta(days=d)
            full_counts.append(daily_counts.get(day, 0))

        mean_daily = float(np.mean(full_counts))
        variance = float(np.var(full_counts))

        if mean_daily == 0:
            continue

        # High variance + long inactivity periods
        inactivity_ratio = inactive_days / span_days
        if variance > 3 * mean_daily and inactivity_ratio > 0.5:
            flagged.add(str(account))

    return flagged


