"""
Dormant Activation Detection Module.

Detects accounts that were dormant for ≥30 days then suddenly activated
with ≥10 transactions within 48 hours.

Pattern: "dormant_activation_spike", Score: +20

Time Complexity: O(V × T) where T = avg transactions per account
Memory: O(V)
"""

from datetime import timedelta
from typing import Set

import pandas as pd

from app.config import (
    DORMANCY_INACTIVE_DAYS,
    DORMANCY_BURST_HOURS,
    DORMANCY_MIN_BURST_TXNS,
)


def detect_dormant_activation(df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts with dormant-then-burst behavior.
    Includes accounts that are silent from the start of the dataset.
    """
    if df.empty:
        return set()
        
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    dataset_start = df["timestamp"].min()

    flagged: Set[str] = set()
    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())

    inactive_threshold = timedelta(days=DORMANCY_INACTIVE_DAYS)
    burst_window = timedelta(hours=DORMANCY_BURST_HOURS)

    for account in all_accounts:
        acct_txns = df[
            (df["sender_id"] == account) | (df["receiver_id"] == account)
        ].sort_values("timestamp")

        if len(acct_txns) < DORMANCY_MIN_BURST_TXNS:
            continue

        timestamps = acct_txns["timestamp"].values
        n = len(timestamps)

        # Check gap between dataset start and first transaction
        first_gap = pd.Timestamp(timestamps[0]) - dataset_start
        if first_gap >= inactive_threshold:
            # Check burst starting from first transaction
            reactivation_ts = pd.Timestamp(timestamps[0])
            burst_end = reactivation_ts + burst_window
            burst_txns = acct_txns[
                (acct_txns["timestamp"] >= reactivation_ts)
                & (acct_txns["timestamp"] <= burst_end)
            ]
            if len(burst_txns) >= DORMANCY_MIN_BURST_TXNS:
                flagged.add(str(account))
                continue

        # Check for gaps between transactions
        for i in range(n - 1):
            gap = pd.Timestamp(timestamps[i + 1]) - pd.Timestamp(timestamps[i])

            if gap >= inactive_threshold:
                reactivation_ts = pd.Timestamp(timestamps[i + 1])
                burst_end = reactivation_ts + burst_window
                burst_txns = acct_txns[
                    (acct_txns["timestamp"] >= reactivation_ts)
                    & (acct_txns["timestamp"] <= burst_end)
                ]

                if len(burst_txns) >= DORMANCY_MIN_BURST_TXNS:
                    flagged.add(str(account))
                    break

    return flagged


