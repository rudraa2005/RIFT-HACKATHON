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

    Logic:
      1. Find the longest gap of inactivity for each account.
      2. If any gap ≥ DORMANCY_INACTIVE_DAYS days exists:
         Check if there are ≥ DORMANCY_MIN_BURST_TXNS transactions
         within DORMANCY_BURST_HOURS hours immediately after the gap.
      3. Flag if both conditions met.

    Returns:
        Set of flagged account IDs

    Time Complexity: O(V × T)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

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

        for i in range(n - 1):
            gap = pd.Timestamp(timestamps[i + 1]) - pd.Timestamp(timestamps[i])

            if gap >= inactive_threshold:
                # Found a dormancy gap — check burst after it
                reactivation_ts = pd.Timestamp(timestamps[i + 1])
                burst_end = reactivation_ts + burst_window

                burst_txns = acct_txns[
                    (acct_txns["timestamp"] >= reactivation_ts)
                    & (acct_txns["timestamp"] <= burst_end)
                ]

                if len(burst_txns) >= DORMANCY_MIN_BURST_TXNS:
                    flagged.add(str(account))
                    break  # One dormancy burst is enough

    return flagged
