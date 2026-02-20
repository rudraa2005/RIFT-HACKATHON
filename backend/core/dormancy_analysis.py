"""
Dormant Activation Detection Module.

Detects accounts that were dormant for ≥30 days then suddenly activated
with ≥10 transactions within 48 hours.

Pattern: "dormant_activation_spike", Score: +20

Time Complexity: O(V × T) where T = avg transactions per account
Memory: O(V)
"""

from typing import Set

import numpy as np
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

    inactive_threshold = pd.Timedelta(days=DORMANCY_INACTIVE_DAYS)
    burst_window = pd.Timedelta(hours=DORMANCY_BURST_HOURS)

    activity = pd.concat(
        [
            df[["sender_id", "timestamp"]].rename(columns={"sender_id": "account_id"}),
            df[["receiver_id", "timestamp"]].rename(columns={"receiver_id": "account_id"}),
        ],
        ignore_index=True,
    ).sort_values(["account_id", "timestamp"])

    flagged: Set[str] = set()
    for account, group in activity.groupby("account_id", sort=False):
        ts = group["timestamp"].to_numpy()
        if ts.size < DORMANCY_MIN_BURST_TXNS:
            continue

        candidate_starts = []
        if (pd.Timestamp(ts[0]) - dataset_start) >= inactive_threshold:
            candidate_starts.append(0)

        if ts.size >= 2:
            gaps = pd.Series(ts).diff().to_numpy()
            gap_idx = np.where(gaps >= inactive_threshold.to_timedelta64())[0]
            candidate_starts.extend(gap_idx.tolist())

        if not candidate_starts:
            continue

        for idx in candidate_starts:
            start_ts = pd.Timestamp(ts[idx])
            end_ts = start_ts + burst_window
            right = np.searchsorted(ts, np.datetime64(end_ts), side="right")
            if (right - idx) >= DORMANCY_MIN_BURST_TXNS:
                flagged.add(str(account))
                break

    return flagged


