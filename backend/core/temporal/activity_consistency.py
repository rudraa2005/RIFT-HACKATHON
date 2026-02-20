"""
Activity Consistency Variance Module.

Computes daily transaction count variance for each account.
Flags accounts with high variance (sharp spike behavior) combined
with long periods of inactivity.

Pattern: "irregular_activity_spike", Score: +20

Time Complexity: O(V × T)
Memory: O(V × D) where D = distinct active days
"""

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
    # Build account-time event table once (sender + receiver participation).
    events = pd.concat(
        [
            df[["sender_id", "timestamp"]].rename(columns={"sender_id": "account_id"}),
            df[["receiver_id", "timestamp"]].rename(columns={"receiver_id": "account_id"}),
        ],
        ignore_index=True,
    )
    if events.empty:
        return set()

    events["date"] = events["timestamp"].dt.floor("D")
    per_day = events.groupby(["account_id", "date"]).size().rename("count").reset_index()

    acct_stats = per_day.groupby("account_id").agg(
        active_days=("date", "nunique"),
        first_day=("date", "min"),
        last_day=("date", "max"),
        total_txns=("count", "sum"),
        sum_sq=("count", lambda s: float(np.square(s.to_numpy(dtype=float)).sum())),
    )
    if acct_stats.empty:
        return set()

    acct_stats["span_days"] = (acct_stats["last_day"] - acct_stats["first_day"]).dt.days + 1
    acct_stats = acct_stats[(acct_stats["total_txns"] >= 10) & (acct_stats["span_days"] >= 7)]
    if acct_stats.empty:
        return set()

    acct_stats["inactive_days"] = acct_stats["span_days"] - acct_stats["active_days"]
    acct_stats["mean_daily"] = acct_stats["total_txns"] / acct_stats["span_days"]
    acct_stats["variance"] = (acct_stats["sum_sq"] / acct_stats["span_days"]) - np.square(acct_stats["mean_daily"])
    acct_stats["inactivity_ratio"] = acct_stats["inactive_days"] / acct_stats["span_days"]

    flagged_idx = acct_stats[
        (acct_stats["mean_daily"] > 0)
        & (acct_stats["variance"] > 3 * acct_stats["mean_daily"])
        & (acct_stats["inactivity_ratio"] > 0.5)
    ].index
    return set(flagged_idx.astype(str))


