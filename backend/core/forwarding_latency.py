"""
Forwarding Latency Analysis — Median-Based Rapid Forwarding Detection.

For each account:
  - For each incoming transaction, find the next outgoing transaction.
  - Compute latency = next_out_ts - in_ts.
  - If median(latencies) < 2 hours → flag "rapid_forwarding".

Pattern: "rapid_forwarding", Score: +20

Time Complexity: O(V × T) where T = avg transactions per account
Memory: O(V)
"""

from typing import Any, Dict, Set, Tuple

import pandas as pd

from app.config import FORWARDING_LATENCY_MEDIAN_HOURS


def detect_rapid_forwarding(
    df: pd.DataFrame,
) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
    """
    Detect accounts that rapidly forward received funds.

    For each incoming transaction to an account, find the next outgoing
    transaction and measure the latency. If median latency < threshold,
    the account is flagged.

    Returns:
        (flagged_accounts_set, details_dict)
        details_dict: {account_id: {"avg_latency_hours", "median_latency_hours"}}

    Time Complexity: O(V × T)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    flagged: Set[str] = set()
    details: Dict[str, Dict[str, Any]] = {}

    incoming = (
        df[["receiver_id", "timestamp"]]
        .rename(columns={"receiver_id": "account_id", "timestamp": "in_ts"})
        .sort_values(["in_ts", "account_id"])
    )
    outgoing = (
        df[["sender_id", "timestamp"]]
        .rename(columns={"sender_id": "account_id", "timestamp": "out_ts"})
        .sort_values(["out_ts", "account_id"])
    )
    if incoming.empty or outgoing.empty:
        return flagged, details

    # Find the next outgoing txn for each incoming txn, per account.
    merged = pd.merge_asof(
        incoming,
        outgoing,
        left_on="in_ts",
        right_on="out_ts",
        by="account_id",
        direction="forward",
    ).dropna(subset=["out_ts"])
    if merged.empty:
        return flagged, details

    merged["latency_h"] = (merged["out_ts"] - merged["in_ts"]).dt.total_seconds() / 3600.0
    merged = merged[merged["latency_h"] >= 0]
    if merged.empty:
        return flagged, details

    stats = merged.groupby("account_id")["latency_h"].agg(["count", "mean", "median"])
    stats = stats[stats["count"] >= 2]
    stats = stats[stats["median"] < FORWARDING_LATENCY_MEDIAN_HOURS]

    for account, row in stats.iterrows():
        acc = str(account)
        flagged.add(acc)
        details[acc] = {
            "avg_latency_hours": round(float(row["mean"]), 2),
            "median_latency_hours": round(float(row["median"]), 2),
        }

    return flagged, details


