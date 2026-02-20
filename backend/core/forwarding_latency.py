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

import numpy as np
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

    all_accounts = set(df["sender_id"].unique()) & set(df["receiver_id"].unique())

    for account in all_accounts:
        incoming = df[df["receiver_id"] == account].sort_values("timestamp")
        outgoing = df[df["sender_id"] == account].sort_values("timestamp")

        if incoming.empty or outgoing.empty:
            continue

        out_times = outgoing["timestamp"].values

        latencies = []
        for _, in_row in incoming.iterrows():
            in_ts = in_row["timestamp"]
            # Find next outgoing after this incoming
            future_outs = out_times[out_times >= np.datetime64(in_ts)]
            if len(future_outs) > 0:
                next_out_ts = pd.Timestamp(future_outs[0])
                latency_h = (next_out_ts - in_ts).total_seconds() / 3600
                latencies.append(max(0, latency_h))

        if len(latencies) < 2:
            continue

        avg_latency = float(np.mean(latencies))
        median_latency = float(np.median(latencies))

        if median_latency < FORWARDING_LATENCY_MEDIAN_HOURS:
            flagged.add(str(account))
            details[str(account)] = {
                "avg_latency_hours": round(avg_latency, 2),
                "median_latency_hours": round(median_latency, 2),
            }

    return flagged, details


