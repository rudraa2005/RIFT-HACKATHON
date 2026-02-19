"""
Holding Time Risk Module — Forwarding Latency Detection.

Detects mule accounts that receive and immediately forward funds.

For each account:
  holding_time = average(first_outgoing_ts - last_incoming_ts)
  forward_ratio = total_outgoing / total_incoming

If holding_time < 2 hours AND forward_ratio > 0.8 → flag "rapid_pass_through"

Time Complexity: O(V × T) where T = avg transactions per account
Memory: O(V)
"""

from typing import Any, Dict, Set, Tuple

import pandas as pd

from app.config import HOLDING_TIME_THRESHOLD_HOURS, HOLDING_FORWARD_RATIO


def detect_rapid_pass_through(
    df: pd.DataFrame,
) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
    """
    Vectorized detection of rapid pass-through behavior (O(N log N)).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    flagged: Set[str] = set()
    details: Dict[str, Dict[str, Any]] = {}

    # Separate incoming and outgoing, pre-sort for asof merge
    in_df = df[["receiver_id", "timestamp", "amount"]].rename(columns={"receiver_id": "account_id", "amount": "amount_in"})
    out_df = df[["sender_id", "timestamp", "amount"]].rename(columns={"sender_id": "account_id", "amount": "amount_out"})

    # Use merge_asof to find last incoming before each outgoing PER account
    in_df_ext = in_df.rename(columns={"timestamp": "ts_in"})
    
    merged = pd.merge_asof(
        out_df,
        in_df_ext,
        left_on="timestamp",
        right_on="ts_in",
        by="account_id",
        direction="backward"
    )
    # Drop rows where there was no incoming transaction before the outgoing one
    merged = merged.dropna(subset=["ts_in"])
    merged["hold_h"] = (merged["timestamp"] - merged["ts_in"]).dt.total_seconds() / 3600
    
    # Aggregates per account
    summary = merged.groupby("account_id").agg({
        "hold_h": "mean",
        "amount_out": "sum" # Total outgoing that had a prior incoming
    })
    
    # Need total incoming sum for ratio
    total_in = in_df.groupby("account_id")["amount_in"].sum()
    
    for account, row in summary.iterrows():
        avg_hold = row["hold_h"]
        total_outgoing = row["amount_out"]
        total_incoming = total_in.get(account, 0)
        
        if total_incoming > 0:
            ratio = total_outgoing / total_incoming
            if avg_hold < HOLDING_TIME_THRESHOLD_HOURS and ratio > HOLDING_FORWARD_RATIO:
                acc_str = str(account)
                flagged.add(acc_str)
                details[acc_str] = {
                    "avg_holding_hours": round(avg_hold, 2),
                    "forward_ratio": round(ratio, 2),
                }

    return flagged, details
