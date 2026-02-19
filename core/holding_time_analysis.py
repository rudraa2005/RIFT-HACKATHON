"""
Holding Time Risk Module — Rapid Pass-Through Detection.

Detects mule accounts that receive and immediately forward funds.

For each account:
  holding_time = average(first_outgoing_ts - last_incoming_ts)
  forward_ratio = total_outgoing / total_incoming

If holding_time < 2 hours AND forward_ratio > 0.8 → flag "rapid_pass_through"

Edge cases handled:
  - Accounts with only incoming or only outgoing transfers → skipped
  - Division-by-zero when no incoming amount → skipped

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
    Detect accounts with rapid pass-through behavior.

    Returns:
        (flagged_accounts_set, details_dict)
        details_dict: {account_id: {"avg_holding_hours": float, "forward_ratio": float}}

    Time Complexity: O(V × T)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    flagged: Set[str] = set()
    details: Dict[str, Dict[str, Any]] = {}

    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())

    for account in all_accounts:
        incoming = df[df["receiver_id"] == account].sort_values("timestamp")
        outgoing = df[df["sender_id"] == account].sort_values("timestamp")

        # Skip if only incoming or only outgoing
        if incoming.empty or outgoing.empty:
            continue

        # Compute holding times: for each outgoing, find the closest prior incoming
        holding_times = []
        for _, out_row in outgoing.iterrows():
            prior_incoming = incoming[incoming["timestamp"] <= out_row["timestamp"]]
            if not prior_incoming.empty:
                last_in_ts = prior_incoming["timestamp"].iloc[-1]
                hold_h = (out_row["timestamp"] - last_in_ts).total_seconds() / 3600
                holding_times.append(max(0, hold_h))

        if not holding_times:
            continue

        avg_holding = sum(holding_times) / len(holding_times)

        # Forward ratio
        total_incoming = incoming["amount"].sum()
        if total_incoming == 0:
            continue
        total_outgoing = outgoing["amount"].sum()
        forward_ratio = total_outgoing / total_incoming

        if avg_holding < HOLDING_TIME_THRESHOLD_HOURS and forward_ratio > HOLDING_FORWARD_RATIO:
            flagged.add(account)
            details[account] = {
                "avg_holding_hours": round(avg_holding, 2),
                "forward_ratio": round(forward_ratio, 2),
            }

    return flagged, details
