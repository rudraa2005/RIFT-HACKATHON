"""
Balance Oscillation Detection.

Cumulative net flow curve over time.
If std(cumulative_balance) is small relative to total flow → oscillation near zero.
Pattern: "balance_oscillation_pass_through", Score: +20

Time Complexity: O(V × T)
Memory: O(V)
"""

from typing import Set

import numpy as np
import pandas as pd


def detect_balance_oscillation(df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts whose cumulative balance oscillates near zero.

    Computes cumulative net flow (inflow - outflow) over time.
    If std(cumulative) / total_flow < 0.1 → oscillation pattern.

    Time Complexity: O(V × T)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    flagged: Set[str] = set()

    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())

    for account in all_accounts:
        # Build time-ordered net flow events
        inflows = df[df["receiver_id"] == account][["timestamp", "amount"]].copy()
        inflows["net"] = inflows["amount"]
        outflows = df[df["sender_id"] == account][["timestamp", "amount"]].copy()
        outflows["net"] = -outflows["amount"]

        events = pd.concat([inflows[["timestamp", "net"]], outflows[["timestamp", "net"]]])
        if len(events) < 4:
            continue

        events = events.sort_values("timestamp")
        cumulative = events["net"].cumsum().values

        total_flow = df[
            (df["sender_id"] == account) | (df["receiver_id"] == account)
        ]["amount"].sum()

        if total_flow == 0:
            continue

        std_cum = float(np.std(cumulative))
        ratio = std_cum / total_flow

        if ratio < 0.1:
            flagged.add(str(account))

    return flagged
