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
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    inflows = df[["receiver_id", "timestamp", "amount"]].rename(
        columns={"receiver_id": "account_id"}
    )
    inflows["net"] = inflows["amount"].astype(float)

    outflows = df[["sender_id", "timestamp", "amount"]].rename(
        columns={"sender_id": "account_id"}
    )
    outflows["net"] = -outflows["amount"].astype(float)

    events = pd.concat([inflows[["account_id", "timestamp", "net"]], outflows[["account_id", "timestamp", "net"]]], ignore_index=True)
    if events.empty:
        return set()

    events = events.sort_values(["account_id", "timestamp"], kind="mergesort")
    events["cum"] = events.groupby("account_id", observed=True)["net"].cumsum()

    counts = events.groupby("account_id", observed=True).size()
    eligible = counts[counts >= 4].index
    if len(eligible) == 0:
        return set()

    cum_std = events.groupby("account_id", observed=True)["cum"].std(ddof=0).reindex(eligible, fill_value=0.0)
    total_flow = events["net"].abs().groupby(events["account_id"], observed=True).sum().reindex(eligible, fill_value=0.0)
    valid = total_flow > 0
    if not valid.any():
        return set()

    ratio = (cum_std[valid] / total_flow[valid]).fillna(np.inf)
    flagged_idx = ratio[ratio < 0.1].index
    return {str(account) for account in flagged_idx}


