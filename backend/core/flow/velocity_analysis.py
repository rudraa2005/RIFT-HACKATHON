"""
Velocity Analysis — identifies high-velocity accounts.

Velocity = total_amount / time_span_hours
Flags accounts whose velocity exceeds HIGH_VELOCITY_THRESHOLD × global average.

Time Complexity: O(V)
Memory: O(V)
"""

from typing import Dict, Set

import numpy as np
import pandas as pd

from app.config import HIGH_VELOCITY_THRESHOLD


def compute_high_velocity_accounts(
    df: pd.DataFrame,
    multiplier_override: float | None = None,
) -> tuple[Set[str], Dict[str, str]]:
    """
    Identify accounts with normalized velocity > threshold.

    Returns:
        Tuple of (set_of_ids, trigger_timestamps)

    Time Complexity: O(V)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    multiplier = multiplier_override or HIGH_VELOCITY_THRESHOLD
    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
    velocities: Dict[str, float] = {}
    last_tx_ts: Dict[str, str] = {}

    for account in all_accounts:
        acct_txns = df[(df["sender_id"] == account) | (df["receiver_id"] == account)]
        if len(acct_txns) < 2:
            continue
        
        acct_txns = acct_txns.sort_values("timestamp")
        span_h = (
            acct_txns["timestamp"].max() - acct_txns["timestamp"].min()
        ).total_seconds() / 3600
        if span_h == 0:
            span_h = 1
        velocities[account] = acct_txns["amount"].sum() / span_h
        last_tx_ts[account] = str(acct_txns["timestamp"].max())

    if not velocities:
        return set(), {}

    global_avg = float(np.mean(list(velocities.values())))
    if global_avg == 0:
        return set(), {}

    high_v_ids = {a for a, v in velocities.items() if v / global_avg > multiplier}
    trigger_ts = {a: last_tx_ts[a] for a in high_v_ids}

    return high_v_ids, trigger_ts
