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


def compute_high_velocity_accounts(df: pd.DataFrame) -> Set[str]:
    """
    Identify accounts with normalized velocity > threshold.

    Returns:
        Set of high-velocity account IDs

    Time Complexity: O(V)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
    velocities: Dict[str, float] = {}

    for account in all_accounts:
        acct_txns = df[(df["sender_id"] == account) | (df["receiver_id"] == account)]
        if len(acct_txns) < 2:
            continue
        span_h = (
            acct_txns["timestamp"].max() - acct_txns["timestamp"].min()
        ).total_seconds() / 3600
        if span_h == 0:
            span_h = 1
        velocities[account] = acct_txns["amount"].sum() / span_h

    if not velocities:
        return set()

    global_avg = float(np.mean(list(velocities.values())))
    if global_avg == 0:
        return set()

    return {a for a, v in velocities.items() if v / global_avg > HIGH_VELOCITY_THRESHOLD}
