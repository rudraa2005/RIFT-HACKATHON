"""
Net Retention Ratio Analysis.

Retention = (Total_Inflow - Total_Outflow) / Total_Inflow

If retention is between -0.1 and +0.1 → pass-through behavior.
Pattern: "low_retention_pass_through", Score: +25

Time Complexity: O(V) where V = unique accounts
Memory: O(V)
"""

from typing import Set

import pandas as pd

from app.config import RETENTION_LOW, RETENTION_HIGH


def detect_low_retention(df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts with near-zero retention (pass-through mules).

    Returns:
        Set of flagged account IDs

    Time Complexity: O(V)
    """
    df = df.copy()
    flagged: Set[str] = set()

    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())

    for account in all_accounts:
        total_inflow = df[df["receiver_id"] == account]["amount"].sum()
        total_outflow = df[df["sender_id"] == account]["amount"].sum()

        if total_inflow == 0:
            continue

        retention = (total_inflow - total_outflow) / total_inflow

        # Clamp to valid bounds
        retention = max(-1.0, min(1.0, retention))

        if RETENTION_LOW <= retention <= RETENTION_HIGH:
            flagged.add(str(account))

    return flagged
