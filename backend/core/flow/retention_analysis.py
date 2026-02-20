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

    inflow = df.groupby("receiver_id", observed=True)["amount"].sum()
    outflow = df.groupby("sender_id", observed=True)["amount"].sum()
    all_accounts = inflow.index.union(outflow.index)

    inflow_all = inflow.reindex(all_accounts, fill_value=0.0).astype(float)
    outflow_all = outflow.reindex(all_accounts, fill_value=0.0).astype(float)

    valid = inflow_all > 0
    if not valid.any():
        return set()

    retention = (inflow_all[valid] - outflow_all[valid]) / inflow_all[valid]
    retention = retention.clip(lower=-1.0, upper=1.0)
    flagged_idx = retention[(retention >= RETENTION_LOW) & (retention <= RETENTION_HIGH)].index
    return {str(account) for account in flagged_idx}
