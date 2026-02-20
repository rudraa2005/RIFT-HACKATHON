"""
Throughput Ratio Analysis.

Throughput = Total_Outflow / Total_Inflow
If 0.9 ≤ Throughput ≤ 1.1 → strong pass-through
Pattern: "high_throughput_ratio", Score: +20

Time Complexity: O(V)
Memory: O(V)
"""

from typing import Set

import pandas as pd

from app.config import THROUGHPUT_LOW, THROUGHPUT_HIGH


def detect_high_throughput(df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts with throughput ratio near 1.0 (strong pass-through).

    Time Complexity: O(V)
    """
    inflow = df.groupby("receiver_id", observed=True)["amount"].sum()
    outflow = df.groupby("sender_id", observed=True)["amount"].sum()
    all_accounts = inflow.index.union(outflow.index)

    inflow_all = inflow.reindex(all_accounts, fill_value=0.0).astype(float)
    outflow_all = outflow.reindex(all_accounts, fill_value=0.0).astype(float)

    valid = inflow_all > 0
    if not valid.any():
        return set()

    throughput = outflow_all[valid] / inflow_all[valid]
    flagged_idx = throughput[(throughput >= THROUGHPUT_LOW) & (throughput <= THROUGHPUT_HIGH)].index
    return {str(account) for account in flagged_idx}
