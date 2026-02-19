"""
Inflow/Outflow Computation.

Basic per-account inflow and outflow aggregation utility.

Time Complexity: O(V)
Memory: O(V)
"""

from typing import Dict, Tuple

import pandas as pd


def compute_inflow_outflow(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Compute total inflow and outflow per account.

    Returns:
        {account_id: (total_inflow, total_outflow)}
    """
    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
    result: Dict[str, Tuple[float, float]] = {}

    for account in all_accounts:
        total_inflow = float(df[df["receiver_id"] == account]["amount"].sum())
        total_outflow = float(df[df["sender_id"] == account]["amount"].sum())
        result[str(account)] = (total_inflow, total_outflow)

    return result
