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
    flagged: Set[str] = set()
    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())

    for account in all_accounts:
        total_inflow = df[df["receiver_id"] == account]["amount"].sum()
        total_outflow = df[df["sender_id"] == account]["amount"].sum()

        if total_inflow == 0:
            continue

        throughput = total_outflow / total_inflow

        if THROUGHPUT_LOW <= throughput <= THROUGHPUT_HIGH:
            flagged.add(str(account))

    return flagged
