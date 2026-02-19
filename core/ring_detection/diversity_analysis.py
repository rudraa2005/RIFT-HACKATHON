"""
Sender Diversity Analysis — Time-Normalized Burst Detection.

Pattern: "high_burst_diversity", Score: +20

Time Complexity: O(V × T)
Memory: O(V)
"""

from typing import Set

import pandas as pd

from app.config import MERCHANT_MIN_SPAN_DAYS


def detect_burst_diversity(df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts receiving from many unique senders in a short burst.

    Flags accounts where:
      - sender_diversity > 0.7
      - time span < 30 days (not a long-running merchant)
      - at least 5 incoming transactions
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    flagged: Set[str] = set()

    for account in df["receiver_id"].unique():
        incoming = df[df["receiver_id"] == account]

        if len(incoming) < 5:
            continue

        unique_senders = incoming["sender_id"].nunique()
        total_txns = len(incoming)
        diversity = unique_senders / total_txns

        time_span_days = (
            incoming["timestamp"].max() - incoming["timestamp"].min()
        ).total_seconds() / 86400

        if time_span_days > MERCHANT_MIN_SPAN_DAYS:
            continue

        if diversity > 0.7 and time_span_days < 30:
            flagged.add(str(account))

    return flagged
