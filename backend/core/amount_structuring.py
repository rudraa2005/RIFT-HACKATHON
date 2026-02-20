"""
Structured Amount Fragmentation Detection.

Within a 72-hour sliding window, if an account's outgoing transactions have:
  - CV (coefficient of variation) < 0.15
  - Transaction count ≥ 5

This indicates deliberate structuring of amounts to avoid detection.

Pattern: "structured_fragmentation", Score: +10

Time Complexity: O(V × T)
Memory: O(V)
"""

from datetime import timedelta
from typing import Set

import numpy as np
import pandas as pd

from app.config import (
    STRUCTURING_CV_THRESHOLD,
    STRUCTURING_MIN_TXNS,
    STRUCTURING_WINDOW_HOURS,
)


def detect_amount_structuring(df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts using structured (near-identical) amounts in a burst window.

    Scans each sender's 72-hour windows. If CV of amounts in any window
    is below threshold with enough transactions → flag.

    Returns:
        Set of flagged account IDs

    Time Complexity: O(V × T)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    flagged: Set[str] = set()
    window_delta = timedelta(hours=STRUCTURING_WINDOW_HOURS)

    for sender in df["sender_id"].unique():
        sender_txns = df[df["sender_id"] == sender].sort_values("timestamp")

        if len(sender_txns) < STRUCTURING_MIN_TXNS:
            continue

        # Slide through windows
        for i in range(len(sender_txns)):
            window_end = sender_txns["timestamp"].iloc[i]
            window_start = window_end - window_delta
            window = sender_txns[
                (sender_txns["timestamp"] >= window_start)
                & (sender_txns["timestamp"] <= window_end)
            ]

            if len(window) < STRUCTURING_MIN_TXNS:
                continue

            amounts = window["amount"].values
            mean_amt = float(np.mean(amounts))
            if mean_amt == 0:
                continue

            cv = float(np.std(amounts) / mean_amt)

            if cv < STRUCTURING_CV_THRESHOLD:
                flagged.add(str(sender))
                break  # One match is enough

    return flagged


