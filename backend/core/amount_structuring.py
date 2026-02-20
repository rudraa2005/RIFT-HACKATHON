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

    # Vectorized rolling window stats per sender.
    tx = df[["sender_id", "timestamp", "amount"]].sort_values(["sender_id", "timestamp"])
    if tx.empty:
        return set()

    rolling = (
        tx.groupby("sender_id")
        .rolling(f"{STRUCTURING_WINDOW_HOURS}h", on="timestamp")["amount"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    if rolling.empty:
        return set()

    rolling["std"] = rolling["std"].fillna(0.0)
    valid_mean = rolling["mean"] > 0
    cv = np.where(valid_mean, rolling["std"] / rolling["mean"], np.inf)
    mask = (rolling["count"] >= STRUCTURING_MIN_TXNS) & (cv < STRUCTURING_CV_THRESHOLD)
    return set(rolling.loc[mask, "sender_id"].astype(str).unique())


