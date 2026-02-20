"""
Sender Diversity Analysis — Time-Normalized Burst Detection.

Pattern: "high_burst_diversity", Score: +20

Time Complexity: O(V × T)
Memory: O(V)
"""

from typing import Set, Tuple, Dict
import pandas as pd
from app.config import MERCHANT_MIN_SPAN_DAYS

def detect_burst_diversity(df: pd.DataFrame) -> Tuple[Set[str], Dict[str, str]]:
    """
    Detect accounts receiving from many unique senders in a short burst.

    Returns:
        Tuple of (flagged_set, trigger_times)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    incoming = df[["receiver_id", "sender_id", "timestamp"]]
    if incoming.empty:
        return set(), {}

    stats = incoming.groupby("receiver_id").agg(
        total_txns=("sender_id", "size"),
        unique_senders=("sender_id", "nunique"),
        min_ts=("timestamp", "min"),
        max_ts=("timestamp", "max"),
    )
    stats = stats[stats["total_txns"] >= 5]
    if stats.empty:
        return set(), {}

    stats["diversity"] = stats["unique_senders"] / stats["total_txns"]
    stats["time_span_days"] = (stats["max_ts"] - stats["min_ts"]).dt.total_seconds() / 86400.0

    hits = stats[
        (stats["time_span_days"] <= MERCHANT_MIN_SPAN_DAYS)
        & (stats["time_span_days"] < 30.0)
        & (stats["diversity"] > 0.7)
    ]

    flagged = set(hits.index.astype(str))
    trigger_times = {str(acct): str(ts) for acct, ts in hits["max_ts"].to_dict().items()}
    return flagged, trigger_times


