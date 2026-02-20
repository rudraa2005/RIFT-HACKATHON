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
    flagged: Set[str] = set()
    trigger_times: Dict[str, str] = {}

    for account in df["receiver_id"].unique():
        incoming = df[df["receiver_id"] == account].sort_values("timestamp")

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
            account_id = str(account)
            flagged.add(account_id)
            trigger_times[account_id] = str(incoming["timestamp"].max())

    return flagged, trigger_times


