"""
Fan-In Detection — Aggregator pattern.

≥10 unique senders → 1 receiver within 72h, small amounts, forward ratio > 0.7.
"""

from datetime import timedelta
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from app.config import (
    SMURFING_FORWARD_RATIO_THRESHOLD,
    SMURFING_MEAN_AMOUNT_FACTOR,
    SMURFING_MIN_COUNTERPARTIES,
    SMURFING_WINDOW_HOURS,
)


def detect_fan_in(
    df: pd.DataFrame,
    min_senders_override: float | None = None,
) -> Tuple[List[Dict[str, Any]], Set[str], Dict[str, str]]:
    """
    Detect fan-in aggregator patterns.

    Returns:
        (rings_list, aggregator_set, trigger_timestamps)
    """
    global_median = df["amount"].median()
    window_delta = timedelta(hours=SMURFING_WINDOW_HOURS)

    min_senders = min_senders_override or SMURFING_MIN_COUNTERPARTIES
    aggregators: Set[str] = set()
    trigger_timestamps: Dict[str, str] = {}
    rings: List[Dict[str, Any]] = []
    ring_counter = 0

    for receiver, group in df.groupby("receiver_id"):
        group = group.sort_values("timestamp")
        if len(group) < min_senders:
            continue

        for i in range(len(group)):
            window_end = group["timestamp"].iloc[i]
            window_start = window_end - window_delta
            window = group[
                (group["timestamp"] >= window_start)
                & (group["timestamp"] <= window_end)
            ]

            unique_senders = window["sender_id"].nunique()
            if unique_senders < min_senders:
                continue

            mean_amount = window["amount"].mean()
            if mean_amount >= global_median * SMURFING_MEAN_AMOUNT_FACTOR:
                continue

            total_received = window["amount"].sum()
            forwarded = df[
                (df["sender_id"] == receiver)
                & (df["timestamp"] >= window_start)
                & (df["timestamp"] <= window_end + window_delta)
            ]["amount"].sum()

            forward_ratio = forwarded / total_received if total_received > 0 else 0
            if forward_ratio < SMURFING_FORWARD_RATIO_THRESHOLD:
                continue

            receiver_id = str(receiver)
            aggregators.add(receiver_id)
            if receiver_id not in trigger_timestamps:
                trigger_timestamps[receiver_id] = str(window_end)
                
            ring_counter += 1
            members = list(window["sender_id"].unique()) + [receiver_id]
            rings.append(
                {
                    "ring_id": f"RING_SMURF_{ring_counter:03d}",
                    "members": members,
                    "pattern_type": "smurfing_fan_in",
                    "risk_score": round(min(100, 70 + unique_senders * 1.5), 2),
                }
            )
            break

    return rings, aggregators, trigger_timestamps
