"""
Smurfing Detection Module — Fan-In / Fan-Out Structuring.

Detects structuring / smurfing patterns in a 72-hour sliding window:

Fan-In (Aggregator):
  - ≥10 unique senders → 1 receiver within 72 h
  - Mean inbound amount < global_median × 0.6
  - Forward ratio > 0.7

Fan-Out (Disperser):
  - 1 sender → ≥10 unique receivers within 72 h
  - CV of outgoing amounts < 0.3

Time Complexity: O(n × k) where n = transactions, k = window iterations per account
Memory: O(n) for grouped DataFrames
"""

from datetime import timedelta
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from app.config import (
    SMURFING_FANOUT_CV_THRESHOLD,
    SMURFING_FORWARD_RATIO_THRESHOLD,
    SMURFING_MEAN_AMOUNT_FACTOR,
    SMURFING_MIN_COUNTERPARTIES,
    SMURFING_WINDOW_HOURS,
)


def detect_smurfing(
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], Set[str], Set[str]]:
    """
    Detect smurfing patterns (fan-in aggregators and fan-out dispersers).

    Args:
        df: Transaction DataFrame

    Returns:
        (rings_list, aggregator_set, disperser_set)

    Time Complexity: O(n × k) per account group
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    global_median = df["amount"].median()
    window_delta = timedelta(hours=SMURFING_WINDOW_HOURS)

    aggregators: Set[str] = set()
    dispersers: Set[str] = set()
    rings: List[Dict[str, Any]] = []
    ring_counter = 0

    # ── Fan-In Detection ──────────────────────────────────────────────
    for receiver, group in df.groupby("receiver_id"):
        group = group.sort_values("timestamp")
        if len(group) < SMURFING_MIN_COUNTERPARTIES:
            continue

        for i in range(len(group)):
            window_end = group["timestamp"].iloc[i]
            window_start = window_end - window_delta
            window = group[
                (group["timestamp"] >= window_start)
                & (group["timestamp"] <= window_end)
            ]

            unique_senders = window["sender_id"].nunique()
            if unique_senders < SMURFING_MIN_COUNTERPARTIES:
                continue

            mean_amount = window["amount"].mean()
            if mean_amount >= global_median * SMURFING_MEAN_AMOUNT_FACTOR:
                continue

            # Forward ratio: funds forwarded out within the next window
            total_received = window["amount"].sum()
            forwarded = df[
                (df["sender_id"] == receiver)
                & (df["timestamp"] >= window_start)
                & (df["timestamp"] <= window_end + window_delta)
            ]["amount"].sum()

            forward_ratio = forwarded / total_received if total_received > 0 else 0
            if forward_ratio < SMURFING_FORWARD_RATIO_THRESHOLD:
                continue

            aggregators.add(str(receiver))
            ring_counter += 1
            members = list(window["sender_id"].unique()) + [str(receiver)]
            rings.append(
                {
                    "ring_id": f"RING_SMURF_{ring_counter:03d}",
                    "members": members,
                    "pattern_type": "smurfing_fan_in",
                    "risk_score": round(min(100, 70 + unique_senders * 1.5), 2),
                }
            )
            break  # one detection per receiver suffices

    # ── Fan-Out Detection ─────────────────────────────────────────────
    for sender, group in df.groupby("sender_id"):
        group = group.sort_values("timestamp")
        if len(group) < SMURFING_MIN_COUNTERPARTIES:
            continue

        for i in range(len(group)):
            window_end = group["timestamp"].iloc[i]
            window_start = window_end - window_delta
            window = group[
                (group["timestamp"] >= window_start)
                & (group["timestamp"] <= window_end)
            ]

            unique_receivers = window["receiver_id"].nunique()
            if unique_receivers < SMURFING_MIN_COUNTERPARTIES:
                continue

            amounts = window["amount"].values
            mean_amt = np.mean(amounts)
            if mean_amt == 0:
                continue
            cv = float(np.std(amounts) / mean_amt)
            if cv >= SMURFING_FANOUT_CV_THRESHOLD:
                continue

            dispersers.add(str(sender))
            ring_counter += 1
            members = [str(sender)] + list(window["receiver_id"].unique())
            rings.append(
                {
                    "ring_id": f"RING_SMURF_{ring_counter:03d}",
                    "members": members,
                    "pattern_type": "smurfing_fan_out",
                    "risk_score": round(min(100, 70 + unique_receivers * 1.5), 2),
                }
            )
            break

    return rings, aggregators, dispersers
