"""
Fan-Out Detection — Disperser pattern.

1 sender → ≥10 unique receivers within 72h, CV of amounts < 0.3.
"""

from datetime import timedelta
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from app.config import (
    SMURFING_FANOUT_CV_THRESHOLD,
    SMURFING_MIN_COUNTERPARTIES,
    SMURFING_WINDOW_HOURS,
)


def detect_fan_out(
    df: pd.DataFrame,
    ring_start: int = 0,
    min_receivers_override: float | None = None,
) -> Tuple[List[Dict[str, Any]], Set[str], Dict[str, str]]:
    """
    Detect fan-out disperser patterns.

    Args:
        df: Transaction DataFrame (already sorted by timestamp)
        ring_start: Starting ring counter offset
        min_receivers_override: Adaptive threshold for receivers

    Returns:
        (rings_list, disperser_set, trigger_timestamps)
    """
    window_delta = timedelta(hours=SMURFING_WINDOW_HOURS)

    min_receivers = min_receivers_override or SMURFING_MIN_COUNTERPARTIES
    dispersers: Set[str] = set()
    trigger_timestamps: Dict[str, str] = {}
    rings: List[Dict[str, Any]] = []
    ring_counter = ring_start

    for sender, group in df.groupby("sender_id"):
        group = group.sort_values("timestamp")
        if len(group) < min_receivers:
            continue

        for i in range(len(group)):
            window_end = group["timestamp"].iloc[i]
            window_start = window_end - window_delta
            window = group[
                (group["timestamp"] >= window_start)
                & (group["timestamp"] <= window_end)
            ]

            unique_receivers = window["receiver_id"].nunique()
            if unique_receivers < min_receivers:
                continue

            amounts = window["amount"].values
            mean_amt = np.mean(amounts)
            if mean_amt == 0:
                continue
            cv = float(np.std(amounts) / mean_amt)
            if cv >= SMURFING_FANOUT_CV_THRESHOLD:
                continue

            sender_id = str(sender)
            dispersers.add(sender_id)
            if sender_id not in trigger_timestamps:
                trigger_timestamps[sender_id] = str(window_end)
                
            ring_counter += 1
            members = [sender_id] + list(window["receiver_id"].unique())
            
            # Explicit behavioral tagging for members
            member_patterns = {str(m): ["fan_out_participant", "structured_amount"] for m in window["receiver_id"].unique()}
            member_patterns[sender_id] = ["smurfing_disperser"]
            
            rings.append(
                {
                    "ring_id": f"RING_SMURF_{ring_counter:03d}",
                    "members": members,
                    "member_patterns": member_patterns,
                    "pattern_type": "smurfing_fan_out",
                    "risk_score": round(min(100, 90 + unique_receivers * 1.0), 2),
                }
            )
            break

    return rings, dispersers, trigger_timestamps
