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
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Detect fan-out disperser patterns.

    Args:
        df: Transaction DataFrame (already sorted by timestamp)
        ring_start: Starting ring counter offset

    Returns:
        (rings_list, disperser_set)
    """
    window_delta = timedelta(hours=SMURFING_WINDOW_HOURS)

    dispersers: Set[str] = set()
    rings: List[Dict[str, Any]] = []
    ring_counter = ring_start

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

    return rings, dispersers
