"""
Fan-Out Detection — Disperser pattern.

1 sender → ≥10 unique receivers within 72h, CV of amounts < 0.3.
"""

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
    window_delta = np.timedelta64(SMURFING_WINDOW_HOURS, "h")

    min_receivers = min_receivers_override or SMURFING_MIN_COUNTERPARTIES
    dispersers: Set[str] = set()
    trigger_timestamps: Dict[str, str] = {}
    rings: List[Dict[str, Any]] = []
    ring_counter = ring_start

    for sender, group in df.groupby("sender_id", sort=False):
        if len(group) < min_receivers:
            continue

        times = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        receivers = group["receiver_id"].astype(str).to_numpy()
        amounts = group["amount"].to_numpy(dtype=float)

        left = 0
        receiver_counts: Dict[str, int] = {}
        win_sum = 0.0
        win_sumsq = 0.0

        for i in range(len(times)):
            end_ts = times[i]
            start_ts = end_ts - window_delta

            # Add current row.
            recv_i = receivers[i]
            receiver_counts[recv_i] = receiver_counts.get(recv_i, 0) + 1
            amt_i = amounts[i]
            win_sum += amt_i
            win_sumsq += amt_i * amt_i

            # Shrink left bound to keep [start_ts, end_ts].
            while left <= i and times[left] < start_ts:
                recv_l = receivers[left]
                cnt = receiver_counts.get(recv_l, 0)
                if cnt <= 1:
                    receiver_counts.pop(recv_l, None)
                else:
                    receiver_counts[recv_l] = cnt - 1
                amt_l = amounts[left]
                win_sum -= amt_l
                win_sumsq -= amt_l * amt_l
                left += 1

            unique_receivers = len(receiver_counts)
            if unique_receivers < min_receivers:
                continue

            n_window = i - left + 1
            if n_window <= 0:
                continue

            mean_amt = win_sum / n_window
            if mean_amt == 0:
                continue
            variance = max((win_sumsq / n_window) - (mean_amt * mean_amt), 0.0)
            cv = float(np.sqrt(variance) / mean_amt)
            if cv >= SMURFING_FANOUT_CV_THRESHOLD:
                continue

            sender_id = str(sender)
            dispersers.add(sender_id)
            if sender_id not in trigger_timestamps:
                trigger_timestamps[sender_id] = str(pd.Timestamp(end_ts))
                
            ring_counter += 1
            members = [sender_id] + list(dict.fromkeys(receivers[left : i + 1]))
            
            # Explicit behavioral tagging for members
            member_patterns = {str(m): ["fan_out_participant", "structured_amount"] for m in members[1:]}
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
