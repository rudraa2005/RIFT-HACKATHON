"""
Fan-In Detection — Aggregator pattern.

≥10 unique senders → 1 receiver within 72h, small amounts, forward ratio > 0.7.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np
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
    global_median = float(df["amount"].median())
    window_delta = np.timedelta64(SMURFING_WINDOW_HOURS, "h")

    min_senders = min_senders_override or SMURFING_MIN_COUNTERPARTIES
    aggregators: Set[str] = set()
    trigger_timestamps: Dict[str, str] = {}
    rings: List[Dict[str, Any]] = []
    ring_counter = 0

    # Pre-index outgoing transactions by sender for O(log n) window-sum lookups.
    outgoing_index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sender, s_group in df.groupby("sender_id", sort=False):
        t = s_group["timestamp"].to_numpy(dtype="datetime64[ns]")
        a = s_group["amount"].to_numpy(dtype=float)
        if len(t) == 0:
            continue
        prefix = np.concatenate(([0.0], np.cumsum(a, dtype=float)))
        outgoing_index[str(sender)] = (t, prefix)

    def _outgoing_sum(sender_id: str, start: np.datetime64, end: np.datetime64) -> float:
        idx = outgoing_index.get(sender_id)
        if idx is None:
            return 0.0
        t, prefix = idx
        l = int(np.searchsorted(t, start, side="left"))
        r = int(np.searchsorted(t, end, side="right"))
        if r <= l:
            return 0.0
        return float(prefix[r] - prefix[l])

    for receiver, group in df.groupby("receiver_id", sort=False):
        if len(group) < min_senders:
            continue

        times = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        senders = group["sender_id"].astype(str).to_numpy()
        amounts = group["amount"].to_numpy(dtype=float)

        left = 0
        sender_counts: Dict[str, int] = {}
        window_sum = 0.0

        for i in range(len(times)):
            end_ts = times[i]
            start_ts = end_ts - window_delta

            # Add current row.
            sender_i = senders[i]
            sender_counts[sender_i] = sender_counts.get(sender_i, 0) + 1
            window_sum += amounts[i]

            # Shrink left bound to keep [start_ts, end_ts].
            while left <= i and times[left] < start_ts:
                left_sender = senders[left]
                cnt = sender_counts.get(left_sender, 0)
                if cnt <= 1:
                    sender_counts.pop(left_sender, None)
                else:
                    sender_counts[left_sender] = cnt - 1
                window_sum -= amounts[left]
                left += 1

            unique_senders = len(sender_counts)
            if unique_senders < min_senders:
                continue

            n_window = i - left + 1
            if n_window <= 0:
                continue
            mean_amount = window_sum / n_window
            if mean_amount >= global_median * SMURFING_MEAN_AMOUNT_FACTOR:
                continue

            total_received = window_sum
            receiver_id = str(receiver)
            forwarded = _outgoing_sum(receiver_id, start_ts, end_ts + window_delta)

            forward_ratio = forwarded / total_received if total_received > 0 else 0
            if forward_ratio < SMURFING_FORWARD_RATIO_THRESHOLD:
                continue

            aggregators.add(receiver_id)
            if receiver_id not in trigger_timestamps:
                trigger_timestamps[receiver_id] = str(pd.Timestamp(end_ts))
                
            ring_counter += 1
            members = list(dict.fromkeys(senders[left : i + 1])) + [receiver_id]
            
            # Explicit behavioral tagging for members
            member_patterns = {str(m): ["fan_in_participant", "structured_amount"] for m in members[:-1]}
            member_patterns[receiver_id] = ["smurfing_aggregator"]
            
            rings.append(
                {
                    "ring_id": f"RING_SMURF_{ring_counter:03d}",
                    "members": members,
                    "member_patterns": member_patterns,
                    "pattern_type": "smurfing_fan_in",
                    "risk_score": round(min(100, 90 + unique_senders * 1.0), 2),
                }
            )
            break

    return rings, aggregators, trigger_timestamps
