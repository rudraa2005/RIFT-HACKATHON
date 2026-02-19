"""
Smurfing Detection Module — Fan-In / Fan-Out Structuring.

Detects structuring / smurfing patterns in a 72-hour sliding window.
Delegates to fan_in and fan_out sub-modules.

Time Complexity: O(n × k)
Memory: O(n)
"""

from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from core.ring_detection.fan_in import detect_fan_in
from core.ring_detection.fan_out import detect_fan_out


def detect_smurfing(
    df: pd.DataFrame,
    min_senders_override: float | None = None,
    min_receivers_override: float | None = None,
) -> Tuple[List[Dict[str, Any]], Set[str], Set[str], Dict[str, Dict[str, str]]]:
    """
    Detect smurfing patterns (fan-in aggregators and fan-out dispersers).

    Returns:
        (rings_list, aggregator_set, disperser_set, trigger_times)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fan_in_rings, aggregators, in_triggers = detect_fan_in(df, min_senders_override)
    fan_out_rings, dispersers, out_triggers = detect_fan_out(
        df, ring_start=len(fan_in_rings), min_receivers_override=min_receivers_override
    )

    trigger_times = {
        "smurfing_aggregator": in_triggers,
        "smurfing_disperser": out_triggers,
    }

    return fan_in_rings + fan_out_rings, aggregators, dispersers, trigger_times
