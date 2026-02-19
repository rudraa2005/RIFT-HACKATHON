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
) -> Tuple[List[Dict[str, Any]], Set[str], Set[str]]:
    """
    Detect smurfing patterns (fan-in aggregators and fan-out dispersers).

    Returns:
        (rings_list, aggregator_set, disperser_set)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fan_in_rings, aggregators = detect_fan_in(df)
    fan_out_rings, dispersers = detect_fan_out(df, ring_start=len(fan_in_rings))

    return fan_in_rings + fan_out_rings, aggregators, dispersers
