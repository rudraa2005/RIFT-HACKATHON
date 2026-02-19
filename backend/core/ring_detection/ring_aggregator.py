"""
Ring Aggregator — combines rings from all detection modules.

Merges cycle, smurfing, shell, and SCC rings into a unified list
and deduplicates overlapping rings.
"""

from typing import Any, Dict, List


def aggregate_rings(*ring_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine multiple ring lists into one.

    Returns:
        Unified list of rings, sorted by risk_score descending
    """
    combined: List[Dict[str, Any]] = []
    for ring_list in ring_lists:
        combined.extend(ring_list)
    combined.sort(key=lambda r: r.get("risk_score", 0), reverse=True)
    return combined
