"""
Cascading Depth Analysis — Time-Bounded Layering Detection.

For each node, compute maximum forward path length within 72-hour window.
Depth search bounded to 5 hops to avoid exponential blowup.

If depth ≥ 3 → flag.
Pattern: "deep_layered_cascade", Score: +25

Time Complexity: O(V × D × branching_factor) where D = CASCADE_MAX_HOPS
Memory: O(V × D)
"""

from typing import Set

import networkx as nx
import pandas as pd
from datetime import timedelta

from app.config import CASCADE_MAX_HOPS, CASCADE_MIN_DEPTH, CASCADE_WINDOW_HOURS


def detect_cascade_depth(G: nx.MultiDiGraph, df: pd.DataFrame) -> Set[str]:
    """
    Detect accounts involved in deep layered cascades within time windows.

    Returns:
        Set of flagged account IDs

    Time Complexity: O(V × D × avg_out_degree), bounded by CASCADE_MAX_HOPS
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    window_delta = timedelta(hours=CASCADE_WINDOW_HOURS)

    flagged: Set[str] = set()
    simple_G = nx.DiGraph(G)

    # Pre-build edge timestamp map for fast lookup
    edge_times: dict = {}
    for _, row in df.iterrows():
        key = (str(row["sender_id"]), str(row["receiver_id"]))
        if key not in edge_times:
            edge_times[key] = []
        edge_times[key].append(row["timestamp"])

    for start_node in simple_G.nodes():
        start_times = []
        for _, row in df[df["sender_id"] == start_node].iterrows():
            start_times.append(row["timestamp"])

        if not start_times:
            continue

        for start_ts in start_times[:3]:  # Limit starting points for performance
            max_depth = 0
            stack = [(start_node, 0, start_ts)]
            visited_in_path: set = {start_node}

            while stack:
                node, depth, last_ts = stack.pop()
                max_depth = max(max_depth, depth)

                if depth >= CASCADE_MAX_HOPS:
                    continue

                for neighbor in simple_G.successors(node):
                    if neighbor in visited_in_path:
                        continue

                    key = (str(node), str(neighbor))
                    times = edge_times.get(key, [])

                    valid = False
                    for t in times:
                        if t >= last_ts and (t - start_ts) <= window_delta:
                            visited_in_path.add(neighbor)
                            stack.append((neighbor, depth + 1, t))
                            valid = True
                            break

            if max_depth >= CASCADE_MIN_DEPTH:
                flagged.add(str(start_node))

    return flagged
