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


def detect_cascade_depth(G: nx.MultiDiGraph, df: pd.DataFrame) -> tuple[list[dict[str, any]], Set[str]]:
    """
    Detect accounts involved in deep layered cascades within time windows.

    Returns:
        (rings_list, flagged_account_ids_set)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    window_delta = timedelta(hours=CASCADE_WINDOW_HOURS)

    flagged: Set[str] = set()
    rings: list[dict[str, any]] = []
    simple_G = nx.DiGraph(G)

    # Pre-build edge timestamp map for fast lookup
    edge_times: dict = {}
    for _, row in df.iterrows():
        key = (str(row["sender_id"]), str(row["receiver_id"]))
        if key not in edge_times:
            edge_times[key] = []
        edge_times[key].append(row["timestamp"])

    ring_idx = 1
    for start_node in simple_G.nodes():
        start_node_str = str(start_node)
        start_times = []
        # Filter df for start_node sender efficiency
        node_df = df[df["sender_id"] == start_node]
        start_times = node_df["timestamp"].tolist()

        if not start_times:
            continue

        for start_ts in start_times[:3]:  # Limit starting points for performance
            max_depth = 0
            # store path in stack: (node, depth, last_ts, path)
            stack = [(start_node_str, 0, start_ts, [start_node_str])]
            best_chain = [start_node_str]

            while stack:
                node, depth, last_ts, path = stack.pop()
                if depth > max_depth:
                    max_depth = depth
                    best_chain = path

                if depth >= CASCADE_MAX_HOPS:
                    continue

                for neighbor in simple_G.successors(node):
                    neighbor_str = str(neighbor)
                    if neighbor_str in path:
                        continue

                    key = (node, neighbor_str)
                    times = edge_times.get(key, [])

                    for t in times:
                        if t >= last_ts and (t - start_ts) <= window_delta:
                            stack.append((neighbor_str, depth + 1, t, path + [neighbor_str]))
                            break

            if max_depth >= CASCADE_MIN_DEPTH:
                flagged.add(start_node_str)
                rings.append({
                    "ring_id": f"RING_CASCADE_{ring_idx:03d}",
                    "members": best_chain,
                    "pattern_type": "deep_layered_cascade",
                    "risk_score": round(min(100, 50 + max_depth * 10), 2)
                })
                ring_idx += 1
                break

    return rings, flagged


