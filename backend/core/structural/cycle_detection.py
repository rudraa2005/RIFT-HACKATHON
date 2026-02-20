"""
Cycle Detection Module — Circular Fund Routing.

Detects suspicious cycles of length 3–5 in the transaction graph where:
  - Time span of all transactions in the cycle < 72 hours
  - Coefficient of variation (CV) of amounts < 0.25

Uses NetworkX simple_cycles with length_bound for efficiency.

Time Complexity: O(V + E) * (V-1)! worst-case for simple_cycles,
    but bounded by MAX_CYCLE_LENGTH to ~O(V * L^L) where L = max cycle length.
Memory: O(V * L) for storing discovered cycles.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Any, Dict, List

from app.config import (
    CYCLE_AMOUNT_CV_THRESHOLD,
    CYCLE_TIME_SPAN_HOURS,
    MAX_CYCLE_LENGTH,
    MIN_CYCLE_LENGTH,
)


def detect_cycles(G: nx.MultiDiGraph, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect suspicious circular fund routing using SCC decomposition for speed.
    """
    simple_G = nx.DiGraph(G)
    suspicious_cycles = []
    ring_counter = 0
    edge_stats: Dict[tuple[str, str], Dict[str, Any]] = {}
    edge_cols = df[["sender_id", "receiver_id", "amount", "timestamp"]]
    for row in edge_cols.itertuples(index=False):
        key = (str(row.sender_id), str(row.receiver_id))
        stats = edge_stats.get(key)
        ts = pd.Timestamp(row.timestamp)
        amt = float(row.amount)
        if stats is None:
            edge_stats[key] = {"sum": amt, "count": 1, "min_ts": ts, "max_ts": ts}
        else:
            stats["sum"] += amt
            stats["count"] += 1
            if ts < stats["min_ts"]:
                stats["min_ts"] = ts
            if ts > stats["max_ts"]:
                stats["max_ts"] = ts

    # Optimization: Filter for SCCs of size >= MIN_CYCLE_LENGTH
    # simple_cycles is much faster when run per component
    sccs = [c for c in nx.strongly_connected_components(simple_G) if len(c) >= MIN_CYCLE_LENGTH]

    MAX_CANDIDATE_CYCLES = 1000

    for component in sccs:
        subgraph = simple_G.subgraph(component)
        try:
            # Use an iterator to limit candidate cycles
            cycle_gen = nx.simple_cycles(subgraph, length_bound=MAX_CYCLE_LENGTH)
            all_cycles = []
            for i, cycle in enumerate(cycle_gen):
                all_cycles.append(cycle)
                if i >= MAX_CANDIDATE_CYCLES:
                    break
        except Exception:
            continue

        for cycle in all_cycles:
            if len(cycle) < MIN_CYCLE_LENGTH or len(cycle) > MAX_CYCLE_LENGTH:
                continue

            amounts = []
            min_ts = None
            max_ts = None

            for i in range(len(cycle)):
                u = str(cycle[i])
                v = str(cycle[(i + 1) % len(cycle)])
                stats = edge_stats.get((u, v))
                if stats is None:
                    continue
                avg_amount = stats["sum"] / max(stats["count"], 1)
                amounts.append(avg_amount)
                edge_min = stats["min_ts"]
                edge_max = stats["max_ts"]
                min_ts = edge_min if min_ts is None or edge_min < min_ts else min_ts
                max_ts = edge_max if max_ts is None or edge_max > max_ts else max_ts

            if not amounts or min_ts is None or max_ts is None:
                continue

            time_span = (max_ts - min_ts).total_seconds() / 3600
            if time_span > CYCLE_TIME_SPAN_HOURS:
                continue

            mean_amt = np.mean(amounts)
            if mean_amt == 0:
                continue
            cv = float(np.std(amounts) / mean_amt)
            if cv > CYCLE_AMOUNT_CV_THRESHOLD:
                continue

            ring_counter += 1
            suspicious_cycles.append(
                {
                    "ring_id": f"RING_CYCLE_{ring_counter:03d}",
                    "members": [str(m) for m in cycle],
                    "pattern_type": f"cycle_length_{len(cycle)}",
                    "amounts": [round(float(a), 2) for a in amounts],
                    "time_span_hours": round(time_span, 2),
                    "cv": round(cv, 4),
                    "risk_score": round(min(100, 60 + (1 - cv) * 40), 2),
                }
            )

    return suspicious_cycles
