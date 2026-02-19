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
    Detect suspicious circular fund routing in the transaction graph.

    Returns:
        List of ring dicts: {ring_id, members, pattern_type, amounts,
                             time_span_hours, cv, risk_score}

    Time Complexity: O(C * L) where C = candidate cycles, L = max cycle length
    """
    simple_G = nx.DiGraph(G)

    suspicious_cycles = []
    ring_counter = 0

    try:
        all_cycles = list(nx.simple_cycles(simple_G, length_bound=MAX_CYCLE_LENGTH))
    except Exception:
        all_cycles = []

    for cycle in all_cycles:
        if len(cycle) < MIN_CYCLE_LENGTH or len(cycle) > MAX_CYCLE_LENGTH:
            continue

        amounts = []
        timestamps = []

        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if G.has_edge(u, v):
                for _, data in G[u][v].items():
                    amounts.append(data["amount"])
                    timestamps.append(pd.Timestamp(data["timestamp"]))

        if not amounts or not timestamps:
            continue

        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
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
                "members": list(cycle),
                "pattern_type": f"cycle_length_{len(cycle)}",
                "amounts": [round(a, 2) for a in amounts],
                "time_span_hours": round(time_span, 2),
                "cv": round(cv, 4),
                "risk_score": round(min(100, 60 + (1 - cv) * 40), 2),
            }
        )

    return suspicious_cycles
