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

import logging
import time

import networkx as nx
import numpy as np
import pandas as pd
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

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

    # Optimization: Filter for SCCs of size >= MIN_CYCLE_LENGTH
    # simple_cycles is much faster when run per component
    sccs = [c for c in nx.strongly_connected_components(simple_G) if len(c) >= MIN_CYCLE_LENGTH]

    MAX_CANDIDATE_CYCLES = 200

    start_time = time.time()
    for component in sccs:
        # Prevent runaway processing on dense SCCs
        if (time.time() - start_time) > 10.0:
            logger.warning("Cycle detection reached 10s SCC limit, stopping early.")
            break

        subgraph = simple_G.subgraph(component)
        try:
            # Use an iterator to limit candidate cycles
            cycle_gen = nx.simple_cycles(subgraph, length_bound=MAX_CYCLE_LENGTH)
            
            for i, cycle in enumerate(cycle_gen):
                if i >= MAX_CANDIDATE_CYCLES:
                    break
                
                if len(cycle) < MIN_CYCLE_LENGTH or len(cycle) > MAX_CYCLE_LENGTH:
                    continue

                amounts = []
                timestamps = []

                # Vectorized-ish gathering: get all edges between pairs in cycle
                valid_cycle = True
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    if not G.has_edge(u, v):
                        valid_cycle = False
                        break
                    
                    # Instead of iterating, use the MultiDiGraph data
                    for _, data in G[u][v].items():
                        amounts.append(data["amount"])
                        timestamps.append(pd.Timestamp(data["timestamp"]))

                if not valid_cycle or not amounts or not timestamps:
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
        except Exception as e:
            logger.error(f"Error in SCC cycle detection: {e}")
            continue

    return suspicious_cycles
