"""
Strongly Connected Component (SCC) Analysis.

Uses NetworkX strongly_connected_components() to find SCCs of size ≥ 3.
Pattern: "large_scc_membership", Score: +20

Time Complexity: O(V + E) — Tarjan's algorithm
Memory: O(V)
"""

from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from app.config import SCC_MIN_SIZE


def detect_scc(
    G: nx.MultiDiGraph,
) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """
    Find strongly connected components of size ≥ SCC_MIN_SIZE.

    Returns:
        (flagged_accounts, scc_rings)

    Time Complexity: O(V + E)
    """
    simple_G = nx.DiGraph(G)
    flagged: Set[str] = set()
    rings: List[Dict[str, Any]] = []
    ring_counter = 0

    for scc in nx.strongly_connected_components(simple_G):
        if len(scc) >= SCC_MIN_SIZE:
            ring_counter += 1
            members = list(scc)
            flagged.update(members)

            subgraph = simple_G.subgraph(members)
            n = len(members)
            max_edges = n * (n - 1)
            density = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0

            rings.append(
                {
                    "ring_id": f"RING_SCC_{ring_counter:03d}",
                    "members": members,
                    "pattern_type": "scc_cluster",
                    "risk_score": round(min(100, 50 + n * 5 + density * 20), 2),
                    "density_score": round(density, 2),
                }
            )

    return flagged, rings
