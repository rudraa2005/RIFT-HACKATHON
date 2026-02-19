"""
Extended Centrality — Closeness Centrality Analysis.

Computes closeness centrality for nodes.
Flags nodes above the 95th percentile within the suspicious subgraph.

Pattern: "high_closeness_centrality", Score: +15

Time Complexity: O(V × E) exact
Memory: O(V)
"""

from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np

from app.config import CENTRALITY_PERCENTILE


def compute_closeness_centrality(
    G: nx.MultiDiGraph,
    suspicious_accounts: Set[str] | None = None,
) -> Tuple[Set[str], Dict[str, float]]:
    """
    Compute closeness centrality and flag high-centrality nodes.

    Returns:
        (high_closeness_accounts, all_closeness_scores)
    """
    simple_G = nx.DiGraph(G)

    if suspicious_accounts and len(suspicious_accounts) >= 3:
        subgraph = simple_G.subgraph(
            [n for n in suspicious_accounts if n in simple_G]
        )
        if subgraph.number_of_nodes() < 3:
            return set(), {}
        target_G = subgraph
    else:
        target_G = simple_G

    n = target_G.number_of_nodes()
    if n == 0:
        return set(), {}

    closeness = nx.closeness_centrality(target_G)

    if not closeness:
        return set(), {}

    values = list(closeness.values())
    threshold = float(np.percentile(values, CENTRALITY_PERCENTILE))

    high_closeness: Set[str] = {
        node for node, score in closeness.items()
        if score > threshold and score > 0
    }

    return high_closeness, {k: round(v, 4) for k, v in closeness.items()}
