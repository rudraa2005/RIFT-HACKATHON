"""
Betweenness Centrality Risk Module.

Computes betweenness centrality and flags nodes above 95th percentile.
Uses approximate centrality (k-sample) when graph has > 500 nodes.

Time Complexity: O(V × E) exact, O(k × E) approximate
Memory: O(V)
"""

from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np

from app.config import CENTRALITY_PERCENTILE


def compute_centrality(
    G: nx.MultiDiGraph,
) -> Tuple[Set[str], Dict[str, float]]:
    """
    Compute betweenness centrality and flag high-centrality nodes.

    Returns:
        (high_centrality_accounts, all_centrality_scores)
    """
    simple_G = nx.DiGraph(G)
    n = simple_G.number_of_nodes()

    if n == 0:
        return set(), {}

    if n > 500:
        k = min(100, n)
        centrality = nx.betweenness_centrality(simple_G, k=k, normalized=True)
    else:
        centrality = nx.betweenness_centrality(simple_G, normalized=True)

    if not centrality:
        return set(), {}

    values = list(centrality.values())
    threshold = float(np.percentile(values, CENTRALITY_PERCENTILE))

    high_centrality: Set[str] = {
        node for node, score in centrality.items() if score > threshold and score > 0
    }

    return high_centrality, {k: round(v, 4) for k, v in centrality.items()}
