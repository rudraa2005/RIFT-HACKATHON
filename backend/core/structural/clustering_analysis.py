"""
Local Clustering Coefficient Analysis.

Computes local clustering coefficient for nodes.
Pattern: "high_local_clustering", Score: +15

Time Complexity: O(V × D²) where D = avg degree
Memory: O(V)
"""

from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np

from app.config import CENTRALITY_PERCENTILE


def detect_high_clustering(
    G: nx.MultiDiGraph,
    suspicious_accounts: Set[str] | None = None,
) -> Tuple[Set[str], Dict[str, float]]:
    """
    Compute local clustering coefficient and flag densely connected nodes.

    Returns:
        (high_clustering_accounts, all_clustering_scores)

    Time Complexity: O(V × D²)
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
    if n < 3:
        return set(), {}

    clustering = nx.clustering(target_G)

    if not clustering:
        return set(), {}

    values = [v for v in clustering.values() if v > 0]
    if not values:
        return set(), {k: round(v, 4) for k, v in clustering.items()}

    threshold = float(np.percentile(values, CENTRALITY_PERCENTILE))

    high_clustering: Set[str] = {
        node for node, score in clustering.items()
        if score > threshold and score > 0
    }

    return high_clustering, {k: round(v, 4) for k, v in clustering.items()}
