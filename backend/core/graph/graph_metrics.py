"""
Graph Metrics — summary statistics for the transaction graph.

Time Complexity: O(V + E)
Memory: O(1)
"""

from typing import Any, Dict

import networkx as nx


def compute_graph_summary(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Return basic graph-level metrics."""
    simple = nx.DiGraph(G)
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "unique_edges": simple.number_of_edges(),
        "density": round(nx.density(simple), 4),
        "is_weakly_connected": nx.is_weakly_connected(simple) if simple.number_of_nodes() > 0 else False,
        "num_weakly_connected_components": nx.number_weakly_connected_components(simple),
    }
