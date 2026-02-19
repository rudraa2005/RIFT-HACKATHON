"""
Graph Builder — constructs directed multigraph from transaction data.

Time Complexity: O(E) where E = number of transactions
Memory: O(V + E)
"""

import pandas as pd
import networkx as nx


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a NetworkX MultiDiGraph from transaction DataFrame."""
    G = nx.MultiDiGraph()
    
    # Use zip for much faster iteration than iterrows()
    edges = zip(
        df["sender_id"],
        df["receiver_id"],
        [
            {"amount": float(a), "timestamp": str(t), "transaction_id": str(tid)}
            for a, t, tid in zip(df["amount"], df["timestamp"], df["transaction_id"])
        ]
    )
    G.add_edges_from(edges)
    return G
