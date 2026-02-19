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
    for _, row in df.iterrows():
        G.add_edge(
            row["sender_id"],
            row["receiver_id"],
            amount=float(row["amount"]),
            timestamp=str(row["timestamp"]),
            transaction_id=str(row["transaction_id"]),
        )
    return G
