"""
Builds a directed multigraph from transaction data using NetworkX.

Nodes represent account IDs. Edges represent individual transactions
with attributes: amount, timestamp, transaction_id.

Time Complexity: O(E) where E = number of transactions
Memory: O(V + E) where V = unique accounts, E = transactions
"""

import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Build a directed multigraph from a transaction DataFrame.

    Uses MultiDiGraph to support multiple transactions between the same pair of accounts.

    Args:
        df: DataFrame with columns [transaction_id, sender_id, receiver_id, amount, timestamp]

    Returns:
        nx.MultiDiGraph with edge attributes (amount, timestamp, transaction_id)

    Time Complexity: O(E)
    Space Complexity: O(V + E)
    """
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


def get_edge_data_between(G: nx.MultiDiGraph, u: str, v: str) -> list:
    """Get all edge data dicts between two nodes. O(degree)."""
    if G.has_edge(u, v):
        return [data for _, data in G[u][v].items()]
    return []


def get_all_edges_for_node(G: nx.MultiDiGraph, node: str) -> list:
    """Get all edges (in + out) for a node with their data. O(degree)."""
    edges = []
    for _, v, data in G.out_edges(node, data=True):
        edges.append({"direction": "out", "counterparty": v, **data})
    for u, _, data in G.in_edges(node, data=True):
        edges.append({"direction": "in", "counterparty": u, **data})
    return edges
