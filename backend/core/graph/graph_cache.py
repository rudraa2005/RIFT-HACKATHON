"""
Graph Cache — lightweight caching for graph construction.

Time Complexity: O(1) for cache hit, O(E) for miss
Memory: O(V + E) for cached graph
"""

import hashlib

import networkx as nx
import pandas as pd

from core.graph.graph_builder import build_graph


class GraphCache:
    """Single-entry cache for the transaction graph."""

    def __init__(self):
        self._hash: str | None = None
        self._graph: nx.MultiDiGraph | None = None

    def get_or_build(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """Return cached graph if DataFrame unchanged, else rebuild."""
        df_hash = hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()

        if df_hash == self._hash and self._graph is not None:
            return self._graph

        self._graph = build_graph(df)
        self._hash = df_hash
        return self._graph

    def invalidate(self):
        self._hash = None
        self._graph = None
