"""
Network Analysis — Neighbor-based risk propagation and connectivity metrics.

Provides:
  1. _build_neighbor_map: undirected neighbor map from transactions
  2. propagate_group_risk: iterative risk diffusion across neighbors
  3. compute_component_concentration: cluster analysis of top-risk accounts

Time Complexity: O(V × iterations) for propagation
Memory: O(V + E)
"""

from typing import Dict, Set

import numpy as np
import pandas as pd


def build_neighbor_map(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build an undirected neighbor map where neighbors are transaction partners.
    Expects df to have: sender_id, receiver_id.
    """
    neighbors: Dict[str, Set[str]] = {}
    for s, r in zip(df["sender_id"].astype(str), df["receiver_id"].astype(str), strict=False):
        if s not in neighbors:
            neighbors[s] = set()
        if r not in neighbors:
            neighbors[r] = set()
        if s != r:
            neighbors[s].add(r)
            neighbors[r].add(s)
    return neighbors


def propagate_group_risk(
    base_risk: Dict[str, float],
    neighbors: Dict[str, Set[str]],
    *,
    alpha: float = 0.2,
    iterations: int = 3,
    min_neighbors: int = 1,
) -> Dict[str, float]:
    """
    Propagate risk across transaction-neighbor links to capture group behavior.

    Update rule:
        for t in iterations:
            risk(a) += alpha * mean(risk(n)) for n in neighbors(a)

    Notes for precision / false positive control:
    - Accounts with < min_neighbors are not updated (prevents spurious boosts).
    - Uses synchronous updates per iteration (all accounts update together).
    """
    if iterations <= 0:
        return dict(base_risk)
    if alpha < 0:
        alpha = 0.0

    risk = dict(base_risk)
    for _ in range(iterations):
        prev = risk
        risk = dict(prev)
        for acct, nbrs in neighbors.items():
            if len(nbrs) < min_neighbors:
                continue
            vals = [prev.get(n, 0.0) for n in nbrs]
            if not vals:
                continue
            risk[acct] = float(prev.get(acct, 0.0) + alpha * float(np.mean(vals)))
    return risk


def compute_component_concentration(
    neighbors: Dict[str, Set[str]], scores: Dict[str, float], *, top_n: int = 50
) -> Dict[str, float]:
    """
    Simple, label-free metric to quantify whether top-risk accounts cluster together.

    Returns:
      - top_component_concentration: max share of top-N accounts in one component (0..1)
      - top_isolation_rate: share of top-N accounts with degree <= 1 (0..1)
    """
    if top_n <= 0 or not scores:
        return {"top_component_concentration": 0.0, "top_isolation_rate": 0.0}

    # Get top accounts by score (ties arbitrary but stable enough for reporting).
    top_accounts = [a for a, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    top_set = set(top_accounts)

    # BFS components (only need to label nodes we might touch)
    comp_id: Dict[str, int] = {}
    cid = 0
    for node in neighbors.keys():
        if node in comp_id:
            continue
        cid += 1
        stack = [node]
        comp_id[node] = cid
        while stack:
            u = stack.pop()
            for v in neighbors.get(u, set()):
                if v not in comp_id:
                    comp_id[v] = cid
                    stack.append(v)

    # Count which components contain the top-N accounts
    comp_counts: Dict[int, int] = {}
    isolated = 0
    for a in top_accounts:
        deg = len(neighbors.get(a, set()))
        if deg <= 1:
            isolated += 1
        c = comp_id.get(a, -1)
        comp_counts[c] = comp_counts.get(c, 0) + 1

    concentration = max(comp_counts.values()) / max(1, len(top_set)) if comp_counts else 0.0
    isolation_rate = isolated / max(1, len(top_set))
    return {
        "top_component_concentration": float(concentration),
        "top_isolation_rate": float(isolation_rate),
    }
