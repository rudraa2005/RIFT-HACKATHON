"""
Risk Propagation Module — Recursive Network Risk.

After base scoring, propagates risk through the transaction graph:
  Risk(X) = BaseScore + α × mean(Risk(neighbors))
  α = 0.2, max 3 iterations.

Scores clamped to [0, 100] after each iteration.

Pattern: "network_risk_exposure" (added if score increases by ≥5)

Time Complexity: O(iterations × (V + E))
Memory: O(V)
"""

from typing import Any, Dict, Set

import networkx as nx

from app.config import RISK_PROPAGATION_ALPHA, RISK_PROPAGATION_ITERATIONS


def propagate_risk(
    G: nx.MultiDiGraph,
    scores: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Propagate risk through the network graph.

    Modifies scores in-place, adding "network_risk_exposure" pattern
    for accounts whose score increases by ≥5 points.

    Returns:
        Updated scores dict

    Time Complexity: O(iterations × (V + E))
    """
    simple_G = nx.DiGraph(G).to_undirected()
    current_scores: Dict[str, float] = {}

    for acct, data in scores.items():
        current_scores[acct] = data["score"]

    base_scores = dict(current_scores)

    for _ in range(RISK_PROPAGATION_ITERATIONS):
        new_scores: Dict[str, float] = {}
        for node in simple_G.nodes():
            if node not in current_scores:
                continue

            neighbors = list(simple_G.neighbors(node))
            if not neighbors:
                new_scores[node] = current_scores[node]
                continue

            neighbor_scores = [
                current_scores.get(n, 0) for n in neighbors
            ]
            mean_neighbor = sum(neighbor_scores) / len(neighbor_scores)

            propagated = current_scores[node] + RISK_PROPAGATION_ALPHA * mean_neighbor
            new_scores[node] = max(0, min(100, propagated))

        current_scores = new_scores

    # Update scores with propagated values
    exposure_accounts: Set[str] = set()
    for acct in scores:
        new_val = current_scores.get(acct, scores[acct]["score"])
        increase = new_val - base_scores.get(acct, 0)

        if increase >= 5:
            exposure_accounts.add(acct)
            if "network_risk_exposure" not in scores[acct]["patterns"]:
                scores[acct]["patterns"].append("network_risk_exposure")

        scores[acct]["score"] = round(max(0, min(100, new_val)), 2)

    return scores
