"""
Risk Propagation Module — Depth-Limited Network Risk.

After base scoring, propagates risk through the transaction graph:
  - Only 1 iteration (single hop)
  - Only propagates FROM accounts that have structural patterns
  - 0.3 decay factor
  - Minimum 15-point increase required for tagging

Time Complexity: O(V + E)
Memory: O(V)
"""

from typing import Any, Dict, Set

import networkx as nx


# Patterns that represent direct structural evidence of fraud
_STRUCTURAL_PATTERNS = {
    "cycle", "smurfing_aggregator", "smurfing_disperser",
    "shell_account", "rapid_pass_through", "rapid_forwarding",
    "deep_layered_cascade", "dormant_activation_spike",
    "structured_fragmentation",
}

PROPAGATION_DECAY = 0.3
EXPOSURE_THRESHOLD = 15.0


def propagate_risk(
    G: nx.MultiDiGraph,
    scores: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Propagate risk through the network graph (depth-limited).

    Only propagates FROM accounts that have at least one structural pattern.
    Single hop, 0.3 decay, minimum 15-point increase for exposure tag.

    Returns:
        Updated scores dict
    """
    simple_G = nx.DiGraph(G).to_undirected()

    # Identify accounts with structural patterns (confirmed suspicious)
    structural_sources: Set[str] = set()
    for acct, data in scores.items():
        acct_patterns = set(data.get("patterns", []))
        if acct_patterns & _STRUCTURAL_PATTERNS:
            structural_sources.add(acct)

    if not structural_sources:
        return scores

    base_scores = {acct: data["score"] for acct, data in scores.items()}

    # Single pass: propagate FROM structural sources to their direct neighbors
    for source in structural_sources:
        if source not in simple_G:
            continue
        source_score = base_scores.get(source, 0)
        if source_score <= 0:
            continue

        for neighbor in simple_G.neighbors(source):
            if neighbor not in scores:
                continue
            # Only boost if neighbor has LOWER score (don't boost already-high accounts)
            boost = source_score * PROPAGATION_DECAY
            current = scores[neighbor]["score"]
            if boost > EXPOSURE_THRESHOLD and current < source_score:
                scores[neighbor]["score"] = round(current + boost * 0.5, 2)
                if "network_risk_exposure" not in scores[neighbor]["patterns"]:
                    scores[neighbor]["patterns"].append("network_risk_exposure")

    return scores
