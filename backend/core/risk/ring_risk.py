"""
Ring Risk Computation.

Computes enhanced ring risk using member scores, density, and velocity.
Extracted from processing_pipeline for modularity.
"""

from typing import Any, Dict, Set

import networkx as nx

from app.config import RING_DENSITY_BONUS_PERCENT, RING_DENSITY_THRESHOLD


def compute_ring_density(
    G: nx.MultiDiGraph, ring: Dict[str, Any]
) -> float:
    """
    Compute density for a ring: actual_edges / max_possible_edges.
    max_possible_edges = n × (n - 1) for directed graph.
    """
    members = ring["members"]
    n = len(members)
    if n < 2:
        return 0.0
    max_edges = n * (n - 1)
    simple_G = nx.DiGraph(G)
    subgraph = simple_G.subgraph(members)
    actual_edges = subgraph.number_of_edges()
    return round(actual_edges / max_edges, 2) if max_edges > 0 else 0.0


def enhance_ring_risk(
    ring: Dict[str, Any],
    density: float,
    scores: Dict[str, Dict[str, Any]],
    high_velocity: Set[str],
) -> float:
    """
    Enhanced ring risk = avg(member_scores) + (density × 20) + velocity_multiplier.
    """
    members = ring["members"]
    if not members:
        return ring.get("risk_score", 0)

    member_scores = [scores.get(m, {}).get("score", 0) for m in members]
    avg_score = sum(member_scores) / len(member_scores)

    density_component = density * 20

    velocity_count = sum(1 for m in members if m in high_velocity)
    velocity_multiplier = 10 if velocity_count >= len(members) * 0.5 else 0

    risk = max(float(ring.get("risk_score", 0)), avg_score + density_component + velocity_multiplier)
    return round(min(100, risk), 2)


def finalize_ring_risks(
    G: nx.MultiDiGraph,
    all_rings: list,
    scores: Dict[str, Dict[str, Any]],
    high_velocity: Set[str],
) -> list:
    """Apply density computation and enhanced risk scoring to all rings."""
    for ring in all_rings:
        density = compute_ring_density(G, ring)
        ring["density_score"] = density
        ring["risk_score"] = enhance_ring_risk(ring, density, scores, high_velocity)
        if density > RING_DENSITY_THRESHOLD:
            ring["risk_score"] = round(
                min(100, ring["risk_score"] * (1 + RING_DENSITY_BONUS_PERCENT / 100)),
                2,
            )
    return all_rings
