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
    G: nx.MultiDiGraph,
) -> float:
    """
    Enhanced ring risk = 
    (Structural Strength × 0.4) + 
    (Velocity Score × 0.2) + 
    (Retention Score × 0.2) + 
    (Temporal Density × 0.2)
    """
    members = ring["members"]
    if not members:
        return float(ring.get("risk_score", 0))

    # 1. Structural Strength (Base risk from detection)
    base_structural = float(ring.get("risk_score", 50))
    
    # 2. Velocity Score (Percentage of members with high velocity)
    velocity_count = sum(1 for m in members if m in high_velocity)
    velocity_score = (velocity_count / len(members)) * 100
    
    # 3. Retention Score (Percentage of members with low retention)
    # Check both specific flags and general low_retention pattern
    retention_count = 0
    for m in members:
        patterns = scores.get(m, {}).get("patterns", [])
        if any(p in patterns for p in ["low_retention_pass_through", "rapid_pass_through", "flow_chain_member"]):
            retention_count += 1
    retention_score = (retention_count / len(members)) * 100
    
    # 4. Temporal Density (Scaled to 100)
    density_score = density * 100

    # 5. Network Influence (Proxy for Centrality Dispersion)
    # Average degree in the subgraph relative to the whole graph
    subgraph_G = G.subgraph(members)
    avg_sub_degree = sum(dict(subgraph_G.degree()).values()) / len(members) if members else 0
    centrality_score = min(100, avg_sub_degree * 20)

    # 6. Ring-Specific Jitter (Ensure unique resolution)
    # Uses hash of member set to add a tiny, consistent offset (max 0.05)
    jitter = (hash(frozenset(members)) % 100) / 2000.0
    
    risk = (
        (base_structural * 0.45) +
        (velocity_score * 0.15) +
        (retention_score * 0.15) +
        (density_score * 0.15) +
        (centrality_score * 0.1) +
        jitter
    )
    
    return float(min(100, risk))


def finalize_ring_risks(
    G: nx.MultiDiGraph,
    all_rings: list,
    scores: Dict[str, Dict[str, Any]],
    high_velocity: Set[str],
) -> list:
    """Apply density computation and enhanced risk scoring to all rings."""
    simple_G = nx.DiGraph(G)
    degree_map = dict(simple_G.degree())
    for ring in all_rings:
        members = [str(m) for m in ring.get("members", [])]
        ring["members"] = members
        n = len(members)
        if n < 2:
            density = 0.0
            avg_sub_degree = 0.0
        else:
            max_edges = n * (n - 1)
            subgraph = simple_G.subgraph(members)
            actual_edges = subgraph.number_of_edges()
            density = round(actual_edges / max_edges, 2) if max_edges > 0 else 0.0
            avg_sub_degree = sum(degree_map.get(m, 0.0) for m in members) / n

        ring["density_score"] = density

        base_structural = float(ring.get("risk_score", 50))
        velocity_count = sum(1 for m in members if m in high_velocity)
        velocity_score = (velocity_count / n) * 100 if n else 0.0

        retention_count = 0
        for m in members:
            patterns = scores.get(m, {}).get("patterns", [])
            if any(p in patterns for p in ["low_retention_pass_through", "rapid_pass_through", "flow_chain_member"]):
                retention_count += 1
        retention_score = (retention_count / n) * 100 if n else 0.0
        density_score = density * 100
        centrality_score = min(100, avg_sub_degree * 20)
        jitter = (hash(frozenset(members)) % 100) / 2000.0

        risk = (
            (base_structural * 0.45) +
            (velocity_score * 0.15) +
            (retention_score * 0.15) +
            (density_score * 0.15) +
            (centrality_score * 0.1) +
            jitter
        )
        ring["risk_score"] = float(min(100, risk))
        ring["density_score"] = density
        if density > RING_DENSITY_THRESHOLD:
            ring["risk_score"] = round(
                min(100, ring["risk_score"] * (1 + RING_DENSITY_BONUS_PERCENT / 100)),
                2,
            )
    return all_rings
