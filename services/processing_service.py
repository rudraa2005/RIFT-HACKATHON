"""
Processing Service — Pipeline Orchestrator.

Coordinates the full detection pipeline (Phases 1–3):
   1. Parse timestamps & build directed multigraph
   2. Cycle detection
   3. Smurfing detection (fan-in / fan-out)
   4. Shell chain detection
   5. Rapid pass-through (holding time)
   6. Activity spike detection
   7. Betweenness centrality
   8. Net retention ratio (NEW)
   9. Throughput ratio + balance oscillation (NEW)
  10. Sender diversity burst (NEW)
  11. SCC detection (NEW)
  12. Cascade depth (NEW)
  13. Activity consistency variance (NEW)
  14. False positive detection
  15. Compute suspicion scores
  16. Normalize scores to [0, 100]
  17. Risk propagation (NEW)
  18. Closeness centrality (NEW)
  19. Local clustering coefficient (NEW)
  20. Ring density + enhanced risk
  21. Format JSON output

Performance: < 30s for 10K transactions.
Memory: O(V + E) for graph + O(R) for rings.
"""

from typing import Any, Dict, List, Set

import networkx as nx
import pandas as pd

from app.config import RING_DENSITY_BONUS_PERCENT, RING_DENSITY_THRESHOLD
from core.activity_consistency import detect_irregular_activity
from core.activity_spike_detection import detect_activity_spikes
from core.amount_structuring import detect_amount_structuring
from core.cascade_depth_analysis import detect_cascade_depth
from core.centrality_analysis import compute_centrality
from core.centrality_extended import compute_closeness_centrality
from core.clustering_analysis import detect_high_clustering
from core.cycle_detection import detect_cycles
from core.diversity_analysis import detect_burst_diversity
from core.dormancy_analysis import detect_dormant_activation
from core.false_positive_filter import detect_false_positives
from core.flow_metrics import detect_flow_metrics
from core.forwarding_latency import detect_rapid_forwarding
from core.graph_builder import build_graph
from core.holding_time_analysis import detect_rapid_pass_through
from core.json_formatter import format_output
from core.retention_analysis import detect_low_retention
from core.risk_normalizer import normalize_scores
from core.risk_propagation import propagate_risk
from core.scc_analysis import detect_scc
from core.scoring_engine import compute_scores
from core.shell_detection import detect_shell_chains
from core.smurfing_detection import detect_smurfing


def _compute_ring_density(
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


def _enhance_ring_risk(
    ring: Dict[str, Any],
    density: float,
    scores: Dict[str, Dict[str, Any]],
    high_velocity: Set[str],
) -> float:
    """
    Enhanced ring risk = avg(member_scores) + (density × 20) + velocity_multiplier.
    velocity_multiplier = +10 if ≥ 50% of members have high_velocity.
    """
    members = ring["members"]
    if not members:
        return ring.get("risk_score", 0)

    member_scores = [scores.get(m, {}).get("score", 0) for m in members]
    avg_score = sum(member_scores) / len(member_scores)

    density_component = density * 20

    velocity_count = sum(1 for m in members if m in high_velocity)
    velocity_multiplier = 10 if velocity_count >= len(members) * 0.5 else 0

    risk = avg_score + density_component + velocity_multiplier
    return round(min(100, risk), 2)


class ProcessingService:
    """Orchestrates the complete money-muling detection pipeline."""

    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the full pipeline on validated transaction data.

        Returns:
            JSON-compatible dict with suspicious_accounts, fraud_rings, summary

        Time Complexity: O(V×E + V×C + n×k + V×3^D)
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 1. Build graph
        G = build_graph(df)
        total_accounts = G.number_of_nodes()

        # 2. Cycle detection
        cycle_rings = detect_cycles(G, df)
        cycle_accounts: set = set()
        for ring in cycle_rings:
            cycle_accounts.update(ring["members"])

        # 3. Smurfing detection
        smurf_rings, aggregators, dispersers = detect_smurfing(df)

        # 4. Shell chain detection
        shell_rings, shell_accounts = detect_shell_chains(G, df)

        # 5. Rapid pass-through detection
        rapid_pt_accounts, _ = detect_rapid_pass_through(df)

        # 6. Activity spike detection
        spike_accounts = detect_activity_spikes(df)

        # 7. Betweenness centrality
        centrality_accounts, _ = compute_centrality(G)

        # 8. Net retention ratio (NEW)
        retention_accounts = detect_low_retention(df)

        # 9. Throughput + balance oscillation (NEW)
        throughput_accounts, oscillation_accounts = detect_flow_metrics(df)

        # 10. Sender diversity burst (NEW)
        diversity_accounts = detect_burst_diversity(df)

        # 11. SCC detection (NEW)
        scc_accounts, scc_rings = detect_scc(G)

        # 12. Cascade depth (NEW)
        cascade_accounts = detect_cascade_depth(G, df)

        # 13. Activity consistency variance
        irregular_accounts = detect_irregular_activity(df)

        # 14. Forwarding latency (median-based)
        forwarding_accounts, _ = detect_rapid_forwarding(df)

        # 15. Dormant activation detection
        dormant_accounts = detect_dormant_activation(df)

        # 16. Structured amount fragmentation
        structuring_accounts = detect_amount_structuring(df)

        # 17. False positive detection
        merchant_accounts, payroll_accounts = detect_false_positives(df)

        # 18. Compute scores (all patterns)
        raw_scores = compute_scores(
            df=df,
            cycle_accounts=cycle_accounts,
            aggregators=aggregators,
            dispersers=dispersers,
            shell_accounts=shell_accounts,
            merchant_accounts=merchant_accounts,
            payroll_accounts=payroll_accounts,
            rapid_pass_through=rapid_pt_accounts,
            activity_spike=spike_accounts,
            high_centrality=centrality_accounts,
            low_retention=retention_accounts,
            high_throughput=throughput_accounts,
            balance_oscillation=oscillation_accounts,
            burst_diversity=diversity_accounts,
            scc_members=scc_accounts,
            cascade_depth=cascade_accounts,
            irregular_activity=irregular_accounts,
            rapid_forwarding=forwarding_accounts,
            dormant_activation=dormant_accounts,
            structured_fragmentation=structuring_accounts,
        )

        # 16. Normalize scores to [0, 100]
        normalized = normalize_scores(raw_scores)

        # 17. Risk propagation (NEW) — mutates normalized in-place
        normalized = propagate_risk(G, normalized)

        # 18. Closeness centrality on suspicious subgraph (NEW)
        suspicious_set = {
            acct for acct, data in normalized.items() if data["score"] > 0
        }
        closeness_accounts, _ = compute_closeness_centrality(G, suspicious_set)

        # 19. Local clustering on suspicious subgraph (NEW)
        clustering_accounts, _ = detect_high_clustering(G, suspicious_set)

        # Add closeness & clustering patterns post-propagation
        for acct in closeness_accounts:
            if acct in normalized:
                normalized[acct]["score"] = min(
                    100, normalized[acct]["score"] + 15
                )
                if "high_closeness_centrality" not in normalized[acct]["patterns"]:
                    normalized[acct]["patterns"].append("high_closeness_centrality")

        for acct in clustering_accounts:
            if acct in normalized:
                normalized[acct]["score"] = min(
                    100, normalized[acct]["score"] + 15
                )
                if "high_local_clustering" not in normalized[acct]["patterns"]:
                    normalized[acct]["patterns"].append("high_local_clustering")

        # 20. Combine all rings
        all_rings = cycle_rings + smurf_rings + shell_rings + scc_rings

        # Collect high_velocity accounts for ring risk enhancement
        high_velocity: Set[str] = set()
        for acct, data in normalized.items():
            if "high_velocity" in data.get("patterns", []):
                high_velocity.add(acct)

        for ring in all_rings:
            density = _compute_ring_density(G, ring)
            ring["density_score"] = density
            ring["risk_score"] = _enhance_ring_risk(
                ring, density, normalized, high_velocity
            )
            if density > RING_DENSITY_THRESHOLD:
                ring["risk_score"] = round(
                    min(100, ring["risk_score"] * (1 + RING_DENSITY_BONUS_PERCENT / 100)),
                    2,
                )

        # 21. Format output
        return format_output(
            scores=normalized,
            all_rings=all_rings,
            total_accounts=total_accounts,
        )
