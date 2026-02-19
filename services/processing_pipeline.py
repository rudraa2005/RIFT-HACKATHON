"""
Processing Pipeline — Full Pipeline Orchestrator.

Coordinates the complete detection pipeline (Phases 1–3):
   1. Parse timestamps & build directed multigraph
   2. Cycle detection
   3. Smurfing detection (fan-in / fan-out)
   4. Shell chain detection
   5. Rapid pass-through (holding time)
   6. Activity spike detection
   7. Betweenness centrality
   8. Net retention ratio
   9. Throughput ratio + balance oscillation
  10. Sender diversity burst
  11. SCC detection
  12. Cascade depth
  13. Activity consistency variance
  14. False positive detection
  15. Compute suspicion scores
  16. Normalize scores to [0, 100]
  17. Risk propagation
  18. Closeness centrality
  19. Local clustering coefficient
  20. Ring density + enhanced risk
  21. Format JSON output

Performance: < 30s for 10K transactions.
Memory: O(V + E) for graph + O(R) for rings.
"""

from typing import Any, Dict, Set

import pandas as pd

from core.graph.graph_builder import build_graph
from core.structural.cycle_detection import detect_cycles
from core.ring_detection.smurfing import detect_smurfing
from core.structural.shell_detection import detect_shell_chains
from core.temporal.forwarding_latency import detect_rapid_pass_through
from core.temporal.burst_detection import detect_activity_spikes
from core.centrality.betweenness import compute_centrality
from core.flow.retention_analysis import detect_low_retention
from core.flow.throughput_analysis import detect_high_throughput
from core.flow.balance_oscillation import detect_balance_oscillation
from core.ring_detection.diversity_analysis import detect_burst_diversity
from core.structural.scc_analysis import detect_scc
from core.structural.cascade_depth import detect_cascade_depth
from core.temporal.activity_consistency import detect_irregular_activity
from core.risk.false_positive_filter import detect_false_positives
from core.risk.base_scoring import compute_scores
from core.risk.normalization import normalize_scores
from core.risk.risk_propagation import propagate_risk
from core.centrality.closeness import compute_closeness_centrality
from core.structural.clustering_analysis import detect_high_clustering
from core.risk.ring_risk import finalize_ring_risks
from core.output.json_formatter import format_output


class ProcessingService:
    """Orchestrates the complete money-muling detection pipeline."""

    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the full pipeline on validated transaction data.

        Returns:
            JSON-compatible dict with suspicious_accounts, fraud_rings, summary
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

        # 8. Net retention ratio
        retention_accounts = detect_low_retention(df)

        # 9. Throughput + balance oscillation
        throughput_accounts = detect_high_throughput(df)
        oscillation_accounts = detect_balance_oscillation(df)

        # 10. Sender diversity burst
        diversity_accounts = detect_burst_diversity(df)

        # 11. SCC detection
        scc_accounts, scc_rings = detect_scc(G)

        # 12. Cascade depth
        cascade_accounts = detect_cascade_depth(G, df)

        # 13. Activity consistency variance
        irregular_accounts = detect_irregular_activity(df)

        # 14. False positive detection
        merchant_accounts, payroll_accounts = detect_false_positives(df)

        # 15. Compute scores (all patterns)
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
        )

        # 16. Normalize scores to [0, 100]
        normalized = normalize_scores(raw_scores)

        # 17. Risk propagation — mutates normalized in-place
        normalized = propagate_risk(G, normalized)

        # 18. Closeness centrality on suspicious subgraph
        suspicious_set = {
            acct for acct, data in normalized.items() if data["score"] > 0
        }
        closeness_accounts, _ = compute_closeness_centrality(G, suspicious_set)

        # 19. Local clustering on suspicious subgraph
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

        # 20. Combine all rings and compute ring risk
        all_rings = cycle_rings + smurf_rings + shell_rings + scc_rings

        high_velocity: Set[str] = set()
        for acct, data in normalized.items():
            if "high_velocity" in data.get("patterns", []):
                high_velocity.add(acct)

        all_rings = finalize_ring_risks(G, all_rings, normalized, high_velocity)

        # 21. Format output
        return format_output(
            scores=normalized,
            all_rings=all_rings,
            total_accounts=total_accounts,
        )
