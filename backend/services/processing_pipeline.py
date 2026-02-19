"""
Processing Pipeline — Full Pipeline Orchestrator.

Coordinates the complete detection pipeline:
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
  17. Risk propagation (graph-based)
  18. Neighbor-based group risk propagation
  19. Closeness centrality
  20. Local clustering coefficient
  21. Ring density + enhanced risk
  22. ML feature vector construction
  23. ML inference + hybrid scoring
  24. Build graph visualization data
  25. Format JSON output

Performance: < 30s for 10K transactions.
Memory: O(V + E) for graph + O(R) for rings.
"""

import logging
import os
import time
from typing import Any, Dict, Set

import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import rankdata

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
from core.risk.adaptive_thresholds import compute_adaptive_thresholds
from core.risk.base_scoring import compute_scores
from core.risk.normalization import normalize_scores
from core.risk.risk_propagation import propagate_risk
from core.risk.network_analysis import (
    build_neighbor_map,
    compute_component_concentration,
)
from core.forwarding_latency import detect_rapid_forwarding
from core.dormancy_analysis import detect_dormant_activation
from core.amount_structuring import detect_amount_structuring
from core.centrality.closeness import compute_closeness_centrality
from core.structural.clustering_analysis import detect_high_clustering
from core.risk.ring_risk import finalize_ring_risks
from core.output.json_formatter import format_output
from core.ml.feature_vector_builder import build_feature_vectors, vectors_to_matrix
from core.ml.ml_model import RiskModel
from core.ml.hybrid_scorer import compute_hybrid_scores
from core.ml.anomaly_detector import detect_anomalies, aggregate_anomaly_scores
from app.config import ML_ENABLED, ML_MODEL_PATH

logger = logging.getLogger(__name__)

# Maximum transactions to process (performance requirement: ≤ 30s)
MAX_TRANSACTIONS = 10_000


class ProcessingService:
    """Orchestrates the complete money-muling detection pipeline."""

    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the full pipeline on validated transaction data.

        Returns:
            JSON-compatible dict with suspicious_accounts, fraud_rings,
            summary, and graph_data.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Cap at MAX_TRANSACTIONS for performance
        if len(df) > MAX_TRANSACTIONS:
            logger.warning(
                "Dataset has %d transactions, sampling down to %d",
                len(df), MAX_TRANSACTIONS,
            )
            df = df.sample(n=MAX_TRANSACTIONS, random_state=42)

        # 0. Compute Adaptive Thresholds
        t_start = time.time()
        thresholds = compute_adaptive_thresholds(df)
        logger.info("Adaptive thresholds computed: %s", thresholds)

        # 1. Build graph
        t0 = time.time()
        G = build_graph(df)
        total_accounts = G.number_of_nodes()

        # Forensic trigger maps (account_id -> timestamp_str)
        trigger_times: Dict[str, Dict[str, str]] = {}

        # 2. Cycle detection
        t_cyc = time.time()
        cycle_rings = detect_cycles(G, df)
        cycle_accounts: set = set()
        cycle_triggers: Dict[str, str] = {}
        for ring in cycle_rings:
            cycle_accounts.update(ring["members"])
            for member in ring["members"]:
                if member not in cycle_triggers:
                    cycle_triggers[member] = str(df["timestamp"].max())
        trigger_times["cycle"] = cycle_triggers

        # 3. Smurfing detection
        t_smurf = time.time()
        smurf_rings, aggregators, dispersers, smurf_triggers = detect_smurfing(
            df,
            min_senders_override=thresholds["smurfing_min_senders"],
            min_receivers_override=thresholds["smurfing_min_receivers"],
        )
        trigger_times.update(smurf_triggers)

        # 4. Shell chain detection
        t_shell = time.time()
        shell_rings, shell_accounts = detect_shell_chains(G, df)
        trigger_times["shell_account"] = {
            acct: str(df["timestamp"].max()) for acct in shell_accounts
        }

        # 5. Rapid pass-through detection
        rapid_pt_accounts, _ = detect_rapid_pass_through(df)
        trigger_times["rapid_pass_through"] = {
            acct: str(df["timestamp"].max()) for acct in rapid_pt_accounts
        }

        # 5.5 Rapid forwarding (forensic)
        forwarding_accounts, _ = detect_rapid_forwarding(df)
        trigger_times["rapid_forwarding"] = {
            acct: str(df["timestamp"].max()) for acct in forwarding_accounts
        }

        # 6. Activity spike detection
        spike_data = detect_activity_spikes(
            df, min_txns_override=thresholds["spike_min_txns"]
        )
        spike_accounts, spike_triggers = spike_data
        trigger_times["sudden_activity_spike"] = spike_triggers

        # 6.5 Dormant activation
        dormant_accounts = detect_dormant_activation(df)
        trigger_times["dormant_activation_spike"] = {
            acct: str(df["timestamp"].max()) for acct in dormant_accounts
        }

        # 6.6 Amount structuring
        structuring_accounts = detect_amount_structuring(df)
        trigger_times["structured_fragmentation"] = {
            acct: str(df["timestamp"].max()) for acct in structuring_accounts
        }

        # 7. Betweenness centrality
        centrality_accounts, _ = compute_centrality(G)

        # 8. Net retention ratio
        retention_accounts = detect_low_retention(df)

        # 9. Throughput + balance oscillation
        throughput_accounts = detect_high_throughput(df)
        oscillation_accounts = detect_balance_oscillation(df)

        # 10. Sender diversity burst
        diversity_accounts, diversity_triggers = detect_burst_diversity(df)
        trigger_times["high_burst_diversity"] = diversity_triggers

        # 11. SCC detection
        scc_accounts, scc_rings = detect_scc(G)
        trigger_times["large_scc_membership"] = {
            acct: str(df["timestamp"].max()) for acct in scc_accounts
        }

        # 12. Cascade depth
        cascade_accounts = detect_cascade_depth(G, df)
        trigger_times["deep_layered_cascade"] = {
            acct: str(df["timestamp"].max()) for acct in cascade_accounts
        }

        # 13. Activity consistency variance
        irregular_accounts = detect_irregular_activity(df)

        # 14. False positive detection
        merchant_accounts, payroll_accounts = detect_false_positives(df)

        # 14.5 Unsupervised Anomaly Detection (Isolation Forest)
        t_ml = time.time()
        txn_anomaly_scores = detect_anomalies(df)
        anomaly_scores = aggregate_anomaly_scores(df, txn_anomaly_scores)

        # 15. Compute scores (all patterns)
        t_score = time.time()
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
            anomaly_scores=anomaly_scores,
            trigger_times=trigger_times,
        )

        # 16. Use raw scores for propagation (Removed normalize_scores to prevent clamping)
        normalized = raw_scores 

        # 17. Risk propagation (graph-based) — mutates normalized in-place
        t_prop = time.time()
        normalized = propagate_risk(G, normalized)

        # 18. Build neighbor map for connectivity analysis
        neighbor_map = build_neighbor_map(df)

        # 19. Closeness centrality on suspicious subgraph
        t_close = time.time()
        suspicious_set = {
            acct for acct, data in normalized.items() if data["score"] > 0
        }
        closeness_accounts, _ = compute_closeness_centrality(G, suspicious_set)

        # 20. Local clustering on suspicious subgraph
        t_clust = time.time()
        clustering_accounts, _ = detect_high_clustering(G, suspicious_set)

        # Add closeness & clustering patterns post-propagation (reduced weight)
        for acct in closeness_accounts:
            if acct in normalized:
                normalized[acct]["score"] = min(
                    100, normalized[acct]["score"] + 5
                )
                if "high_closeness_centrality" not in normalized[acct]["patterns"]:
                    normalized[acct]["patterns"].append("high_closeness_centrality")

        for acct in clustering_accounts:
            if acct in normalized:
                normalized[acct]["score"] = min(
                    100, normalized[acct]["score"] + 5
                )
                if "high_local_clustering" not in normalized[acct]["patterns"]:
                    normalized[acct]["patterns"].append("high_local_clustering")

        # 21. Combine all rings (SCC excluded — it's a supplementary pattern, not a ring)
        all_rings = cycle_rings + smurf_rings + shell_rings

        # Deduplicate rings by member set
        # Deduplicate rings: exact match by member set
        seen_member_sets = {}
        for ring in all_rings:
            key = frozenset(ring["members"])
            if key not in seen_member_sets or ring["risk_score"] > seen_member_sets[key]["risk_score"]:
                seen_member_sets[key] = ring
        
        deduped_rings = list(seen_member_sets.values())
        # Collapse subset rings: if ring A ⊂ ring B, drop A
        # We prioritize cycles > smurfing > shell
        final_rings = []
        # Sort by length descending, then by pattern priority
        priority = {"cycle": 3, "fan_in": 2, "fan_out": 2, "shell_chain": 1}
        sorted_rings = sorted(
            deduped_rings, 
            key=lambda r: (len(r["members"]), priority.get(r.get("pattern_type", ""), 0)), 
            reverse=True
        )

        for ring in sorted_rings:
            ring_set = frozenset(ring["members"])
            is_subset = False
            for other in final_rings:
                if ring_set <= frozenset(other["members"]):
                    is_subset = True
                    break
            if not is_subset:
                final_rings.append(ring)
        
        # 21.5 Re-sequence IDs and ensure they start from 001
        for i, ring in enumerate(final_rings, 1):
            p_type = ring.get("pattern_type", "UNKNOWN").split("_")[0].upper()
            ring["ring_id"] = f"RING_{p_type}_{i:03d}"
        
        all_rings = final_rings

        high_velocity: Set[str] = set()
        for acct, data in normalized.items():
            if "high_velocity" in data.get("patterns", []):
                high_velocity.add(acct)

        all_rings = finalize_ring_risks(G, all_rings, normalized, high_velocity)

        # 22. ML feature vector construction
        all_accounts = set(df["sender_id"].unique()) | set(
            df["receiver_id"].unique()
        )
        feature_vectors, account_list = build_feature_vectors(
            all_accounts=all_accounts,
            cycle_accounts=cycle_accounts,
            aggregators=aggregators,
            dispersers=dispersers,
            shell_accounts=shell_accounts,
            high_velocity=high_velocity,
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
            high_closeness=closeness_accounts,
            high_clustering=clustering_accounts,
            rapid_forwarding=forwarding_accounts,
            dormant_activation=dormant_accounts,
            structured_fragmentation=structuring_accounts,
            G=G,
            df=df,
        )

        # 23. ML inference + hybrid scoring
        ml_scores = None
        if ML_ENABLED:
            # Resolve model path relative to backend root
            model_path = ML_MODEL_PATH
            if not os.path.isabs(model_path):
                backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(backend_root, model_path)
            
            if os.path.exists(model_path):
                try:
                    model = RiskModel()
                    model.load(model_path)
                    if model.is_trained:
                        X = vectors_to_matrix(feature_vectors, account_list)
                        probs = model.predict(X)
                        ml_scores = {
                            acct: float(prob)
                            for acct, prob in zip(account_list, probs)
                        }
                        logger.info("ML scoring completed for %d accounts", len(ml_scores))
                    else:
                        logger.warning("ML model loaded but not trained; using rule-only scoring")
                except Exception as e:
                    logger.warning(
                        "ML scoring unavailable, falling back to rule-only: %s", e
                    )
                    ml_scores = None
            else:
                logger.warning("ML model file not found at %s; using rule-only scoring", model_path)

        normalized = compute_hybrid_scores(normalized, ml_scores)

        # 23.6 Final Structural Suppression Gate
        # Only flag accounts that have at least one concrete structural/behavioral motif.
        # This prevents innocent bystanders from being caught by ML/Propagation noise.
        _STRUCTURAL_MOTIFS = {
            "cycle", "smurfing_aggregator", "smurfing_disperser",
            "shell_account", "high_velocity", "rapid_pass_through",
            "rapid_forwarding", "deep_layered_cascade", "low_retention_pass_through",
            "high_throughput_ratio", "balance_oscillation_pass_through",
            "sudden_activity_spike", "dormant_activation_spike",
            "structured_fragmentation",
        }
        for acct in normalized:
            acc_patterns = set(normalized[acct].get("patterns", []))
            if not (acc_patterns & _STRUCTURAL_MOTIFS):
                normalized[acct]["score"] = 0.0
                normalized[acct]["final_risk_score"] = 0.0
                # Wipe patterns for zero-score accounts to keep output clean
                normalized[acct]["patterns"] = []

        # 24. Network Connectivity Analysis
        score_map = {acct: data.get("score", 0.0) for acct, data in normalized.items()}
        conn_metrics = compute_component_concentration(neighbor_map, score_map, top_n=50)

        suspicious_nodes = [
            node for node, data in normalized.items()
            if data.get("score", 0.0) >= 1.0
        ]
        susp_G = G.subgraph(suspicious_nodes).to_undirected()
        wcc_list = list(nx.connected_components(susp_G))

        # 23.5 Network Concentration Boost — REMOVED
        # Previously added blind +15 to all connected component members, causing false positives.

        conn_metrics["is_single_network"] = len(wcc_list) == 1 if wcc_list else False
        conn_metrics["connected_components_count"] = len(wcc_list)
        conn_metrics["largest_component_size"] = max(len(c) for c in wcc_list) if wcc_list else 0
        
        # New: Compute SCC distribution and Depth distribution for Analytics
        import numpy as np
        scc_sizes = sorted([len(c) for c in wcc_list], reverse=True)[:20]
        # Pad with zeros if less than 20
        if len(scc_sizes) < 20:
            scc_sizes += [0] * (20 - len(scc_sizes))
        # Scale to match the UI expectations (0-100 bars)
        max_size = max(scc_sizes) if scc_sizes and max(scc_sizes) > 0 else 1
        scc_bars = [(s / max_size) * 100 for s in scc_sizes]
        
        # Depth analysis logic
        depths = []
        for account in normalized:
            if "deep_layered_cascade" in normalized[account].get("patterns", []):
                # Sample depth or use degree as proxy if actual depth per account isn't stored
                depths.append(G.in_degree(account) + G.out_degree(account))
        
        avg_depth = sum(depths) / len(depths) if depths else 0
        depth_bars = sorted([min(100, d * 10) for d in depths], reverse=True)[:8]
        if len(depth_bars) < 8:
            depth_bars += [10] * (8 - len(depth_bars))

        # 72H Burst activity (simulated from timestamps)
        df['hour_bucket'] = df['timestamp'].dt.floor('h')
        burst_counts = df.groupby('hour_bucket').size().tail(20).tolist()
        if len(burst_counts) < 20:
            burst_counts = [0] * (20 - len(burst_counts)) + burst_counts
        max_burst = max(burst_counts) if burst_counts and max(burst_counts) > 0 else 1
        burst_series = [(b / max_burst) * 100 for b in burst_counts]

        conn_metrics["scc_distribution"] = scc_bars
        conn_metrics["avg_cascade_depth"] = round(avg_depth, 2)
        conn_metrics["depth_distribution"] = depth_bars
        conn_metrics["burst_activity"] = burst_series

        # 25. Build Graph Data for Visualization
        nodes = []
        for acct in G.nodes():
            score_data = normalized.get(acct, {"score": 0.0, "patterns": []})
            is_suspicious = score_data.get("score", 0.0) >= 50.0

            display_patterns = [
                p for p in score_data.get("patterns", [])
                if p not in ["multi_pattern", "nonlinear_amplifier"]
            ]
            primary_pattern = display_patterns[0] if display_patterns else "None"

            nodes.append({
                "id": str(acct),
                "label": str(acct),
                "risk_score": float(round(score_data.get("score", 0.0), 2)),
                "flagged": "Yes" if is_suspicious else "No",
                "pattern_type": primary_pattern,
                "is_suspicious": bool(is_suspicious),
            })

        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):
            edges.append({
                "source": str(u),
                "target": str(v),
                "amount": float(data.get("amount", 0.0)),
                "timestamp": str(data.get("timestamp", "")),
                "transaction_id": str(data.get("transaction_id", "")),
            })

        graph_data = {
            "nodes": nodes,
            "edges": edges,
        }

        # 26. Final Nuanced Scoring - Rank-based normalization
        # This occurs after ALL boosts (ML, propagation, centrality, etc.)
        acct_ids = list(normalized.keys())
        raw_vals = np.array([normalized[aid]["score"] for aid in acct_ids])
        
        nonzero_mask = raw_vals > 0
        if np.any(nonzero_mask):
            # Using 'ordinal' method breaks ties to force a distribution
            # instead of bunching same-score accounts at the same rank.
            ranks = rankdata(raw_vals[nonzero_mask], method='ordinal')
            num_suspicious = len(ranks)
            percentiles = ranks / num_suspicious
            
            # Non-linear spread (x^2.8)
            # This ensures only the top ~10% get scores > 80
            final_percentiles = np.power(percentiles, 2.8) 
            scaled_scores = final_percentiles * 100.0
            
            idx = 0
            for i, aid in enumerate(acct_ids):
                if nonzero_mask[i]:
                    new_score = round(float(scaled_scores[idx]), 2)
                    normalized[aid]["score"] = new_score
                    # Also sync final_risk_score so format_output picks it up
                    normalized[aid]["final_risk_score"] = round(new_score / 100.0, 4)
                    idx += 1
                else:
                    normalized[aid]["score"] = 0.0
                    normalized[aid]["final_risk_score"] = 0.0

        # 27. Format output
        res = format_output(
            scores=normalized,
            all_rings=all_rings,
            total_accounts=total_accounts,
            graph_data=graph_data,
        )
        res["summary"]["network_connectivity"] = conn_metrics
        res["summary"]["processing_time_seconds"] = time.time() - t_start
        return res
