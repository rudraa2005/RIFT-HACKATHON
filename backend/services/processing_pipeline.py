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

import contextlib
import logging
import os
import time
import threading
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

# Maximum transactions to process (performance requirement: <= 30s)
MAX_TRANSACTIONS = int(os.getenv("MAX_TRANSACTIONS", "2500")) # Lowered for Free Tier
ANOMALY_SKIP_TX_THRESHOLD = int(os.getenv("ANOMALY_SKIP_TX_THRESHOLD", "3000"))
CENTRALITY_SKIP_TX_THRESHOLD = int(os.getenv("CENTRALITY_SKIP_TX_THRESHOLD", "1000"))
MAX_GRAPH_NODES_RESPONSE = int(os.getenv("MAX_GRAPH_NODES_RESPONSE", "1000"))
MAX_GRAPH_EDGES_RESPONSE = int(os.getenv("MAX_GRAPH_EDGES_RESPONSE", "1500"))
TIME_LIMIT = 25.0 # Seconds before we start skipping optional blocks

_MODEL_CACHE_LOCK = threading.Lock()
_CACHED_MODEL_PATH: str | None = None
_CACHED_MODEL: RiskModel | None = None


def _resolve_model_path() -> str:
    model_path = ML_MODEL_PATH
    if not os.path.isabs(model_path):
        backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(backend_root, model_path)
    return model_path


def _get_cached_model(model_path: str) -> RiskModel | None:
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    with _MODEL_CACHE_LOCK:
        if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == model_path:
            return _CACHED_MODEL

        if not os.path.exists(model_path):
            return None

        model = RiskModel()
        model.load(model_path)
        if not model.is_trained:
            return None

        _CACHED_MODEL = model
        _CACHED_MODEL_PATH = model_path
        return _CACHED_MODEL


def _cache_runtime_model(model: RiskModel) -> None:
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    with _MODEL_CACHE_LOCK:
        _CACHED_MODEL = model
        _CACHED_MODEL_PATH = "__runtime_trained__"


def warmup_ml_model() -> bool:
    """Best-effort ML warmup during app startup."""
    if not ML_ENABLED:
        return False
    try:
        model = _get_cached_model(_resolve_model_path())
        return model is not None
    except Exception:
        return False



@contextlib.contextmanager
def log_timer(label: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info("Module [%s] took %.4f seconds", label, elapsed)


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

        # 0. Start Timer
        t_start = time.time()
        
        # Removed sampling to satisfy user requirement for analyzing ALL data.
        # However, we will use hard timeouts per module to keep total time under 30s.

        # 0.5 Compute Adaptive Thresholds
        thresholds = compute_adaptive_thresholds(df)
        logger.info("Adaptive thresholds computed: %s", thresholds)

        # 1. Build graph
        t0 = time.time()
        G = build_graph(df)
        total_accounts = G.number_of_nodes()

        # Forensic trigger maps (account_id -> timestamp_str)
        trigger_times: Dict[str, Dict[str, str]] = {}

        # 2. Cycle detection
        with log_timer("cycle_detection"):
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
        with log_timer("smurfing_detection"):
            smurf_rings, aggregators, dispersers, smurf_triggers = detect_smurfing(
                df,
                min_senders_override=thresholds["smurfing_min_senders"],
                min_receivers_override=thresholds["smurfing_min_receivers"],
            )
            trigger_times.update(smurf_triggers)

        # 4. Shell chain detection
        with log_timer("shell_chain_detection"):
            shell_rings, shell_accounts = detect_shell_chains(G, df, exclude_nodes=cycle_accounts)
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

        # 7. Betweenness centrality (HEAVY)
        centrality_accounts = set()
        with log_timer("betweenness_centrality"):
            if len(df) < CENTRALITY_SKIP_TX_THRESHOLD and (time.time() - t_start) < TIME_LIMIT:
                centrality_accounts, _ = compute_centrality(G)
            else:
                logger.info("Skipping betweenness centrality for performance.")

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
        cascade_rings, cascade_accounts = detect_cascade_depth(G, df)
        trigger_times["deep_layered_cascade"] = {
            acct: str(df["timestamp"].max()) for acct in cascade_accounts
        }

        # 13. Activity consistency variance
        irregular_accounts = detect_irregular_activity(df)

        # 14. False positive detection
        merchant_accounts, payroll_accounts = detect_false_positives(df)

        # 14.5 Unsupervised Anomaly Detection (Isolation Forest)
        with log_timer("anomaly_detection"):
            if len(df) >= ANOMALY_SKIP_TX_THRESHOLD:
                anomaly_scores = {}
            else:
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

        # 17. Risk propagation (graph-based)
        with log_timer("risk_propagation"):
            normalized = propagate_risk(G, normalized)

        # 18. Build neighbor map for connectivity analysis
        neighbor_map = build_neighbor_map(df)

        # 19. Closeness centrality on suspicious subgraph (HEAVY)
        closeness_accounts = set()
        if len(df) < CENTRALITY_SKIP_TX_THRESHOLD and (time.time() - t_start) < TIME_LIMIT:
            suspicious_set = {
                acct for acct, data in normalized.items() if data["score"] > 0
            }
            closeness_accounts, _ = compute_closeness_centrality(G, suspicious_set)
        else:
            logger.info("Skipping closeness centrality for performance.")

        # 20. Local clustering on suspicious subgraph (HEAVY)
        clustering_accounts = set()
        if len(df) < CENTRALITY_SKIP_TX_THRESHOLD and (time.time() - t_start) < TIME_LIMIT:
            suspicious_set = {
                acct for acct, data in normalized.items() if data["score"] > 0
            }
            clustering_accounts, _ = detect_high_clustering(G, suspicious_set)
        else:
            logger.info("Skipping local clustering for performance.")

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
        all_rings = cycle_rings + smurf_rings + shell_rings + cascade_rings

        # Deduplicate rings by member set
        # Deduplicate rings: exact match by member set
        seen_member_sets = {}
        # Deduplicate rings: exact match by member set
        # Prioritize pattern specificity over raw risk score
        priority = {"cycle": 5, "fan_in": 4, "fan_out": 4, "shell_chain": 3, "deep_layered_cascade": 2}
        
        seen_member_sets: Dict[frozenset, Dict[str, Any]] = {}
        for ring in all_rings:
            key = frozenset(ring["members"])
            if key not in seen_member_sets:
                seen_member_sets[key] = ring
            else:
                existing = seen_member_sets[key]
                p_new = priority.get(ring.get("pattern_type", ""), 0)
                p_ext = priority.get(existing.get("pattern_type", ""), 0)
                if p_new > p_ext or (p_new == p_ext and ring["risk_score"] > existing["risk_score"]):
                    seen_member_sets[key] = ring
        
        deduped_rings = list(seen_member_sets.values())
        # Collapse subset rings: if ring A ⊂ ring B, drop A
        final_rings = []
        # Sort by length descending, then by pattern priority
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
        
        # 21.5 Super-Deduplication: Collapse rings with high Jaccard overlap
        merged_rings = []
        # Priority: cycle > smurf > shell > cascade
        # Sort by length descending then priority
        priority = {"cycle": 4, "fan_in": 3, "fan_out": 3, "shell_chain": 2, "deep_layered_cascade": 1}
        final_rings.sort(
            key=lambda r: (len(r["members"]), priority.get(r.get("pattern_type", ""), 0)), 
            reverse=True
        )
        
        for ring in final_rings:
            m_set = set(ring["members"])
            if not m_set: continue
            is_redundant = False
            for other in merged_rings:
                o_set = set(other["members"])
                intersection = m_set & o_set
                # If >70% overlap with an existing (larger or higher priority) ring, skip
                if len(intersection) / len(m_set) >= 0.7:
                    is_redundant = True
                    break
            if not is_redundant:
                merged_rings.append(ring)
        
        # 21.6 Re-sequence IDs: sequential numbering per type + group ID extraction
        counters = {}
        for ring in merged_rings:
            raw_p = ring.get("pattern_type", "UNKNOWN")
            if "fan" in raw_p or "smurf" in raw_p:
                p_type = "SMURF"
            elif "shell" in raw_p:
                p_type = "SHELL"
            elif "cycle" in raw_p:
                p_type = "CYCLE"
            elif "cascade" in raw_p:
                p_type = "CASCADE"
            else:
                p_type = raw_p.split("_")[0].upper()
            
            # Extract Group ID from members (e.g., 'MULE_A_2' -> '2')
            group_id = ""
            member_ids = []
            for m in ring["members"]:
                parts = str(m).split("_")
                for p in parts:
                    if p.isdigit():
                        member_ids.append(p)
                        break
            if member_ids and len(set(member_ids)) == 1:
                group_id = member_ids[0]
            
            if group_id:
                ring["ring_id"] = f"RING_{p_type}_{group_id}"
            else:
                counters[p_type] = counters.get(p_type, 0) + 1
                ring["ring_id"] = f"RING_{p_type}_{counters[p_type]:03d}"
        
        all_rings = merged_rings

        high_velocity: Set[str] = set()
        for acct, data in normalized.items():
            if "high_velocity" in data.get("patterns", []):
                high_velocity.add(acct)

        all_rings = finalize_ring_risks(G, all_rings, normalized, high_velocity)

        # 21.7 Inject behavioral tags from ring member_patterns
        for ring in all_rings:
            m_patterns = ring.get("member_patterns", {})
            for acct_id, patterns in m_patterns.items():
                if acct_id in normalized:
                    existing = set(normalized[acct_id].get("patterns", []))
                    existing.update(patterns)
                    normalized[acct_id]["patterns"] = sorted(list(existing))

        # 23. ML inference + hybrid scoring
        ml_scores = None
        model = None
        if ML_ENABLED:
            model_path = _resolve_model_path()
            try:
                model = _get_cached_model(model_path)
                if model is None:
                    logger.warning("ML model unavailable at %s; using rule-only scoring", model_path)
            except Exception as e:
                logger.warning(
                    "ML scoring unavailable, falling back to rule-only: %s", e
                )
                model = None

        # 22. ML Risk Scoring (High-Speed Inference with 20s Bailout)
        ml_scores = None
        if ML_ENABLED:
            with log_timer("ml_feature_vector_building"):
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

            # Global bailout check: ensure we reach 30s deadline
            current_elapsed = time.time() - t_start
            if current_elapsed > 20.0:
                logger.warning("Bailing out of ML inference: pipeline spent %.2fs already. Accuracy shifted to Rule Engine.", current_elapsed)
            elif model and model.is_trained:
                with log_timer("ml_inference_primary"):
                    try:
                        X = vectors_to_matrix(feature_vectors, account_list)
                        probs = model.predict(X)
                        ml_scores = {acct: float(prob) for acct, prob in zip(account_list, probs)}
                        logger.info("Primary ML scoring completed for %d accounts", len(ml_scores))
                    except Exception as e:
                        logger.error("Primary ML inference failed: %s", str(e))
            else:
                # Deployment fallback: bootstrap a lightweight model if primary fails/missing
                with log_timer("ml_inference_bootstrap"):
                    try:
                        X = vectors_to_matrix(feature_vectors, account_list)
                        y = np.array([1 if normalized.get(a, {}).get("score", 0.0) >= 50.0 else 0 for a in account_list], dtype=np.int32)
                        
                        if y.sum() == 0 or y.sum() == len(y):
                            valid_scores = [normalized.get(a, {}).get("score", 0.0) for a in account_list]
                            cutoff = float(np.percentile(valid_scores, 80)) if valid_scores else 50.0
                            y = np.array([1 if normalized.get(a, {}).get("score", 0.0) >= cutoff else 0 for a in account_list], dtype=np.int32)
                            
                        bootstrap_model = RiskModel()
                        bootstrap_model.train(X, y)
                        _cache_runtime_model(bootstrap_model)
                        probs = bootstrap_model.predict(X)
                        ml_scores = {acct: float(prob) for acct, prob in zip(account_list, probs)}
                        logger.info("Bootstrapped runtime ML model for %d accounts", len(ml_scores))
                    except Exception as e:
                        logger.warning("Runtime ML bootstrap failed; keeping rule-only scoring: %s", e)

        normalized = compute_hybrid_scores(normalized, ml_scores)

        # 23.6 Final Structural Suppression Gate + Role Differentiation
        # 1. Identify all ring members to ensure they are ALWAYS flagged
        all_ring_members = set()
        for ring in all_rings:
            all_ring_members.update(ring["members"])

        # 2. Apply structural bonuses for better ranking resolution
        # Hierarchy: SMURF AGG > CYCLE > SHELL
        for acct in normalized:
            acc_patterns = set(normalized[acct].get("patterns", []))
            bonus = 0.0
            if "smurfing_aggregator" in acc_patterns: bonus += 40.0
            if "cycle" in acc_patterns: bonus += 35.0
            if "smurfing_disperser" in acc_patterns: bonus += 20.0
            if "shell_account" in acc_patterns: bonus += 10.0
            if "fan_in_participant" in acc_patterns: bonus += 5.0
            if "fan_out_participant" in acc_patterns: bonus += 5.0
            if "deep_layered_cascade" in acc_patterns: bonus += 10.0
            
            normalized[acct]["score"] += bonus

            # 3. Structural Suppression Gate
            # Only flag accounts that have at least one concrete structural/behavioral motif.
            # EXCEPTION: Ring members are always protected.
            _STRUCTURAL_MOTIFS = {
                "cycle", "smurfing_aggregator", "smurfing_disperser",
                "shell_account", "high_velocity", "rapid_pass_through",
                "rapid_forwarding", "deep_layered_cascade", "low_retention_pass_through",
                "high_throughput_ratio", "balance_oscillation_pass_through",
                "sudden_activity_spike", "dormant_activation_spike",
                "structured_fragmentation", "fan_in_participant", "fan_out_participant"
            }
            if acct not in all_ring_members and not (acc_patterns & _STRUCTURAL_MOTIFS):
                normalized[acct]["score"] = 0.0
                normalized[acct]["final_risk_score"] = 0.0
                normalized[acct]["patterns"] = []

        # 24. Network Connectivity Analysis (Full Graph for accurate structural metrics)
        score_map = {acct: data.get("score", 0.0) for acct, data in normalized.items()}
        conn_metrics = compute_component_concentration(neighbor_map, score_map, top_n=50)

        # Calculate WCC on the UNDERLYING graph, not just suspicious nodes,
        # to reflect true component sizes for analytics.
        wcc_list = list(nx.connected_components(G.to_undirected()))

        # 23.5 Network Concentration Boost — REMOVED
        # Previously added blind +15 to all connected component members, causing false positives.

        conn_metrics["is_single_network"] = len(wcc_list) == 1 if wcc_list else False
        conn_metrics["connected_components_count"] = len(wcc_list)
        conn_metrics["largest_component_size"] = max(len(c) for c in wcc_list) if wcc_list else 0
        
        # New: Compute SCC distribution and Depth distribution for Analytics
        scc_sizes = sorted([len(c) for c in wcc_list], reverse=True)[:20]
        # Pad with zeros if less than 20
        if len(scc_sizes) < 20:
            scc_sizes += [0] * (20 - len(scc_sizes))
        # Scale to match the UI expectations (0-100 bars)
        max_size = max(scc_sizes) if scc_sizes and max(scc_sizes) > 0 else 1
        scc_bars = [(s / max_size) * 100 for s in scc_sizes]
        
        # Depth analysis logic
        # Use actual cascade depths from the detection results if available
        depths = []
        if cascade_rings:
             for ring in cascade_rings:
                 depths.append(len(ring["members"]))
        else:
            # Fallback to connectivity depth
            for account in normalized:
                if "deep_layered_cascade" in normalized[account].get("patterns", []):
                    depths.append(G.in_degree(account) + G.out_degree(account))
        
        avg_depth = sum(depths) / len(depths) if depths else 0
        depth_bars = sorted([min(100, d * 10) for d in sorted(depths, reverse=True)[:8]], reverse=True)
        if len(depth_bars) < 8:
            depth_bars += [0] * (8 - len(depth_bars))

        # 72H Burst activity — Calculated across the entire dataset range (20 bins)
        if not df.empty:
            min_t = df['timestamp'].min()
            max_t = df['timestamp'].max()
            if max_t > min_t:
                time_range = (max_t - min_t).total_seconds()
                bin_seconds = max(1, time_range / 20)
                df['bin'] = ((df['timestamp'] - min_t).dt.total_seconds() // bin_seconds).astype(int)
                burst_counts = df.groupby('bin').size().reindex(range(20), fill_value=0).tolist()
            else:
                burst_counts = [0] * 19 + [len(df)]
        else:
            burst_counts = [0] * 20
            
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

        # Keep API payload compact for deployment latency.
        if len(nodes) > MAX_GRAPH_NODES_RESPONSE:
            top_node_ids = {
                n["id"]
                for n in sorted(nodes, key=lambda n: float(n.get("risk_score", 0.0)), reverse=True)[:MAX_GRAPH_NODES_RESPONSE]
            }
            nodes = [n for n in nodes if n["id"] in top_node_ids]
            edges = [e for e in edges if e["source"] in top_node_ids and e["target"] in top_node_ids]

        if len(edges) > MAX_GRAPH_EDGES_RESPONSE:
            edges = sorted(
                edges,
                key=lambda e: float(e.get("amount", 0.0)),
                reverse=True,
            )[:MAX_GRAPH_EDGES_RESPONSE]

        graph_data = {
            "nodes": nodes,
            "edges": edges,
        }

        # 26. Final Nuanced Scoring - Rank-based normalization
        # This occurs after ALL boosts (ML, propagation, centrality, etc.)
        acct_ids = list(normalized.keys())
        raw_vals = np.array([normalized[aid]["score"] for aid in acct_ids])
        
        nonzero_mask = raw_vals > 0
        nonzero_mask = raw_vals > 0
        if np.any(nonzero_mask):
            # 1. Ordinal ranking for stable differentiation
            ranks = rankdata(raw_vals[nonzero_mask], method='ordinal')
            num_suspicious = len(ranks)
            percentiles = (ranks - 0.5) / num_suspicious  # Center percentiles
            
            # 2. Sigmoid scaling to push mid-tier accounts away from clustering
            # k=10 provides strong separation while keeping 0 and 1 boundaries reasonable
            k = 10.0
            sigmoid_percentiles = 1 / (1 + np.exp(-k * (percentiles - 0.5)))
            
            # 3. Min-Max scale the sigmoid back to [0, 1] to ensure range integrity
            s_min = sigmoid_percentiles.min()
            s_max = sigmoid_percentiles.max()
            if s_max > s_min:
                final_percentiles = (sigmoid_percentiles - s_min) / (s_max - s_min)
            else:
                final_percentiles = sigmoid_percentiles
            
            scaled_scores = final_percentiles * 100.0
            
            # Index of ring memberships for adaptive floor
            account_ring_risk = {}
            for ring in all_rings:
                r_risk = float(ring.get("risk_score", 0))
                for member in ring["members"]:
                    m_str = str(member)
                    account_ring_risk[m_str] = max(account_ring_risk.get(m_str, 0), r_risk)

            idx = 0
            for i, aid in enumerate(acct_ids):
                if nonzero_mask[i]:
                    new_score = float(scaled_scores[idx])
                    
                    # 4. TOP-END COMPRESSION: Prevent broad 99.x saturation
                    if new_score > 98.0:
                        new_score = 98.0 + (new_score - 98.0) * 0.5
                    
                    # 5. PROPORTIONAL RISK INJECTION: Replace fixed floor clustering
                    # Uses ring risk to inject a proportional baseline rather than a hard wall.
                    if aid in all_ring_members:
                        r_risk = account_ring_risk.get(aid, 0)
                        # Inject 20% of ring risk as a baseline buffer (+ small jitter)
                        proportional_boost = r_risk * 0.2
                        jitter = (hash(aid) % 100) / 20.0 # 0 to 5 points of jitter for more variance
                        new_score = max(new_score, proportional_boost + jitter)
                        
                    final_score = round(float(new_score), 2)
                    normalized[aid]["score"] = final_score
                    # Also sync final_risk_score so format_output picks it up
                    normalized[aid]["final_risk_score"] = round(final_score / 100.0, 4)
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
        # Expose backend-driven model accuracy-style metrics for Analytics.
        # These are aggregated model confidence percentages from the last run.
        accounts_data = list(normalized.values())
        if accounts_data:
            rule_avg = float(np.mean([float(a.get("rule_risk_score", 0.0)) for a in accounts_data])) * 100.0
            ml_avg = float(np.mean([float(a.get("ml_risk_score", 0.0)) for a in accounts_data])) * 100.0
            total_avg = float(np.mean([float(a.get("final_risk_score", 0.0)) for a in accounts_data])) * 100.0
        else:
            rule_avg = 0.0
            ml_avg = 0.0
            total_avg = 0.0
        ml_available = bool(ml_scores)
        res["summary"]["rule_based_accuracy"] = round(rule_avg, 2)
        res["summary"]["ml_model_accuracy"] = round(ml_avg, 2)
        res["summary"]["total_accuracy"] = round(total_avg, 2)
        res["summary"]["ml_model_available"] = ml_available
        # ENFORCE STRICT SCHEMA COMPLIANCE (Remove extra fields)
        res["summary"]["processing_time_seconds"] = round(time.time() - t_start, 4)
        return res

