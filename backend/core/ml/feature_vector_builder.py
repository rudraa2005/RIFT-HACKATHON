"""
Feature Vector Builder (v4) — schema-compliant behavioral engineering.

Strictly uses 5 columns: transaction_id, sender_id, receiver_id, amount, timestamp.

Feature order (32 features):
  0–19  Binary pattern flags
  20    in_degree_ratio         float [0, 1]
  21    out_degree_ratio        float [0, 1]
  22    tx_count                log-scaled
  23    mean_amount             log-scaled
  24    amount_std              log-scaled
  25    unique_counterparties   log-scaled
  26    temporal_span_hours     log-scaled
  27    is_round_amount         binary
  28    is_night_transaction    binary
  29    is_high_amount_outlier  binary
  30    pagerank                continuous
  31    local_clustering        continuous
"""

from typing import Any, Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx

FEATURE_NAMES: List[str] = [
    # Binary pattern flags (0–19)
    "cycle", "smurfing_aggregator", "smurfing_disperser", "shell_account",
    "high_velocity", "rapid_pass_through", "sudden_activity_spike",
    "high_betweenness_centrality", "low_retention_pass_through",
    "high_throughput_ratio", "balance_oscillation_pass_through",
    "high_burst_diversity", "large_scc_membership", "deep_layered_cascade",
    "irregular_activity_spike", "high_closeness_centrality",
    "high_local_clustering", "rapid_forwarding", "dormant_activation_spike",
    "structured_fragmentation",
    # Behavioral Features (20–26)
    "in_degree_ratio", "out_degree_ratio", "tx_count", "mean_amount",
    "amount_std", "unique_counterparties", "temporal_span_hours",
    # Schema-Signals (27–29)
    "is_round_amount", "is_night_transaction", "is_high_amount_outlier",
    # Advanced Structural (30-31)
    "pagerank", "local_clustering",
    # New Behavioral Extensions (32-37)
    "tx_per_hour", "max_amount", "min_amount", "total_volume", "in_out_flow_ratio", "avg_time_between_tx"
]

NUM_BINARY_FEATURES: int = 20
NUM_FEATURES: int = len(FEATURE_NAMES)

def _safe_log1p(x: float) -> float:
    return float(np.log1p(max(0.0, float(x))))

def _compute_behavioral_features(
    account: str,
    G: Optional[nx.MultiDiGraph],
    df: Optional[pd.DataFrame],
) -> List[float]:
    res = [0.0] * 13 # 7 + 6 new ones
    if G is None or not G.has_node(account):
        return res

    in_deg = G.in_degree(account)
    out_deg = G.out_degree(account)
    total_deg = in_deg + out_deg
    if total_deg > 0:
        res[0] = in_deg / total_deg
        res[1] = out_deg / total_deg

    if df is not None:
        mask = (df["sender_id"] == account) | (df["receiver_id"] == account)
        acct_txns = df[mask].copy()
        if not acct_txns.empty:
            amounts = acct_txns["amount"].astype(float).values
            res[2] = _safe_log1p(len(amounts))
            res[3] = _safe_log1p(np.mean(amounts))
            res[4] = _safe_log1p(np.std(amounts))
            counterparties = set(acct_txns["sender_id"]) | set(acct_txns["receiver_id"])
            res[5] = _safe_log1p(len(counterparties) - 1)
            
            ts = pd.to_datetime(acct_txns["timestamp"])
            span = (ts.max() - ts.min()).total_seconds() / 3600
            res[6] = _safe_log1p(span)
            
            # New Behavioral Extensions
            res[7] = (len(amounts) / max(1.0, span)) # tx_per_hour
            res[8] = _safe_log1p(np.max(amounts)) # max_amount
            res[9] = _safe_log1p(np.min(amounts)) # min_amount
            res[10] = _safe_log1p(np.sum(amounts)) # total_volume
            
            in_vol = acct_txns[acct_txns["receiver_id"] == account]["amount"].sum()
            out_vol = acct_txns[acct_txns["sender_id"] == account]["amount"].sum()
            res[11] = in_vol / max(1.0, out_vol) # in_out_flow_ratio
            
            if len(amounts) > 1:
                intervals = ts.sort_values().diff().dt.total_seconds().dropna() / 3600
                res[12] = _safe_log1p(intervals.mean()) # avg_time_between_tx
                
    return res

def build_feature_vectors(
    all_accounts: Set[str],
    cycle_accounts: Set[str],
    aggregators: Set[str],
    dispersers: Set[str],
    shell_accounts: Set[str],
    high_velocity: Set[str],
    rapid_pass_through: Set[str],
    activity_spike: Set[str],
    high_centrality: Set[str],
    low_retention: Set[str],
    high_throughput: Set[str],
    balance_oscillation: Set[str],
    burst_diversity: Set[str],
    scc_members: Set[str],
    cascade_depth: Set[str],
    irregular_activity: Set[str],
    high_closeness: Set[str],
    high_clustering: Set[str],
    rapid_forwarding: Set[str],
    dormant_activation: Set[str],
    structured_fragmentation: Set[str],
    G: Optional[nx.MultiDiGraph] = None,
    df: Optional[pd.DataFrame] = None,
    schema_signals: Optional[Dict[str, Dict[str, float]]] = None,
    structural_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    pattern_sets = [
        cycle_accounts, aggregators, dispersers, shell_accounts, high_velocity,
        rapid_pass_through, activity_spike, high_centrality, low_retention,
        high_throughput, balance_oscillation, burst_diversity, scc_members,
        cascade_depth, irregular_activity, high_closeness, high_clustering,
        rapid_forwarding, dormant_activation, structured_fragmentation
    ]
    
    account_list = sorted(all_accounts)
    vectors: Dict[str, np.ndarray] = {}

    for account in account_list:
        vec = np.zeros(NUM_FEATURES, dtype=np.float32)
        for i, pset in enumerate(pattern_sets):
            if account in pset: vec[i] = 1.0

        behavioral = _compute_behavioral_features(account, G, df)
        for i, val in enumerate(behavioral):
            vec[20 + i] = val

        if schema_signals and account in schema_signals:
            s = schema_signals[account]
            vec[27] = s.get("is_round_amount", 0.0)
            vec[28] = s.get("is_night_transaction", 0.0)
            vec[29] = s.get("is_high_amount_outlier", 0.0)

        if structural_scores and account in structural_scores:
            st = structural_scores[account]
            vec[30] = st.get("pagerank", 0.0)
            vec[31] = st.get("local_clustering", 0.0)

        vectors[account] = vec

    return vectors, account_list

def vectors_to_matrix(vectors: Dict[str, np.ndarray], account_list: List[str]) -> np.ndarray:
    return np.vstack([vectors[acct] for acct in account_list])
