"""
Feature Vector Builder (v3) — No-Leakage Feature Engineering.

Converts pattern sets + transaction data into feature vectors for ML.

Changes from v2:
  - DROPPED: `pattern_count` (was a direct label proxy = sum of other features)
  - ADDED: 7 continuous graph/behavioral features that the rule engine
    does NOT threshold on, breaking circular learning:
      - in_degree_ratio, out_degree_ratio
      - tx_count, mean_amount, amount_std
      - unique_counterparties, temporal_span_hours

Feature order (27 features):
  0–19  Binary pattern flags (same as v2)
  20    in_degree_ratio         float [0, 1]
  21    out_degree_ratio        float [0, 1]
  22    tx_count                int   (log-scaled)
  23    mean_amount             float (log-scaled)
  24    amount_std              float (log-scaled)
  25    unique_counterparties   int   (log-scaled)
  26    temporal_span_hours     float (log-scaled)

Time Complexity: O(V × P + E)
Memory: O(V × P)
"""

from typing import Any, Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

# Canonical feature names in fixed order — must NOT be reordered.
FEATURE_NAMES: List[str] = [
    # Binary pattern flags (0–19)
    "cycle",
    "smurfing_aggregator",
    "smurfing_disperser",
    "shell_account",
    "high_velocity",
    "rapid_pass_through",
    "sudden_activity_spike",
    "high_betweenness_centrality",
    "low_retention_pass_through",
    "high_throughput_ratio",
    "balance_oscillation_pass_through",
    "high_burst_diversity",
    "large_scc_membership",
    "deep_layered_cascade",
    "irregular_activity_spike",
    "high_closeness_centrality",
    "high_local_clustering",
    "rapid_forwarding",
    "dormant_activation_spike",
    "structured_fragmentation",
    # Continuous graph features (20–21) - Purely structural
    "in_degree_ratio",
    "out_degree_ratio",
]

NUM_BINARY_FEATURES: int = 20
NUM_FEATURES: int = len(FEATURE_NAMES)


def _safe_log1p(x: float) -> float:
    """Log-scale a non-negative value: log(1 + x)."""
    return float(np.log1p(max(0.0, x)))


def _compute_continuous_features(
    account: str,
    G: Optional[nx.MultiDiGraph],
    df: Optional[pd.DataFrame],
) -> List[float]:
    """
    Compute 2 purely structural continuous features for an account.

    Returns [in_degree_ratio, out_degree_ratio]
    """
    if G is None or not G.has_node(account):
        return [0.0, 0.0]

    # Degree ratios - Purely structural
    in_deg = G.in_degree(account)
    out_deg = G.out_degree(account)
    total_deg = in_deg + out_deg
    in_ratio = in_deg / total_deg if total_deg > 0 else 0.0
    out_ratio = out_deg / total_deg if total_deg > 0 else 0.0

    return [in_ratio, out_ratio]


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
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Build a fixed-order feature vector for every account.

    Args:
        all_accounts: Set of all account IDs.
        *_accounts / *: Pattern membership sets.
        G: Transaction graph (optional, for continuous features).
        df: Transaction DataFrame (optional, for continuous features).

    Returns:
        Tuple of:
            - dict mapping account_id → numpy array of shape (NUM_FEATURES,)
            - list of account IDs in sorted order
    """
    # Ordered list of pattern sets — MUST match FEATURE_NAMES[:20]
    pattern_sets: List[Set[str]] = [
        cycle_accounts,
        aggregators,
        dispersers,
        shell_accounts,
        high_velocity,
        rapid_pass_through,
        activity_spike,
        high_centrality,
        low_retention,
        high_throughput,
        balance_oscillation,
        burst_diversity,
        scc_members,
        cascade_depth,
        irregular_activity,
        high_closeness,
        high_clustering,
        rapid_forwarding,
        dormant_activation,
        structured_fragmentation,
    ]

    account_list = sorted(all_accounts)
    vectors: Dict[str, np.ndarray] = {}

    for account in account_list:
        vec = np.zeros(NUM_FEATURES, dtype=np.float32)

        # Binary pattern flags (0–19)
        for i, pset in enumerate(pattern_sets):
            if account in pset:
                vec[i] = 1.0

        # Continuous features (20–26)
        continuous = _compute_continuous_features(account, G, df)
        for i, val in enumerate(continuous):
            vec[NUM_BINARY_FEATURES + i] = val

        vectors[account] = vec

    return vectors, account_list


def vectors_to_matrix(
    vectors: Dict[str, np.ndarray],
    account_list: List[str],
) -> np.ndarray:
    """
    Stack individual feature vectors into a 2-D numpy matrix.

    Args:
        vectors: dict mapping account_id → feature vector
        account_list: ordered list of account IDs

    Returns:
        np.ndarray of shape (len(account_list), NUM_FEATURES)
    """
    return np.vstack([vectors[acct] for acct in account_list])
