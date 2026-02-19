"""
Suspicion Scoring Engine.

Computes a per-account suspicion score based on detected patterns.
No single metric exceeds 30 points. Multi-pattern stacking allowed.
False positive dampeners applied after all scoring.

Score caps per pattern (max 30 each):
  +30  cycle member (capped from 40)
  +25  smurfing aggregator
  +25  smurfing disperser
  +30  shell account (capped from 35)
  +20  high velocity
  +25  rapid pass-through
  +20  sudden activity spike
  +20  high betweenness centrality
  +25  low retention pass-through
  +20  high throughput ratio
  +20  balance oscillation
  +20  high burst diversity
  +20  large SCC membership
  +25  deep layered cascade
  +20  irregular activity spike
  +15  high closeness centrality
  +15  high local clustering
  +20  rapid forwarding
  +20  dormant activation spike
  +10  structured fragmentation
  +15  multi-pattern bonus (≥2 patterns)
  nonlinear amplifier for ≥3 patterns
  -40  merchant-like (false positive)
  -30  payroll-like  (false positive)

Final score is later normalized to [0, 100].

Time Complexity: O(V) where V = number of accounts
Memory: O(V)
"""

from typing import Any, Dict, Set

import numpy as np
import pandas as pd

from app.config import (
    HIGH_VELOCITY_THRESHOLD,
    NONLINEAR_BASE_BONUS,
    NONLINEAR_MIN_PATTERNS,
    NONLINEAR_PER_PATTERN,
    SCORE_ACTIVITY_SPIKE,
    SCORE_BALANCE_OSCILLATION,
    SCORE_BURST_DIVERSITY,
    SCORE_CYCLE,
    SCORE_DEEP_CASCADE,
    SCORE_DORMANT_ACTIVATION,
    SCORE_HIGH_CENTRALITY,
    SCORE_HIGH_CLOSENESS,
    SCORE_HIGH_CLUSTERING,
    SCORE_HIGH_THROUGHPUT,
    SCORE_HIGH_VELOCITY,
    SCORE_IRREGULAR_ACTIVITY,
    SCORE_LARGE_SCC,
    SCORE_LOW_RETENTION,
    SCORE_MERCHANT_PENALTY,
    SCORE_MULTI_PATTERN_BONUS,
    SCORE_PAYROLL_PENALTY,
    SCORE_RAPID_FORWARDING,
    SCORE_RAPID_PASS_THROUGH,
    SCORE_SHELL,
    SCORE_SMURFING_AGGREGATOR,
    SCORE_SMURFING_DISPERSER,
    SCORE_STRUCTURED_FRAGMENTATION,
)

# Cap: no single metric exceeds 30
_CAP = 30


def _capped(raw: float) -> float:
    return min(_CAP, raw)


def _compute_high_velocity_accounts(df: pd.DataFrame) -> Set[str]:
    """
    Identify accounts with normalized velocity > threshold.
    Time Complexity: O(n)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
    velocities: Dict[str, float] = {}

    for account in all_accounts:
        acct_txns = df[(df["sender_id"] == account) | (df["receiver_id"] == account)]
        if len(acct_txns) < 2:
            continue
        span_h = (
            acct_txns["timestamp"].max() - acct_txns["timestamp"].min()
        ).total_seconds() / 3600
        if span_h == 0:
            span_h = 1
        velocities[account] = acct_txns["amount"].sum() / span_h

    if not velocities:
        return set()

    global_avg = float(np.mean(list(velocities.values())))
    if global_avg == 0:
        return set()

    return {a for a, v in velocities.items() if v / global_avg > HIGH_VELOCITY_THRESHOLD}


def compute_scores(
    df: pd.DataFrame,
    cycle_accounts: Set[str],
    aggregators: Set[str],
    dispersers: Set[str],
    shell_accounts: Set[str],
    merchant_accounts: Set[str],
    payroll_accounts: Set[str],
    rapid_pass_through: Set[str] | None = None,
    activity_spike: Set[str] | None = None,
    high_centrality: Set[str] | None = None,
    low_retention: Set[str] | None = None,
    high_throughput: Set[str] | None = None,
    balance_oscillation: Set[str] | None = None,
    burst_diversity: Set[str] | None = None,
    scc_members: Set[str] | None = None,
    cascade_depth: Set[str] | None = None,
    irregular_activity: Set[str] | None = None,
    high_closeness: Set[str] | None = None,
    high_clustering: Set[str] | None = None,
    rapid_forwarding: Set[str] | None = None,
    dormant_activation: Set[str] | None = None,
    structured_fragmentation: Set[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute raw suspicion scores for every account in the dataset.

    Returns:
        {account_id: {"score": float, "patterns": [str, ...]}}

    Time Complexity: O(V)
    """
    rapid_pass_through = rapid_pass_through or set()
    activity_spike = activity_spike or set()
    high_centrality = high_centrality or set()
    low_retention = low_retention or set()
    high_throughput = high_throughput or set()
    balance_oscillation = balance_oscillation or set()
    burst_diversity = burst_diversity or set()
    scc_members = scc_members or set()
    cascade_depth = cascade_depth or set()
    irregular_activity = irregular_activity or set()
    high_closeness = high_closeness or set()
    high_clustering = high_clustering or set()
    rapid_forwarding = rapid_forwarding or set()
    dormant_activation = dormant_activation or set()
    structured_fragmentation = structured_fragmentation or set()

    high_velocity = _compute_high_velocity_accounts(df)
    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
    scores: Dict[str, Dict[str, Any]] = {}

    for account in all_accounts:
        score = 0.0
        patterns: list[str] = []
        pattern_count = 0

        # --- Original patterns (capped at 30) ---
        if account in cycle_accounts:
            score += _capped(SCORE_CYCLE)
            patterns.append("cycle")
            pattern_count += 1

        if account in aggregators:
            score += _capped(SCORE_SMURFING_AGGREGATOR)
            patterns.append("smurfing_aggregator")
            pattern_count += 1

        if account in dispersers:
            score += _capped(SCORE_SMURFING_DISPERSER)
            patterns.append("smurfing_disperser")
            pattern_count += 1

        if account in shell_accounts:
            score += _capped(SCORE_SHELL)
            patterns.append("shell_account")
            pattern_count += 1

        if account in high_velocity:
            score += _capped(SCORE_HIGH_VELOCITY)
            patterns.append("high_velocity")
            pattern_count += 1

        # --- Phase 2 patterns ---
        if account in rapid_pass_through:
            score += _capped(SCORE_RAPID_PASS_THROUGH)
            patterns.append("rapid_pass_through")
            pattern_count += 1

        if account in activity_spike:
            score += _capped(SCORE_ACTIVITY_SPIKE)
            patterns.append("sudden_activity_spike")
            pattern_count += 1

        if account in high_centrality:
            score += _capped(SCORE_HIGH_CENTRALITY)
            patterns.append("high_betweenness_centrality")
            pattern_count += 1

        # --- Phase 3 patterns ---
        if account in low_retention:
            score += _capped(SCORE_LOW_RETENTION)
            patterns.append("low_retention_pass_through")
            pattern_count += 1

        if account in high_throughput:
            score += _capped(SCORE_HIGH_THROUGHPUT)
            patterns.append("high_throughput_ratio")
            pattern_count += 1

        if account in balance_oscillation:
            score += _capped(SCORE_BALANCE_OSCILLATION)
            patterns.append("balance_oscillation_pass_through")
            pattern_count += 1

        if account in burst_diversity:
            score += _capped(SCORE_BURST_DIVERSITY)
            patterns.append("high_burst_diversity")
            pattern_count += 1

        if account in scc_members:
            score += _capped(SCORE_LARGE_SCC)
            patterns.append("large_scc_membership")
            pattern_count += 1

        if account in cascade_depth:
            score += _capped(SCORE_DEEP_CASCADE)
            patterns.append("deep_layered_cascade")
            pattern_count += 1

        if account in irregular_activity:
            score += _capped(SCORE_IRREGULAR_ACTIVITY)
            patterns.append("irregular_activity_spike")
            pattern_count += 1

        if account in high_closeness:
            score += _capped(SCORE_HIGH_CLOSENESS)
            patterns.append("high_closeness_centrality")
            pattern_count += 1

        if account in high_clustering:
            score += _capped(SCORE_HIGH_CLUSTERING)
            patterns.append("high_local_clustering")
            pattern_count += 1

        # --- New Phase 4 patterns ---
        if account in rapid_forwarding:
            score += _capped(SCORE_RAPID_FORWARDING)
            patterns.append("rapid_forwarding")
            pattern_count += 1

        if account in dormant_activation:
            score += _capped(SCORE_DORMANT_ACTIVATION)
            patterns.append("dormant_activation_spike")
            pattern_count += 1

        if account in structured_fragmentation:
            score += _capped(SCORE_STRUCTURED_FRAGMENTATION)
            patterns.append("structured_fragmentation")
            pattern_count += 1

        # Multi-pattern bonus (≥2)
        if pattern_count >= 2:
            score += SCORE_MULTI_PATTERN_BONUS
            patterns.append("multi_pattern")

        # --- False positive dampeners (applied after all scoring) ---
        if account in merchant_accounts:
            score += SCORE_MERCHANT_PENALTY
            patterns.append("merchant_like")

        if account in payroll_accounts:
            score += SCORE_PAYROLL_PENALTY
            patterns.append("payroll_like")

        # Nonlinear amplifier for ≥3 real patterns
        if pattern_count >= NONLINEAR_MIN_PATTERNS:
            nonlinear_bonus = NONLINEAR_BASE_BONUS + NONLINEAR_PER_PATTERN * (
                pattern_count - NONLINEAR_MIN_PATTERNS
            )
            score += nonlinear_bonus
            patterns.append("nonlinear_amplifier")

        scores[account] = {"score": score, "patterns": patterns}

    return scores
