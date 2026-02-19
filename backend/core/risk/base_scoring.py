"""
Suspicion Scoring Engine (Base Scoring).

Computes per-account suspicion score from detected patterns.
No single metric exceeds 30 points. Multi-pattern stacking allowed.
False positive dampeners applied after all scoring.

Time Complexity: O(V)
Memory: O(V)
"""

from typing import Any, Dict, Set

import pandas as pd

from app.config import (
    NONLINEAR_BASE_BONUS,
    NONLINEAR_MIN_PATTERNS,
    NONLINEAR_PER_PATTERN,
    SCORE_ACTIVITY_SPIKE,
    SCORE_BALANCE_OSCILLATION,
    SCORE_BURST_DIVERSITY,
    SCORE_CYCLE,
    SCORE_DEEP_CASCADE,
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
    SCORE_RAPID_PASS_THROUGH,
    SCORE_SHELL,
    SCORE_SMURFING_AGGREGATOR,
    SCORE_SMURFING_DISPERSER,
)
from core.flow.velocity_analysis import compute_high_velocity_accounts

# Cap: allow single patterns to reach 100 if configured
_CAP = 100


def _capped(raw: float) -> float:
    return min(_CAP, raw)


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
    # Forensic timing maps (account_id -> timestamp_str)
    trigger_times: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute raw suspicion scores for every account in the dataset.

    Returns:
        {account_id: {
            "score": float, 
            "patterns": [str, ...],
            "breakdown": {pattern: score},
            "timeline": [{"time": str, "event": str}]
        }}
    """
    from app.config import (
        SCORE_RAPID_FORWARDING,
        SCORE_DORMANT_ACTIVATION,
        SCORE_STRUCTURED_FRAGMENTATION,
    )

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
    trigger_times = trigger_times or {}

    high_velocity, velocity_triggers = compute_high_velocity_accounts(df)
    
    # Merge velocity triggers into the main trigger map
    if "high_velocity" not in trigger_times:
        trigger_times["high_velocity"] = velocity_triggers

    all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
    scores: Dict[str, Dict[str, Any]] = {}

    pattern_scores = {
        "cycle": _capped(SCORE_CYCLE),
        "smurfing_aggregator": _capped(SCORE_SMURFING_AGGREGATOR),
        "smurfing_disperser": _capped(SCORE_SMURFING_DISPERSER),
        "shell_account": _capped(SCORE_SHELL),
        "high_velocity": _capped(SCORE_HIGH_VELOCITY),
        "rapid_pass_through": _capped(SCORE_RAPID_PASS_THROUGH),
        "sudden_activity_spike": _capped(SCORE_ACTIVITY_SPIKE),
        "high_betweenness_centrality": _capped(SCORE_HIGH_CENTRALITY),
        "low_retention_pass_through": _capped(SCORE_LOW_RETENTION),
        "high_throughput_ratio": _capped(SCORE_HIGH_THROUGHPUT),
        "balance_oscillation_pass_through": _capped(SCORE_BALANCE_OSCILLATION),
        "high_burst_diversity": _capped(SCORE_BURST_DIVERSITY),
        "large_scc_membership": _capped(SCORE_LARGE_SCC),
        "deep_layered_cascade": _capped(SCORE_DEEP_CASCADE),
        "irregular_activity_spike": _capped(SCORE_IRREGULAR_ACTIVITY),
        "high_closeness_centrality": _capped(SCORE_HIGH_CLOSENESS),
        "high_local_clustering": _capped(SCORE_HIGH_CLUSTERING),
        "rapid_forwarding": _capped(SCORE_RAPID_FORWARDING),
        "dormant_activation_spike": _capped(SCORE_DORMANT_ACTIVATION),
        "structured_fragmentation": _capped(SCORE_STRUCTURED_FRAGMENTATION),
    }

    pattern_map = {
        "cycle": cycle_accounts,
        "smurfing_aggregator": aggregators,
        "smurfing_disperser": dispersers,
        "shell_account": shell_accounts,
        "high_velocity": high_velocity,
        "rapid_pass_through": rapid_pass_through,
        "sudden_activity_spike": activity_spike,
        "high_betweenness_centrality": high_centrality,
        "low_retention_pass_through": low_retention,
        "high_throughput_ratio": high_throughput,
        "balance_oscillation_pass_through": balance_oscillation,
        "high_burst_diversity": burst_diversity,
        "large_scc_membership": scc_members,
        "deep_layered_cascade": cascade_depth,
        "irregular_activity_spike": irregular_activity,
        "high_closeness_centrality": high_closeness,
        "high_local_clustering": high_clustering,
        "rapid_forwarding": rapid_forwarding,
        "dormant_activation_spike": dormant_activation,
        "structured_fragmentation": structured_fragmentation,
    }

    for account in all_accounts:
        score = 0.0
        patterns: list[str] = []
        breakdown: Dict[str, float] = {}
        timeline: List[Dict[str, str]] = []
        pattern_count = 0

        for p_name, p_set in pattern_map.items():
            if account in p_set:
                p_score = pattern_scores[p_name]
                score += p_score
                patterns.append(p_name)
                breakdown[p_name] = p_score
                pattern_count += 1
                
                # Forensic timeline event
                event_time = trigger_times.get(p_name, {}).get(account, "Unknown")
                timeline.append({"time": event_time, "event": p_name.replace("_", " ").title()})

        # Sort timeline by time
        timeline.sort(key=lambda x: x["time"])

        # Multi-pattern bonus (≥2)
        if pattern_count >= 2:
            score += SCORE_MULTI_PATTERN_BONUS
            patterns.append("multi_pattern")
            breakdown["multi_pattern_bonus"] = SCORE_MULTI_PATTERN_BONUS

        # False positive dampeners
        if account in merchant_accounts:
            score += SCORE_MERCHANT_PENALTY
            patterns.append("merchant_like")
            breakdown["merchant_penalty"] = SCORE_MERCHANT_PENALTY

        if account in payroll_accounts:
            score += SCORE_PAYROLL_PENALTY
            patterns.append("payroll_like")
            breakdown["payroll_penalty"] = SCORE_PAYROLL_PENALTY

        # Nonlinear amplifier
        if pattern_count >= NONLINEAR_MIN_PATTERNS:
            nonlinear_bonus = NONLINEAR_BASE_BONUS + NONLINEAR_PER_PATTERN * (
                pattern_count - NONLINEAR_MIN_PATTERNS
            )
            score += nonlinear_bonus
            patterns.append("nonlinear_amplifier")
            breakdown["nonlinear_multiplier"] = nonlinear_bonus
        # Behavioural adjustments:
        # - Down-weight single-pattern hits (likely weak evidence)
        # - Up-weight rich multi-pattern behaviour
        core_pattern_count = sum(
            1 for p in patterns
            if p not in {"multi_pattern", "nonlinear_amplifier", "merchant_like", "payroll_like"}
        )
        if core_pattern_count == 1:
            score *= 0.75
        elif core_pattern_count >= 3:
            score *= 1.25

        # Clamp preliminary score to a safe range
        score = max(0.0, _capped(score))

        scores[account] = {
            "score": score,
            "patterns": patterns,
            "breakdown": breakdown,
            "timeline": timeline,
            "pattern_count": core_pattern_count,
        }

    # Second pass: globally normalize + non-linear shaping to avoid linear feel.
    import math

    max_score = max((data["score"] for data in scores.values()), default=0.0)
    if max_score > 0:
        for acct, data in scores.items():
            base = data["score"] / max_score  # 0–1
            # Exponential shaping: emphasise higher-risk accounts while keeping smooth.
            shaped = base ** 0.7
            data["score"] = round(shaped * 100.0, 2)

    return scores


def score_transactions_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load transactions from a tab-separated CSV file and compute a basic
    suspicion score per transaction using only essential features.
    """
    raw = pd.read_csv(csv_path, sep="\t")

    tx_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                raw["Timestamp"], format="%d-%m-%Y %H:%M", errors="coerce"
            ),
            "sender_id": raw["Sender UPI ID"],
            "receiver_id": raw["Receiver UPI ID"],
            "amount": pd.to_numeric(raw["Amount (INR)"], errors="coerce"),
        }
    )

    mask_valid = (
        tx_df["timestamp"].notna()
        & tx_df["sender_id"].notna()
        & tx_df["receiver_id"].notna()
        & tx_df["amount"].notna()
    )
    tx_df_valid = tx_df[mask_valid].copy()
    raw_valid = raw[mask_valid].copy()

    account_scores = compute_scores(
        df=tx_df_valid,
        cycle_accounts=set(),
        aggregators=set(),
        dispersers=set(),
        shell_accounts=set(),
        merchant_accounts=set(),
        payroll_accounts=set(),
    )

    def _txn_score(row: pd.Series) -> float:
        s = account_scores.get(row["sender_id"], {"score": 0.0})["score"]
        r = account_scores.get(row["receiver_id"], {"score": 0.0})["score"]
        return float((s + r) / 2.0)

    raw_valid["suspicion_score"] = tx_df_valid.apply(_txn_score, axis=1)

    if not mask_valid.all():
        raw_invalid = raw[~mask_valid].copy()
        raw_invalid["suspicion_score"] = 0.0
        raw_out = pd.concat([raw_valid, raw_invalid], ignore_index=True)
    else:
        raw_out = raw_valid

    return raw_out
