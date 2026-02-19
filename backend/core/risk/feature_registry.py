"""
Feature Registry — maps pattern names to their score contributions.

Central registry of all detection patterns and their scoring weights.
Used by base_scoring.py and for introspection/documentation.
"""

from typing import Dict

from app.config import (
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
    SCORE_RAPID_PASS_THROUGH,
    SCORE_SHELL,
    SCORE_SMURFING_AGGREGATOR,
    SCORE_SMURFING_DISPERSER,
)


FEATURE_REGISTRY: Dict[str, int] = {
    "cycle": SCORE_CYCLE,
    "smurfing_aggregator": SCORE_SMURFING_AGGREGATOR,
    "smurfing_disperser": SCORE_SMURFING_DISPERSER,
    "shell_account": SCORE_SHELL,
    "high_velocity": SCORE_HIGH_VELOCITY,
    "rapid_pass_through": SCORE_RAPID_PASS_THROUGH,
    "sudden_activity_spike": SCORE_ACTIVITY_SPIKE,
    "high_betweenness_centrality": SCORE_HIGH_CENTRALITY,
    "low_retention_pass_through": SCORE_LOW_RETENTION,
    "high_throughput_ratio": SCORE_HIGH_THROUGHPUT,
    "balance_oscillation_pass_through": SCORE_BALANCE_OSCILLATION,
    "high_burst_diversity": SCORE_BURST_DIVERSITY,
    "large_scc_membership": SCORE_LARGE_SCC,
    "deep_layered_cascade": SCORE_DEEP_CASCADE,
    "irregular_activity_spike": SCORE_IRREGULAR_ACTIVITY,
    "high_closeness_centrality": SCORE_HIGH_CLOSENESS,
    "high_local_clustering": SCORE_HIGH_CLUSTERING,
}


def get_all_patterns() -> list[str]:
    """Return sorted list of all registered pattern names."""
    return sorted(FEATURE_REGISTRY.keys())


def get_score(pattern: str) -> int:
    """Return score contribution for a pattern name."""
    return FEATURE_REGISTRY.get(pattern, 0)
