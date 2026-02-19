"""
Risk Score Normalizer.

Clamps raw suspicion scores to [0, 100] and rounds to 2 decimal places.

Time Complexity: O(V)
Memory: O(V)
"""

from typing import Any, Dict


def normalize_scores(
    scores: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Clamp all scores to [0, 100] and round to 2 decimal places."""
    return {
        account: {**data, "score": round(max(0.0, min(100.0, data["score"])), 2)}
        for account, data in scores.items()
    }
