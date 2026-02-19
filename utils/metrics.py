"""
Processing statistics tracker.

Stores metrics from the most recent processing run for the /metrics endpoint.

Time Complexity: O(1) per operation
Memory: O(1)
"""

from typing import Any, Dict


class MetricsTracker:
    """Tracks processing statistics across API calls."""

    def __init__(self):
        self._last_metrics: Dict[str, Any] = {
            "status": "no_processing_yet",
            "total_runs": 0,
        }
        self._total_runs: int = 0

    def record(self, summary: Dict[str, Any]) -> None:
        """Record metrics from a processing run."""
        self._total_runs += 1
        self._last_metrics = {
            "status": "ready",
            "total_runs": self._total_runs,
            "last_run": summary,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return the latest metrics."""
        return self._last_metrics
