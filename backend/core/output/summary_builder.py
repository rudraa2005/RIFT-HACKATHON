"""
Summary Builder — constructs processing summary statistics.

Time Complexity: O(V + R)
Memory: O(1)
"""

from typing import Any, Dict, List


def build_summary(
    total_accounts: int,
    suspicious_count: int,
    rings_count: int,
    processing_time: float = 0.0,
) -> Dict[str, Any]:
    """Build the summary block for the JSON output."""
    return {
        "total_accounts_analyzed": total_accounts,
        "suspicious_accounts_flagged": suspicious_count,
        "fraud_rings_detected": rings_count,
        "processing_time_seconds": round(processing_time, 2),
    }
