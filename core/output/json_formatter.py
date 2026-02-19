"""
JSON Output Formatter.

Produces the strict required output structure:
{
    "suspicious_accounts": [...],
    "fraud_rings": [...],
    "summary": {...}
}

Time Complexity: O(V log V) for sorting
Memory: O(V + R)
"""

from typing import Any, Dict, List


def _build_account_ring_map(rings: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map each account to its primary (first-encountered) ring_id."""
    account_ring: Dict[str, str] = {}
    for ring in rings:
        for member in ring["members"]:
            if member not in account_ring:
                account_ring[member] = ring["ring_id"]
    return account_ring


# Patterns that are internal markers, not user-facing
_INTERNAL_PATTERNS = {"merchant_like", "payroll_like", "multi_pattern", "nonlinear_amplifier"}


def format_output(
    scores: Dict[str, Dict[str, Any]],
    all_rings: List[Dict[str, Any]],
    total_accounts: int,
) -> Dict[str, Any]:
    """Build the final JSON-compatible output dict."""
    account_ring_map = _build_account_ring_map(all_rings)

    suspicious_accounts: List[Dict[str, Any]] = []
    for account_id, data in scores.items():
        if data["score"] <= 0:
            continue

        detected_patterns = [
            p for p in data["patterns"] if p not in _INTERNAL_PATTERNS
        ]
        if not detected_patterns:
            continue

        suspicious_accounts.append(
            {
                "account_id": account_id,
                "suspicion_score": round(data["score"], 2),
                "detected_patterns": detected_patterns,
                "ring_id": account_ring_map.get(account_id, "RING_NONE"),
            }
        )

    suspicious_accounts.sort(key=lambda x: x["suspicion_score"], reverse=True)

    fraud_rings: List[Dict[str, Any]] = []
    for ring in all_rings:
        ring_obj = {
            "ring_id": ring["ring_id"],
            "member_accounts": ring["members"],
            "pattern_type": ring["pattern_type"],
            "risk_score": round(ring["risk_score"], 2),
        }
        if "density_score" in ring:
            ring_obj["density_score"] = round(ring["density_score"], 2)
        fraud_rings.append(ring_obj)

    fraud_rings.sort(key=lambda x: x["risk_score"], reverse=True)

    summary = {
        "total_accounts_analyzed": total_accounts,
        "suspicious_accounts_flagged": len(suspicious_accounts),
        "fraud_rings_detected": len(fraud_rings),
        "processing_time_seconds": 0.0,
    }

    return {
        "suspicious_accounts": suspicious_accounts,
        "fraud_rings": fraud_rings,
        "summary": summary,
    }
