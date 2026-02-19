"""
False Positive Control Module.

Detects legitimate business patterns to reduce false positives:
  Merchant Detection and Payroll Detection.

Time Complexity: O(V × E/V) ≈ O(E)
Memory: O(V)
"""

from typing import Set, Tuple

import pandas as pd

from app.config import (
    MERCHANT_MIN_COUNTERPARTIES,
    MERCHANT_MIN_SPAN_DAYS,
    PAYROLL_MIN_MONTHS,
    PAYROLL_SIMILAR_AMOUNT_CV,
)


def detect_merchants(df: pd.DataFrame) -> Set[str]:
    """Flag receiver accounts that look like merchants."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    merchants: Set[str] = set()

    for account in df["receiver_id"].unique():
        incoming = df[df["receiver_id"] == account]

        if incoming["sender_id"].nunique() < MERCHANT_MIN_COUNTERPARTIES:
            continue

        span_days = (incoming["timestamp"].max() - incoming["timestamp"].min()).days
        if span_days < MERCHANT_MIN_SPAN_DAYS:
            continue

        mean_amt = incoming["amount"].mean()
        if mean_amt > 0 and len(incoming) > 1:
            cv = incoming["amount"].std() / mean_amt
            if cv > 0.5:
                merchants.add(str(account))

    return merchants


def detect_payroll(df: pd.DataFrame) -> Set[str]:
    """Flag sender accounts that look like payroll disbursers."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    payroll: Set[str] = set()

    for account in df["sender_id"].unique():
        outgoing = df[df["sender_id"] == account]
        if len(outgoing) < 5:
            continue

        outgoing = outgoing.copy()
        outgoing["month"] = outgoing["timestamp"].dt.to_period("M")
        monthly_counts = outgoing.groupby("month").size()

        if len(monthly_counts) < PAYROLL_MIN_MONTHS:
            continue

        if (monthly_counts >= 3).sum() < PAYROLL_MIN_MONTHS:
            continue

        mean_amt = outgoing["amount"].mean()
        if mean_amt == 0:
            continue
        cv = outgoing["amount"].std() / mean_amt
        if cv < PAYROLL_SIMILAR_AMOUNT_CV:
            payroll.add(str(account))

    return payroll


def detect_false_positives(df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """
    Run all false-positive detection checks.

    Returns:
        (merchant_accounts, payroll_accounts)
    """
    return detect_merchants(df), detect_payroll(df)
