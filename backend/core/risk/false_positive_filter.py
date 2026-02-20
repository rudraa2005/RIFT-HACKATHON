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
    """Flag receiver accounts that look like merchants (legitimate businesses)."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    grouped = df.groupby("receiver_id").agg(
        unique_senders=("sender_id", "nunique"),
        min_ts=("timestamp", "min"),
        max_ts=("timestamp", "max"),
        mean_amt=("amount", "mean"),
        std_amt=("amount", "std"),
        total_txns=("amount", "size"),
    )
    if grouped.empty:
        return set()

    grouped["span_days"] = (grouped["max_ts"] - grouped["min_ts"]).dt.days
    grouped["std_amt"] = grouped["std_amt"].fillna(0.0)
    grouped["cv"] = grouped["std_amt"] / grouped["mean_amt"].replace(0, pd.NA)

    base = (
        (grouped["unique_senders"] >= MERCHANT_MIN_COUNTERPARTIES)
        & (grouped["span_days"] >= MERCHANT_MIN_SPAN_DAYS)
        & (grouped["mean_amt"] > 0)
        & (grouped["total_txns"] > 1)
    )
    varied = base & (grouped["cv"].fillna(0.0) > 0.5)
    wide_small_medium = (
        (grouped["unique_senders"] >= 10)
        & (grouped["span_days"] >= 30)
        & (grouped["mean_amt"] >= 100)
        & (grouped["mean_amt"] <= 10000)
    )
    return set(grouped[varied | wide_small_medium].index.astype(str))


def detect_payroll(df: pd.DataFrame) -> Set[str]:
    """Flag sender accounts that look like payroll disbursers AND recipients."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    outgoing = df[["sender_id", "receiver_id", "amount", "timestamp"]].copy()
    if outgoing.empty:
        return set()

    outgoing["month"] = outgoing["timestamp"].dt.to_period("M")
    sender_stats = outgoing.groupby("sender_id").agg(
        total_txns=("amount", "size"),
        mean_amt=("amount", "mean"),
        std_amt=("amount", "std"),
    )
    sender_stats = sender_stats[sender_stats["total_txns"] >= 5]
    if sender_stats.empty:
        return set()

    monthly = outgoing.groupby(["sender_id", "month"]).size().rename("month_count").reset_index()
    month_summary = monthly.groupby("sender_id").agg(
        months_seen=("month", "nunique"),
        months_with_3plus=("month_count", lambda s: int((s >= 3).sum())),
    )
    sender_stats = sender_stats.join(month_summary, how="left").fillna(0)
    sender_stats["cv"] = sender_stats["std_amt"].fillna(0.0) / sender_stats["mean_amt"].replace(0, pd.NA)

    payroll_senders = sender_stats[
        (sender_stats["months_seen"] >= PAYROLL_MIN_MONTHS)
        & (sender_stats["months_with_3plus"] >= PAYROLL_MIN_MONTHS)
        & (sender_stats["mean_amt"] > 0)
        & (sender_stats["cv"].fillna(1.0) < PAYROLL_SIMILAR_AMOUNT_CV)
    ].index.astype(str)

    payroll: Set[str] = set(payroll_senders)
    if len(payroll_senders) == 0:
        return payroll

    recipient_ids = outgoing[outgoing["sender_id"].astype(str).isin(payroll_senders)]["receiver_id"].astype(str).unique()
    payroll.update(recipient_ids.tolist())
    return payroll


def detect_false_positives(df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """
    Run all false-positive detection checks.

    Returns:
        (merchant_accounts, payroll_accounts)
    """
    return detect_merchants(df), detect_payroll(df)


