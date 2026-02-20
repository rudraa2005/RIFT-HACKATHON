"""
Offline evaluator for labeled CSVs.

This script does NOT affect API runtime latency because it is not used by
the upload route. It is only for local/CI evaluation.

Usage:
  cd backend
  python scripts/evaluate_labeled_csv.py --csv data/labeled.csv --label-column is_fraud
"""

import argparse
import os
import sys
import time
from typing import Set

import pandas as pd


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Normalize common label formats (1/0, true/false, yes/no)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )
    positives = {"1", "true", "t", "yes", "y", "fraud", "suspicious"}
    return normalized.isin(positives)


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pipeline on labeled CSV.")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV file.")
    parser.add_argument(
        "--label-column",
        default="is_fraud",
        help="Label column name (default: is_fraud).",
    )
    args = parser.parse_args()

    csv_path = args.csv
    label_column = args.label_column

    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        return 1

    # Resolve backend imports no matter where script is run from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(script_dir)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    from services.processing_pipeline import ProcessingService

    df = pd.read_csv(csv_path)

    required = {"transaction_id", "sender_id", "receiver_id", "amount", "timestamp"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: missing required columns: {', '.join(missing)}")
        return 1
    if label_column not in df.columns:
        print(f"Error: label column not found: {label_column}")
        return 1

    # Build account-level ground truth from transaction-level labels:
    # if an account appears in any positive-labeled txn -> positive account.
    label_mask = _to_bool_series(df[label_column])
    fraud_tx = df[label_mask]

    positive_accounts: Set[str] = set(fraud_tx["sender_id"].astype(str).tolist()) | set(
        fraud_tx["receiver_id"].astype(str).tolist()
    )
    all_accounts: Set[str] = set(df["sender_id"].astype(str).tolist()) | set(
        df["receiver_id"].astype(str).tolist()
    )

    t0 = time.time()
    result = ProcessingService().process(df)
    elapsed = time.time() - t0

    predicted_accounts: Set[str] = {
        str(a.get("account_id"))
        for a in result.get("suspicious_accounts", [])
        if a.get("account_id") is not None
    }

    tp = len(predicted_accounts & positive_accounts)
    fp = len(predicted_accounts - positive_accounts)
    fn = len(positive_accounts - predicted_accounts)
    tn = len(all_accounts - (predicted_accounts | positive_accounts))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(all_accounts))

    print(f"Rows: {len(df)}")
    print(f"Accounts: {len(all_accounts)}")
    print(f"Runtime: {elapsed:.2f}s")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

