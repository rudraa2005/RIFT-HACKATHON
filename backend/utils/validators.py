"""
CSV structure validation.

Ensures uploaded CSV has the required columns with correct types and formats.

Time Complexity: O(n) where n = number of rows
Memory: O(1) additional beyond the DataFrame
"""

import pandas as pd
from typing import Any

REQUIRED_COLUMNS = [
    "transaction_id",
    "sender_id",
    "receiver_id",
    "amount",
    "timestamp",
]


def validate_csv(df: Any) -> str | None:
    import pandas as pd
    """
    Validate CSV structure. Returns error message if invalid, None if valid.

    Checks:
        1. All required columns present
        2. No null values in required columns
        3. 'amount' is numeric
        4. 'timestamp' is parseable as YYYY-MM-DD HH:MM:SS
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return f"Missing required columns: {', '.join(missing)}"

    if df.empty:
        return "CSV file is empty."

    null_cols = [col for col in REQUIRED_COLUMNS if df[col].isnull().any()]
    if null_cols:
        return f"Null values found in columns: {', '.join(null_cols)}"

    try:
        pd.to_numeric(df["amount"], errors="raise")
    except (ValueError, TypeError):
        return "Column 'amount' must contain numeric values."

    try:
        pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Column 'timestamp' must be in format YYYY-MM-DD HH:MM:SS."

    return None
