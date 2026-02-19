"""
Rolling Window Utility.

Provides helper for creating time-bounded sliding windows over transaction data.

Time Complexity: O(n)
Memory: O(n)
"""

from datetime import timedelta
from typing import Generator

import pandas as pd


def rolling_windows(
    df: pd.DataFrame,
    window_hours: int = 72,
) -> Generator:
    """
    Yield (row_index, window_df) for each row's trailing window.

    Args:
        df: Must have a 'timestamp' column (already datetime).
        window_hours: Size of the trailing window in hours.

    Yields:
        (int, pd.DataFrame): Row index and the window DataFrame
    """
    delta = timedelta(hours=window_hours)
    df = df.sort_values("timestamp")

    for i in range(len(df)):
        end = df["timestamp"].iloc[i]
        start = end - delta
        window = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        yield i, window
