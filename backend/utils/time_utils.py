"""
Time utility functions for sliding window and time span calculations.

Time Complexity: O(n log n) for sorting-based operations
Memory: O(n) for sorted copies
"""

import pandas as pd
from datetime import timedelta
from typing import List


def get_time_span_hours(timestamps: List) -> float:
    """Calculate time span in hours between min and max timestamps."""
    if not timestamps:
        return 0.0
    ts = pd.to_datetime(timestamps)
    span = ts.max() - ts.min()
    return span.total_seconds() / 3600


def is_within_hours(timestamps: List, hours: int) -> bool:
    """Check if all timestamps fall within the given hour window."""
    if not timestamps:
        return True
    ts = pd.to_datetime(timestamps)
    span = (ts.max() - ts.min()).total_seconds() / 3600
    return span <= hours


def sliding_window_groups(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    window_hours: int,
) -> dict:
    """
    Group transactions by account within a sliding time window.

    Returns dict of {account_id: [list of DataFrames per window]}.

    Time Complexity: O(n * k) where n = rows, k = average windows per account
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    result = {}
    window_delta = timedelta(hours=window_hours)

    for account, group in df.groupby(group_col):
        group = group.sort_values(time_col).reset_index(drop=True)
        if group.empty:
            continue

        windows = []
        start_idx = 0

        for i in range(len(group)):
            while group[time_col].iloc[start_idx] < group[time_col].iloc[i] - window_delta:
                start_idx += 1
            window_df = group.iloc[start_idx : i + 1]
            if len(window_df) > 1:
                windows.append(window_df)

        if windows:
            result[account] = windows

    return result
