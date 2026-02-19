"""
Adaptive Threshold System — Percentile-based Dynamic Thresholds.

Innovation: Instead of hardcoded limits, we calculate thresholds based on 
the distribution of the current dataset (e.g., 95th percentile).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_adaptive_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate dynamic thresholds based on the statistical distribution of the dataset.
    
    Returns:
        Dict mapping metric_name to threshold_value
    """
    thresholds = {}
    
    # 1. Smurfing Fan-In Threshold (Unique Senders)
    # Group by receiver and count unique senders
    senders_per_acct = df.groupby("receiver_id")["sender_id"].nunique()
    if not senders_per_acct.empty:
        # We take the 95th percentile but floor it at a sane minimum (e.g. 3) 
        # to avoid noise in tiny datasets
        thresholds["smurfing_min_senders"] = max(3, float(np.percentile(senders_per_acct, 95)))
    else:
        thresholds["smurfing_min_senders"] = 10.0 # Fallback
        
    # 2. Smurfing Fan-Out Threshold (Unique Receivers)
    receivers_per_acct = df.groupby("sender_id")["receiver_id"].nunique()
    if not receivers_per_acct.empty:
        thresholds["smurfing_min_receivers"] = max(3, float(np.percentile(receivers_per_acct, 95)))
    else:
        thresholds["smurfing_min_receivers"] = 10.0 # Fallback
        
    # 3. Velocity Threshold (Multiplier)
    # Total amount per account / span hours
    # (Simplified version for quick threshold calculation)
    acct_amounts = df.groupby("sender_id")["amount"].sum()
    if not acct_amounts.empty:
        avg_amt = acct_amounts.mean()
        thresholds["velocity_multiplier"] = max(2.0, float(np.percentile(acct_amounts, 95) / avg_amt))
    else:
        thresholds["velocity_multiplier"] = 2.5 # Fallback
        
    # 4. Activity Spike Threshold (Transaction Count)
    tx_counts = df.groupby("sender_id").size()
    if not tx_counts.empty:
        thresholds["spike_min_txns"] = max(5, float(np.percentile(tx_counts, 95)))
    else:
        thresholds["spike_min_txns"] = 10.0 # Fallback
        
    return thresholds
