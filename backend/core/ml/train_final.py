"""
Final Training Pipeline — Combined structural + forensic model.

Loads:
  1. data/bank_transactions_data_2.csv
  2. data/transactions.csv
  3. data/synthetic_transactions_60neg_40pos.csv

Usage:
    python3 core/ml/train_final.py
"""

import os, sys, time, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from core.ml.feature_vector_builder import build_feature_vectors, vectors_to_matrix, FEATURE_NAMES
from core.graph.graph_builder import build_graph
from core.structural.cycle_detection import detect_cycles
from core.ring_detection.smurfing import detect_smurfing
from core.structural.shell_detection import detect_shell_chains
from core.temporal.forwarding_latency import detect_rapid_pass_through
from core.temporal.burst_detection import detect_activity_spikes
from core.centrality.betweenness import compute_centrality
from core.flow.retention_analysis import detect_low_retention
from core.flow.throughput_analysis import detect_high_throughput
from core.flow.balance_oscillation import detect_balance_oscillation
from core.ring_detection.diversity_analysis import detect_burst_diversity
from core.structural.scc_analysis import detect_scc
from core.structural.cascade_depth import detect_cascade_depth
from core.temporal.activity_consistency import detect_irregular_activity
from core.risk.false_positive_filter import detect_false_positives
from core.centrality.closeness import compute_closeness_centrality
from core.structural.clustering_analysis import detect_high_clustering
from core.risk.base_scoring import compute_scores, compute_high_velocity_accounts
from core.risk.normalization import normalize_scores

# Labeling
DETECTION_FAMILIES = {
    "structural":  {"cycle", "shell_account", "large_scc_membership", "deep_layered_cascade"},
    "flow":        {"low_retention_pass_through", "high_throughput_ratio", "balance_oscillation_pass_through"},
    "temporal":    {"rapid_pass_through", "sudden_activity_spike", "irregular_activity_spike"},
    "network":     {"high_betweenness_centrality", "high_closeness_centrality",
                    "high_local_clustering", "high_burst_diversity"},
    "behavioral":  {"high_velocity", "smurfing_aggregator", "smurfing_disperser",
                    "structured_fragmentation"},
}
CONSENSUS_FAMILIES_REQUIRED = 2
TARGET_FRAUD_RATIO = 0.10

CSV1 = "data/bank_transactions_data_2.csv"
CSV2 = "data/transactions.csv"
CSV3 = "data/synthetic_transactions_60neg_40pos.csv"
MODEL_DIR = "core/ml/models"

def _hdr(title, c="═"):
    w = 76
    print(f"\n{c * w}\n  {title}\n{c * w}")

def load_bank_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = []
    for _, r in raw.iterrows():
        if r["TransactionType"] == "Debit":
            rows.append({"transaction_id": r["TransactionID"],
                         "sender_id": r["AccountID"], "receiver_id": r["MerchantID"],
                         "amount": r["TransactionAmount"], "timestamp": r["TransactionDate"]})
        else:
            rows.append({"transaction_id": r["TransactionID"],
                         "sender_id": r["MerchantID"], "receiver_id": r["AccountID"],
                         "amount": r["TransactionAmount"], "timestamp": r["TransactionDate"]})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  [CSV1] Loaded {len(df)} txns")
    return df

def load_upi_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = []
    for _, r in raw.iterrows():
        rows.append({
            "transaction_id": r["Transaction ID"],
            "sender_id": str(r["Sender UPI ID"]),
            "receiver_id": str(r["Receiver UPI ID"]),
            "amount": float(r["Amount (INR)"]),
            "timestamp": r["Timestamp"],
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  [CSV2] Loaded {len(df)} txns")
    return df

def load_generic_csv(path: str) -> pd.DataFrame:
    """Load generic synthetic csv."""
    df = pd.read_csv(path)
    
    # Mapper for synthetic data: Sender_account -> sender_id, etc.
    rename_map = {
        "Sender_account": "sender_id",
        "Receiver_account": "receiver_id",
        "Amount": "amount"
    }
    df = df.rename(columns=rename_map)
    
    # Synthesize timestamp if Date and Time exist
    if "Date" in df.columns and "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    
    # Ensure transaction_id
    if "transaction_id" not in df.columns:
        df["transaction_id"] = ["SYN_" + str(i) for i in range(len(df))]
        
    required = ["sender_id", "receiver_id", "amount", "timestamp", "transaction_id"]
    df = df[required]
    
    print(f"  [CSV3] Loaded {len(df)} txns from {path}")
    return df

def load_combined() -> pd.DataFrame:
    frames = []
    if os.path.exists(CSV1): frames.append(load_bank_csv(CSV1))
    if os.path.exists(CSV2): frames.append(load_upi_csv(CSV2))
    if os.path.exists(CSV3):
        df3 = load_generic_csv(CSV3)
        # Sample to prevent bottlenecks in detection modules during training
        if len(df3) > 20000:
            print(f"  [SAMPLING] Reducing CSV3 from {len(df3)} to 20,000 for training efficiency.")
            df3 = df3.sample(n=20000, random_state=42)
        frames.append(df3)
    
    if not frames:
        raise FileNotFoundError("No CSV files found")
    df = pd.concat(frames, ignore_index=True)
    
    # Sanitize IDs (NaNs or mixed types break sorted())
    df = df.dropna(subset=["sender_id", "receiver_id"])
    df["sender_id"] = df["sender_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)
    
    print(f"  [COMBINED] {len(df)} total txns after sanitization/sampling")
    return df

def run_detection(df):
    _hdr("RUNNING DETECTION MODULES")
    G = build_graph(df)

    print(f"  - Detecting cycles...")
    cycle_rings = detect_cycles(G, df)
    cycle_accounts = set()
    for ring in cycle_rings:
        cycle_accounts.update(ring["members"])

    print(f"  - Detecting smurfing...")
    _, aggregators, dispersers, _ = detect_smurfing(df)
    
    print(f"  - Detecting shell accounts...")
    _, shell_accounts = detect_shell_chains(G, df)
    
    print(f"  - Detecting rapid pass-through...")
    rapid_pt, _ = detect_rapid_pass_through(df)
    
    print(f"  - Detecting activity spikes...")
    spikes, _ = detect_activity_spikes(df)
    
    print(f"  - Computing centrality...")
    centrality, _ = compute_centrality(G)
    
    print(f"  - Analyzing flow...")
    retention = detect_low_retention(df)
    throughput = detect_high_throughput(df)
    oscillation = detect_balance_oscillation(df)
    
    print(f"  - Analyzing burst diversity...")
    diversity, _ = detect_burst_diversity(df)
    
    print(f"  - Analyzing SCC...")
    scc_accts, _ = detect_scc(G)
    
    print(f"  - Analyzing cascade depth...")
    cascade = detect_cascade_depth(G, df)
    
    print(f"  - Analyzing activity consistency...")
    irregular = detect_irregular_activity(df)
    
    print(f"  - Filtering false positives...")
    merchant, payroll = detect_false_positives(df)
    
    print(f"  - Computing network metrics...")
    closeness, _ = compute_closeness_centrality(G, set())
    clustering, _ = detect_high_clustering(G, set())
    high_vel = compute_high_velocity_accounts(df)

    all_accts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
    print(f"  - Building feature vectors for {len(all_accts)} accounts...")
    fv, al = build_feature_vectors(
        all_accounts=all_accts, cycle_accounts=cycle_accounts,
        aggregators=aggregators, dispersers=dispersers,
        shell_accounts=shell_accounts, high_velocity=high_vel,
        rapid_pass_through=rapid_pt, activity_spike=spikes,
        high_centrality=centrality, low_retention=retention,
        high_throughput=throughput, balance_oscillation=oscillation,
        burst_diversity=diversity, scc_members=scc_accts,
        cascade_depth=cascade, irregular_activity=irregular,
        high_closeness=closeness, high_clustering=clustering,
        rapid_forwarding=set(), dormant_activation=set(),
        structured_fragmentation=set(), G=G, df=df,
    )
    X = vectors_to_matrix(fv, al)

    pattern_to_family = {}
    for fam, pats in DETECTION_FAMILIES.items():
        for p in pats:
            if p in FEATURE_NAMES:
                pattern_to_family[FEATURE_NAMES.index(p)] = fam

    y = np.zeros(len(al), dtype=np.int32)
    account_details = {}
    for i, acct in enumerate(al):
        if acct in merchant or acct in payroll:
            continue
        triggered = set()
        triggered_patterns = []
        for fi, fn in pattern_to_family.items():
            if fi < len(fv[acct]) and fv[acct][fi] > 0:
                triggered.add(fn)
                triggered_patterns.append(FEATURE_NAMES[fi])
        if len(triggered) >= CONSENSUS_FAMILIES_REQUIRED:
            y[i] = 1
        account_details[acct] = {
            "patterns": triggered_patterns,
            "label": int(y[i]),
        }

    print(f"  Accounts: {len(al)} | Positive: {y.sum()} ({y.mean()*100:.1f}%)")
    return X, y, al, fv, account_details, cycle_rings

def train_final_model(X, y):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = (n_neg / max(n_pos, 1)) * 5.0 # Moderated weighting

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    
    params = {
        "n_estimators": 600, "max_depth": 5, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": spw, "eval_metric": "logloss",
        "random_state": 42, "tree_method": "hist"
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    _hdr("RIFT FINAL TRAINING ENGINE")
    df = load_combined()
    X, y, al, fv, details, cycle_rings = run_detection(df)
    
    _hdr("TRAINING FINAL MODEL")
    model, X_test, y_test = train_final_model(X, y)
    
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= 0.5).astype(int)
    
    _hdr("MODEL PERFORMANCE")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision: {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"  Recall:    {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"  F1:        {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
    
    _hdr("SAVING MODEL")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "risk_model_final.json"))
    model.save_model(os.path.join(MODEL_DIR, "risk_model_v1.json"))
    print("  ✅ Final model saved to risk_model_final.json and v1 baseline updated.")

if __name__ == "__main__":
    main()
