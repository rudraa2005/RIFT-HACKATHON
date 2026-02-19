"""
ML Training & Evaluation Script.

Usage:
    python3 scripts/train_model.py [dataset_path]

Logic:
    1. Load labeled synthetic data.
    2. Sample data to balance labels and optimize speed.
    3. Run rule-based detectors to generate feature vectors.
    4. Label accounts based on 'Is_laundering' flag.
    5. Train Logistic Regression model.
    6. Evaluate and print Accuracy, Precision, Recall, F1, ROC-AUC.
"""

import os
import sys
import time
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from core.centrality.closeness import compute_closeness_centrality
from core.structural.clustering_analysis import detect_high_clustering
from core.forwarding_latency import detect_rapid_forwarding
from core.dormancy_analysis import detect_dormant_activation
from core.amount_structuring import detect_amount_structuring
from core.ml.feature_vector_builder import build_feature_vectors, vectors_to_matrix
from core.ml.ml_model import RiskModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_transactions_60neg_40pos.csv"
    
    if not os.path.exists(dataset_path):
        logger.error("Dataset not found: %s", dataset_path)
        return

    logger.info("Loading dataset: %s", dataset_path)
    df_raw = pd.read_csv(dataset_path)
    
    # Map columns to internal names
    # Time,Date,Sender_account,Receiver_account,Amount,Payment_currency,...Is_laundering
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"]),
        "sender_id": df_raw["Sender_account"].astype(str),
        "receiver_id": df_raw["Receiver_account"].astype(str),
        "amount": df_raw["Amount"],
        "is_laundering": df_raw["Is_laundering"],
        "transaction_id": [f"TX_{i:08d}" for i in range(len(df_raw))]
    })
    
    # Sample data for speed (keep it under 50k transactions)
    # Ensure we keep a good mix of positives
    positives = df[df["is_laundering"] == 1]
    negatives = df[df["is_laundering"] == 0]
    
    # Limit to 30,000 for training efficiency
    df_train = pd.concat([
        positives.sample(min(len(positives), 15000), random_state=42),
        negatives.sample(min(len(negatives), 15000), random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info("Processing %d transactions for training...", len(df_train))
    
    # 1. Build ground truth labels at account level
    # An account is suspicious if it was involved in any laundering transaction
    suspicious_accounts = set(df_train[df_train["is_laundering"] == 1]["sender_id"].unique()) | \
                          set(df_train[df_train["is_laundering"] == 1]["receiver_id"].unique())
    
    all_accounts = set(df_train["sender_id"].unique()) | set(df_train["receiver_id"].unique())
    y_map = {acct: (1 if acct in suspicious_accounts else 0) for acct in all_accounts}

    # 2. Run rule-based detectors to generate features
    logger.info("Running rule-based detectors for feature extraction...")
    G = build_graph(df_train)
    
    # We call detectors (using default thresholds for training)
    cycles = detect_cycles(G, df_train)
    cycle_accts = {m for r in cycles for m in r["members"]}
    _, aggregators, dispersers, _ = detect_smurfing(df_train)
    _, shell_accts = detect_shell_chains(G, df_train)
    rapid_pt, _ = detect_rapid_pass_through(df_train)
    spike_accts, _ = detect_activity_spikes(df_train)
    centrality_accts, _ = compute_centrality(G)
    low_ret_accts = detect_low_retention(df_train)
    high_thru_accts = detect_high_throughput(df_train)
    osc_accts = detect_balance_oscillation(df_train)
    diversity_accts, _ = detect_burst_diversity(df_train)
    scc_accts, _ = detect_scc(G)
    cascade_accts = detect_cascade_depth(G, df_train)
    irreg_accts = detect_irregular_activity(df_train)
    
    # Advanced
    susp_subset = cycle_accts | aggregators | dispersers | shell_accts
    closeness_accts, _ = compute_closeness_centrality(G, susp_subset)
    clustering_accts, _ = detect_high_clustering(G, susp_subset)
    forwarding_accts, _ = detect_rapid_forwarding(df_train)
    dormant_accts = detect_dormant_activation(df_train)
    structuring_accts = detect_amount_structuring(df_train)

    # 3. Build feature vectors
    logger.info("Constructing feature vectors...")
    vectors, account_list = build_feature_vectors(
        all_accounts=all_accounts,
        cycle_accounts=cycle_accts,
        aggregators=aggregators,
        dispersers=dispersers,
        shell_accounts=shell_accts,
        high_velocity=set(), # velocity computed inside scoring usually, but we can skip if needed
        rapid_pass_through=rapid_pt,
        activity_spike=spike_accts,
        high_centrality=centrality_accts,
        low_retention=low_ret_accts,
        high_throughput=high_thru_accts,
        balance_oscillation=osc_accts,
        burst_diversity=diversity_accts,
        scc_members=scc_accts,
        cascade_depth=cascade_accts,
        irregular_activity=irreg_accts,
        high_closeness=closeness_accts,
        high_clustering=clustering_accts,
        rapid_forwarding=forwarding_accts,
        dormant_activation=dormant_accts,
        structured_fragmentation=structuring_accts,
        G=G,
        df=df_train
    )
    
    X = vectors_to_matrix(vectors, account_list)
    y = np.array([y_map[acct] for acct in account_list])
    
    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Label distribution: Suspicious=%d, Clean=%d", np.sum(y == 1), np.sum(y == 0))

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Train Model
    logger.info("Training Logistic Regression model...")
    model = RiskModel()
    model.train(X_train, y_train)
    
    # 6. Evaluate
    logger.info("Evaluating model...")
    probs = model.predict(X_test)
    preds = (probs >= 0.5).astype(int)
    
    print("\n" + "="*40)
    print("      ML MODEL PERFORMANCE REPORT")
    print("="*40)
    print(classification_report(y_test, preds, target_names=["Clean", "Suspicious"]))
    
    auc = roc_auc_score(y_test, probs)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Precision at Top 5%: {eval_metrics['precision_at_top5_pct']:.4f}")
    print("="*40 + "\n")

    # 7. Save Model
    model_dir = "core/ml/models"
    save_path = model.save(model_dir, version=1)
    logger.info("Model saved successfully: %s", save_path)

if __name__ == "__main__":
    main()
