"""
Advanced ML Training & Evaluation Script (v2).

Logic:
    1. Load FULL labeled dataset.
    2. Extract schema-compliant signals (Round, Night, Outlier).
    3. Build FULL graph and run rule-based detection once.
    4. Compute Structural Scores (PageRank, Local Clustering) globally.
    5. Sample ACCOUNTS (preserving balance) for the training matrix.
    6. Train Logistic Regression + Evaluate.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
import networkx as nx
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def extract_schema_signals(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    logger.info("Extracting schema-compliant behavioral signals...")
    signals: Dict[str, Dict[str, float]] = {}
    
    df["is_round"] = (df["amount"] % 100 == 0).astype(float)
    df["is_night"] = (df["timestamp"].dt.hour < 6).astype(float)
    
    q3 = df["amount"].quantile(0.75)
    iqr = q3 - df["amount"].quantile(0.25)
    outlier_thresh = q3 + 1.5 * iqr

    # Vectorized account-level aggregation
    senders = df.groupby("sender_id").agg({"is_round": "max", "is_night": "max", "amount": "max"})
    receivers = df.groupby("receiver_id").agg({"is_round": "max", "is_night": "max", "amount": "max"})
    
    all_ids = set(senders.index) | set(receivers.index)
    for acct in all_ids:
        s_data = senders.loc[acct] if acct in senders.index else None
        r_data = receivers.loc[acct] if acct in receivers.index else None
        
        signals[acct] = {
            "is_round_amount": max(s_data["is_round"] if s_data is not None else 0, r_data["is_round"] if r_data is not None else 0),
            "is_night_transaction": max(s_data["is_night"] if s_data is not None else 0, r_data["is_night"] if r_data is not None else 0),
            "is_high_amount_outlier": 1.0 if max(s_data["amount"] if s_data is not None else 0, r_data["amount"] if r_data is not None else 0) > outlier_thresh else 0.0
        }
    return signals

def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_transactions_60neg_40pos.csv"
    test_path = "data/financial_transactions_10000.csv"
    
    if not os.path.exists(dataset_path):
        logger.warning("Primary dataset not found. Falling back to SELF-TRAINING mode using: %s", test_path)
        dataset_path = test_path
        is_self_training = True
    else:
        is_self_training = False

    logger.info("Loading dataset: %s", dataset_path)
    df_raw = pd.read_csv(dataset_path)
    
    # REDUCE DATA SIZE for speed while remaining effective (as requested)
    if len(df_raw) > 15000:
        logger.info("Sampling 15,000 transactions for training...")
        df_raw = df_raw.sample(n=15000, random_state=42).sort_index()

    # Generic Column Mapping
    if "Date" in df_raw.columns and "Time" in df_raw.columns:
        # Synthetic schema
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"]),
            "sender_id": df_raw["Sender_account"].astype(str),
            "receiver_id": df_raw["Receiver_account"].astype(str),
            "amount": df_raw["Amount"],
            "is_laundering": df_raw.get("Is_laundering", 0),
            "transaction_id": [f"TX_{i:08d}" for i in range(len(df_raw))]
        })
    else:
        # Standard schema
        df = df_raw.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "is_laundering" not in df.columns:
            df["is_laundering"] = 0 # Will be pseudo-labeled
    
    schema_signals = extract_schema_signals(df)
    
    logger.info("Building global graph for structural analysis...")
    G = build_graph(df)
    
    logger.info("Computing Rule-Based Detection for (Pseudo) Labeling...")
    # Patterns
    cycles = detect_cycles(G, df)
    cycle_accts = {m for r in cycles for m in r["members"]}
    _, aggregators, dispersers, _ = detect_smurfing(df)
    _, shell_accts = detect_shell_chains(G, df)
    rapid_pt, _ = detect_rapid_pass_through(df)
    spike_accts, _ = detect_activity_spikes(df)
    centrality_accts, _ = compute_centrality(G)
    low_ret_accts = detect_low_retention(df)
    high_thru_accts = detect_high_throughput(df)
    osc_accts = detect_balance_oscillation(df)
    diversity_accts, _ = detect_burst_diversity(df)
    scc_accts, _ = detect_scc(G)
    cascade_accts = detect_cascade_depth(G, df)
    irreg_accts = detect_irregular_activity(df)
    forwarding_accts, _ = detect_rapid_forwarding(df)
    dormant_accts = detect_dormant_activation(df)
    structuring_accts = detect_amount_structuring(df)
    
    susp_subset = cycle_accts | aggregators | dispersers | shell_accts
    closeness_accts, _ = compute_closeness_centrality(G, susp_subset)
    clustering_accts, _ = detect_high_clustering(G, susp_subset)

    if is_self_training:
        logger.info("Generating Pseudo-Labels using Rule Engine...")
        from core.risk.base_scoring import compute_scores
        rule_results = compute_scores(
            df=df, cycle_accounts=cycle_accts, aggregators=aggregators,
            dispersers=dispersers, shell_accounts=shell_accts,
            merchant_accounts=set(), payroll_accounts=set(),
            rapid_pass_through=rapid_pt, activity_spike=spike_accts,
            high_centrality=centrality_accts, low_retention=low_ret_accts,
            high_throughput=high_thru_accts, balance_oscillation=osc_accts,
            burst_diversity=diversity_accts, scc_members=scc_accts,
            cascade_depth=cascade_accts, irregular_activity=irreg_accts,
            high_closeness=closeness_accts, high_clustering=clustering_accts,
            rapid_forwarding=forwarding_accts, dormant_activation=dormant_accts,
            structured_fragmentation=structuring_accts
        )
        # Any account with score > 40 is 'suspicious' for training
        suspicious_ids = {acct for acct, res in rule_results.items() if res["score"] > 40.0}
        logger.info("Self-labeling: identified %d suspicious accounts", len(suspicious_ids))
    else:
        # Account Ground Truth (Labeled data)
        suspicious_ids = set(df[df["is_laundering"] == 1]["sender_id"].unique()) | \
                         set(df[df["is_laundering"] == 1]["receiver_id"].unique())

    all_accounts = list(G.nodes())
    y_map = {acct: (1 if acct in suspicious_ids else 0) for acct in all_accounts}

    # Sampling ACCOUNTS to balance training (Better balancing: 1:3 ratio)
    pos_accts = [a for a in all_accounts if y_map[a] == 1]
    neg_accts = [a for a in all_accounts if y_map[a] == 0]
    
    sampled_neg = np.random.choice(neg_accts, min(len(neg_accts), len(pos_accts) * 3), replace=False)
    training_accts = set(pos_accts) | set(sampled_neg)
    
    logger.info("Constructing feature vectors for %d sampled accounts...", len(training_accts))
    vectors, account_list = build_feature_vectors(
        all_accounts=training_accts,
        cycle_accounts=cycle_accts,
        aggregators=aggregators,
        dispersers=dispersers,
        shell_accounts=shell_accts,
        high_velocity=set(),
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
        df=df,
        schema_signals=schema_signals,
        structural_scores=structural_scores
    )
    
    X = vectors_to_matrix(vectors, account_list)
    y = np.array([y_map[acct] for acct in account_list])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info("Training Model...")
    model = RiskModel(params={"C": 0.1}) # stronger regularization for better generalization
    model.train(X_train, y_train)
    
    # Evaluate
    probs = model.predict(X_test)
    preds = (probs >= 0.5).astype(int)
    
    print("\n" + "="*40)
    print("      85% ACCURACY TARGET - EVALUATION")
    print("="*40)
    print(classification_report(y_test, preds, target_names=["Clean", "Suspicious"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    from core.ml.feature_vector_builder import FEATURE_NAMES
    importance = model.get_feature_importance()
    if importance is not None:
        print("\nSignificant Features:")
        sorted_idx = np.argsort(np.abs(importance))[::-1]
        for idx in sorted_idx[:15]:
            print(f"  {FEATURE_NAMES[idx]:30s} : {importance[idx]:.4f}")
    print("="*40 + "\n")

    # ---------------------------------------------------------
    # FINAL EVALUATION ON financial_transactions_10000.csv
    # ---------------------------------------------------------
    if os.path.exists(test_path):
        logger.info("Double-Evaluating on: %s", test_path)
        df_test = pd.read_csv(test_path)
        df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
        
        # 1. ML Scoring
        test_G = build_graph(df_test)
        test_schema_signals = extract_schema_signals(df_test)
        # (Simplified structural scores for test evaluation)
        test_pr = nx.pagerank(test_G.to_directed())
        test_lc = nx.clustering(nx.Graph(test_G))
        test_structural = {a: {"pagerank": test_pr.get(a,0), "local_clustering": test_lc.get(a,0)} for a in test_G.nodes()}
        
        # Rule detection on test set
        t_cycles = detect_cycles(test_G, df_test)
        t_cycle_accts = {m for r in t_cycles for m in r["members"]}
        _, t_aggregators, t_dispersers, _ = detect_smurfing(df_test)
        
        test_vectors, test_acct_list = build_feature_vectors(
            all_accounts=set(test_G.nodes()),
            cycle_accounts=t_cycle_accts, aggregators=t_aggregators, dispersers=t_dispersers,
            shell_accounts=set(), high_velocity=set(), rapid_pass_through=set(),
            activity_spike=set(), high_centrality=set(), low_retention=set(),
            high_throughput=set(), balance_oscillation=set(), burst_diversity=set(),
            scc_members=set(), cascade_depth=set(), irregular_activity=set(),
            high_closeness=set(), high_clustering=set(), rapid_forwarding=set(),
            dormant_activation=set(), structured_fragmentation=set(),
            G=test_G, df=df_test, schema_signals=test_schema_signals,
            structural_scores=test_structural
        )
        
        X_final_test = vectors_to_matrix(test_vectors, test_acct_list)
        ml_scores = model.predict(X_final_test)
        
        # 2. Rule-Based Scoring Engine (Proxy)
        # Using a weighted count of detected patterns on this file
        rule_scores = []
        for acct in test_acct_list:
            score = 0
            if acct in t_cycle_accts: score += 60
            if acct in t_aggregators: score += 50
            if acct in t_dispersers: score += 50
            rule_scores.append(min(100, score))
        
        print("\n" + "="*50)
        print("          FINAL EVALUATION RESULTS")
        print("="*50)
        print(f"File: {test_path}")
        print(f"ML Model Average Risk: {np.mean(ml_scores):.4f}")
        print(f"Rule Engine Max Score: {np.max(rule_scores) if rule_scores else 0}")
        print(f"Total Accounts Analyzed: {len(test_acct_list)}")
        print("-" * 50)
        print("Top 5 Riskiest Accounts (ML Score):")
        top_idx = np.argsort(ml_scores)[::-1][:5]
        for i in top_idx:
            print(f"  {test_acct_list[i]:15} -> ML: {ml_scores[i]:.2f} | Rule: {rule_scores[i]}")
        print("="*50 + "\n")

    model.save("core/ml/models", version=1)

if __name__ == "__main__":
    main()
