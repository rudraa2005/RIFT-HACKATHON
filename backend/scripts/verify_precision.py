"""
Verification script for precision fixes on aml_stress_test_dataset.csv.
Checks: no duplicate patterns, sane ring count, correct top accounts, no CUST flagged.
"""
import pandas as pd
import logging
import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath("."))

from services.processing_pipeline import ProcessingService

logging.basicConfig(level=logging.WARNING)

def verify():
    df = pd.read_csv("data/aml_stress_test_dataset.csv")
    service = ProcessingService()

    t0 = time.time()
    results = service.process(df)
    elapsed = time.time() - t0

    print(f"Processing Time: {elapsed:.2f}s")

    # 1. Check top suspicious accounts
    suspicious = results["suspicious_accounts"]
    print(f"\nTotal Suspicious Accounts: {len(suspicious)}")
    
    print("\n--- Top 10 Accounts ---")
    for acc in suspicious[:10]:
        print(f"  {acc['account_id']:20s} score={acc['suspicion_score']:7.2f} patterns={acc['detected_patterns']}")
    
    # 2. Check for duplicate patterns
    dup_count = 0
    for acc in suspicious:
        patterns = acc["detected_patterns"]
        if len(patterns) != len(set(patterns)):
            dup_count += 1
            print(f"  DUPLICATE PATTERNS in {acc['account_id']}: {patterns}")
    
    if dup_count == 0:
        print("\n✅ No duplicate patterns found.")
    else:
        print(f"\n❌ {dup_count} accounts have duplicate patterns!")
    
    # 3. Check ring count and deduplication
    rings = results["fraud_rings"]
    print(f"\nFraud Rings: {len(rings)}")
    for ring in rings:
        print(f"  {ring['ring_id']:20s} type={ring['pattern_type']:15s} members={ring['member_accounts']} risk={ring['risk_score']}")
    
    # Check for duplicate member sets
    member_sets = [frozenset(r["member_accounts"]) for r in rings]
    unique_sets = set(member_sets)
    if len(member_sets) == len(unique_sets):
        print("✅ No duplicate rings (by member set).")
    else:
        print(f"❌ {len(member_sets) - len(unique_sets)} duplicate ring(s)!")
    
    # 4. Check CUST accounts are NOT in top positions
    top_ids = [a["account_id"] for a in suspicious[:10]]
    cust_in_top = [aid for aid in top_ids if aid.startswith("CUST_")]
    if not cust_in_top:
        print("✅ No CUST accounts in top 10.")
    else:
        print(f"❌ CUST accounts in top 10: {cust_in_top}")
    
    # 5. Check MERCHANT is NOT flagged
    merchant_flagged = [a for a in suspicious if a["account_id"].startswith("MERCHANT")]
    if not merchant_flagged:
        print("✅ No MERCHANT accounts flagged.")
    else:
        print(f"❌ MERCHANT accounts flagged: {[m['account_id'] for m in merchant_flagged]}")
    
    # 6. Score distribution
    scores_list = [a["suspicion_score"] for a in suspicious]
    if scores_list:
        print(f"\nScore Distribution:")
        high = sum(1 for s in scores_list if s >= 90)
        med = sum(1 for s in scores_list if 50 <= s < 90)
        low = sum(1 for s in scores_list if s < 50)
        print(f"  High (>=90): {high}")
        print(f"  Med  (50-89): {med}")
        print(f"  Low  (<50):   {low}")
    
    # 7. Processing time check
    if elapsed < 30:
        print(f"\n✅ Processing time {elapsed:.2f}s is under 30s limit.")
    else:
        print(f"\n❌ Processing time {elapsed:.2f}s exceeds 30s limit!")
    
    # 8. Check SCC ring is not present
    scc_rings = [r for r in rings if r["pattern_type"] == "scc_cluster"]
    if not scc_rings:
        print("✅ No SCC-only rings in output.")
    else:
        print(f"❌ {len(scc_rings)} SCC-only ring(s) still present!")

if __name__ == "__main__":
    verify()
