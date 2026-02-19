import pandas as pd
import json
import os
import sys
import time

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from services.processing_pipeline import ProcessingService
from core.output.json_formatter import format_output

def verify():
    csv_path = "backend/data/financial_transactions_10000.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    service = ProcessingService()
    
    start_time = time.time()
    results = service.process(df)
    end_time = time.time()
    
    proc_time = end_time - start_time
    print(f"Processing Time: {proc_time:.2f} seconds")
    
    suspicious = results.get('suspicious_accounts', [])
    scores = [a['suspicion_score'] for a in suspicious]
    
    if not scores:
        print("No suspicious accounts found.")
        return
        
    print(f"Total Suspicious Accounts: {len(scores)}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Avg Score: {sum(scores)/len(scores):.2f}")
    
    # Check distribution
    high_risk = len([s for s in scores if s >= 90])
    med_risk = len([s for s in scores if 50 <= s < 90])
    low_risk = len([s for s in scores if 1 <= s < 50])
    
    print("\nScore Distribution:")
    print(f"  High Risk (>=90): {high_risk} ({high_risk/len(scores)*100:.1f}%)")
    print(f"  Med Risk (50-89): {med_risk} ({med_risk/len(scores)*100:.1f}%)")
    print(f"  Low Risk (1-49):  {low_risk} ({low_risk/len(scores)*100:.1f}%)")
    
    # Check connectivity metrics existence
    conn = results.get('summary', {}).get('network_connectivity', {})
    print("\nConnectivity Metrics Check:")
    print(f"  SCC Distribution: {'OK' if 'scc_distribution' in conn else 'MISSING'}")
    print(f"  Burst Activity: {'OK' if 'burst_activity' in conn else 'MISSING'}")
    print(f"  Depth Dist: {'OK' if 'depth_distribution' in conn else 'MISSING'}")

    if proc_time > 30:
        print("\nWARNING: Processing time exceeds 30s!")
    else:
        print("\nSUCCESS: Processing time within limits.")

if __name__ == "__main__":
    verify()
