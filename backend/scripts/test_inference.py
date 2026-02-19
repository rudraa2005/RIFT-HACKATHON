
import pandas as pd
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath("."))

from services.processing_pipeline import ProcessingService

logging.basicConfig(level=logging.INFO)

def test_inference():
    # Load sample data
    df = pd.read_csv("data/money_muling_test_dataset.csv")
    
    # Initialize service
    service = ProcessingService()
    
    # Process
    try:
        results = service.process(df)
        print("Successfully processed data!")
        
        # Verify JSON serialization
        import json
        json_output = json.dumps(results)
        print("JSON serialization successful!")
        
        # Log a few scores
        for node in results["suspicious_accounts"][:5]:
            print(f"Account: {node['account_id']}, Score: {node['suspicion_score']}, patterns: {node['detected_patterns']}")
            
        # Check for the new boost pattern
        for node in results["suspicious_accounts"]:
            if "network_concentration_boost" in node.get("score_breakdown", {}):
                 print(f"Found Network Concentration Boost for {node['account_id']}! Boost value: {node['score_breakdown']['network_concentration_boost']}")
            
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_inference()
