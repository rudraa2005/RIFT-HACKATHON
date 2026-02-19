# 🔍 Graph-Based Money Muling Detection Engine

A production-ready **FastAPI backend** that detects money muling rings using graph-based analysis of financial transaction data.

---

## 📌 Problem Statement

Money muling is a form of financial crime where illicit funds are moved through a chain of bank accounts to obscure their origin. This engine ingests raw transaction data (CSV), constructs a directed graph of money flows, and applies multiple detection algorithms to identify suspicious patterns.

---

## 🏗 Project Structure

```
app/
    main.py              # FastAPI endpoints
    config.py            # All detection thresholds
core/
    graph_builder.py     # NetworkX directed multigraph
    cycle_detection.py   # Circular fund routing (length 3–5)
    smurfing_detection.py# Fan-in / fan-out structuring
    shell_detection.py   # Shell account chains
    scoring_engine.py    # Suspicion scoring model
    false_positive_filter.py # Merchant & payroll filters
    risk_normalizer.py   # Score normalization [0–100]
    json_formatter.py    # Strict JSON output
services/
    processing_service.py# Pipeline orchestrator
utils/
    validators.py        # CSV validation
    time_utils.py        # Time window helpers
    metrics.py           # Processing stats
tests/
    test_detection.py    # Unit + integration tests
```

---

## 🧠 Detection Logic

### 1. Cycle Detection (Circular Fund Routing)

Finds simple cycles of length **3–5** in the transaction graph using `networkx.simple_cycles`.

**Filters:**

- Time span of cycle transactions **< 72 hours**
- Coefficient of variation (CV) of amounts **< 0.25** → similar amounts circling back

### 2. Smurfing Detection (Structuring)

**Fan-In (Aggregator):** ≥10 unique senders → 1 receiver within 72 h, with:

- Mean amount < `global_median × 0.6`
- Forward ratio > 0.7 (receiver quickly forwards most funds)

**Fan-Out (Disperser):** 1 sender → ≥10 unique receivers within 72 h, with:

- CV of outgoing amounts < 0.3 (similar-sized distributions)

### 3. Shell Chain Detection

**Shell account:** degree ≤ 3, ≤ 3 transactions, holding time < 24 h.

Finds chains of **≥ 3 hops** where all intermediate nodes are shell accounts — classic layering.

### 4. High Velocity Detection

`Velocity = total_amount / time_span`; accounts with normalized velocity **> 2.5×** global average are flagged.

---

## 🎯 Scoring Model

| Pattern                  | Points |
| ------------------------ | ------ |
| Cycle member             | +40    |
| Smurfing aggregator      | +30    |
| Smurfing disperser       | +30    |
| Shell account            | +35    |
| High velocity            | +20    |
| Multi-pattern bonus (≥2) | +15    |
| Merchant-like            | −40    |
| Payroll-like             | −30    |

Final score normalized to **0–100**, sorted descending.

---

## 🛡 False Positive Control

- **Merchant:** >50 counterparties, >90-day span, high variance → penalty
- **Payroll:** Monthly bulk transfers, similar amounts (CV < 0.1), ≥3 months → penalty

---

## 📊 Complexity Analysis

| Module          | Time                    | Space      |
| --------------- | ----------------------- | ---------- |
| Graph build     | O(E)                    | O(V+E)     |
| Cycle detection | O(V × L^L), L=5         | O(V×L)     |
| Smurfing        | O(n × k)                | O(n)       |
| Shell chains    | O(V × 3^D), D≤8         | O(V)       |
| Scoring         | O(V)                    | O(V)       |
| **Total**       | **< 30 s for 10K txns** | **O(V+E)** |

---

## 📄 API Endpoints

| Method | Path       | Description                                 |
| ------ | ---------- | ------------------------------------------- |
| `POST` | `/upload`  | Upload CSV → returns JSON detection results |
| `GET`  | `/health`  | System health check                         |
| `GET`  | `/metrics` | Processing stats from last run              |

### CSV Format (strict)

```csv
transaction_id,sender_id,receiver_id,amount,timestamp
TXN_001,ACC_001,ACC_002,1000.00,2024-01-10 10:00:00
```

### JSON Response Format

```json
{
  "suspicious_accounts": [
    {
      "account_id": "ACC_00123",
      "suspicion_score": 87.5,
      "detected_patterns": ["cycle_length_3", "high_velocity"],
      "ring_id": "RING_001"
    }
  ],
  "fraud_rings": [
    {
      "ring_id": "RING_001",
      "member_accounts": ["ACC_00123", "..."],
      "pattern_type": "cycle",
      "risk_score": 95.3
    }
  ],
  "summary": {
    "total_accounts_analyzed": 500,
    "suspicious_accounts_flagged": 15,
    "fraud_rings_detected": 4,
    "processing_time_seconds": 2.3
  }
}
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- pip

### Install

```bash
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Tests

```bash
python -m pytest tests/test_detection.py -v
```

### Test Upload (curl)

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@transactions.csv"
```

---

## 🐳 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t crime-detection .
docker run -p 8000:8000 crime-detection
```

### Cloud (any provider)

1. Push to Git
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

## 📜 License

MIT
