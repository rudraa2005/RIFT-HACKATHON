# 🔍 Graph-Based Money Muling Detection Engine

A production-ready **FastAPI backend** that detects money muling rings using graph-based analysis of financial transaction data. Features **20 behavioral and structural detection patterns**, false positive suppression, risk propagation, and full ring density scoring.

---

## 📌 Problem Statement

Money muling is a form of financial crime where illicit funds are moved through a chain of bank accounts to obscure their origin. This engine ingests raw transaction data (CSV), constructs a directed graph of money flows, and applies multiple detection algorithms to identify suspicious patterns.

---

## 🏗 Project Structure

```
app/
    main.py                      # FastAPI application & endpoints
    config.py                    # All detection thresholds & scoring constants
    __init__.py

core/
    scoring_engine.py            # 20-pattern suspicion scoring model
    forwarding_latency.py        # Median-based rapid forwarding detection
    dormancy_analysis.py         # Dormant activation spike detection
    amount_structuring.py        # Structured amount fragmentation
    throughput_analysis.py       # Standalone throughput ratio analysis

    graph/
        graph_builder.py         # NetworkX directed multigraph construction
        graph_cache.py           # DataFrame hash-based graph caching
        graph_metrics.py         # Graph-level summary statistics

    flow/
        inflow_outflow.py        # Per-account inflow/outflow computation
        retention_analysis.py    # Net retention ratio (pass-through detection)
        throughput_analysis.py   # Throughput ratio analysis
        balance_oscillation.py   # Balance oscillation detection
        velocity_analysis.py     # High velocity account detection

    temporal/
        rolling_window.py        # Time-bounded sliding window utility
        burst_detection.py       # Activity spike detection
        activity_consistency.py  # Irregular activity variance
        forwarding_latency.py    # Holding time / rapid pass-through

    structural/
        cycle_detection.py       # Circular fund routing (length 3–5)
        scc_analysis.py          # Strongly connected components
        cascade_depth.py         # Deep layering via depth-limited DFS
        clustering_analysis.py   # Local clustering coefficient
        shell_detection.py       # Shell account chain detection

    centrality/
        betweenness.py           # Betweenness centrality (hub detection)
        closeness.py             # Closeness centrality

    risk/
        base_scoring.py          # Core scoring logic
        feature_registry.py      # Pattern-to-weight mapping
        risk_propagation.py      # Network-based risk propagation
        false_positive_filter.py # Merchant & payroll filters
        normalization.py         # Score clamping [0–100]
        ring_risk.py             # Ring density & enhanced risk scoring

    ring_detection/
        smurfing.py              # Fan-in / fan-out structuring
        fan_in.py                # Aggregator pattern detection
        fan_out.py               # Disperser pattern detection
        diversity_analysis.py    # Sender diversity burst detection
        ring_aggregator.py       # Ring list merging utility

    output/
        json_formatter.py        # Strict JSON output formatting
        summary_builder.py       # Processing summary construction

services/
    processing_service.py        # Full pipeline orchestrator

api/
    routes.py                    # API route definitions

utils/
    validators.py                # CSV structure validation
    metrics.py                   # Processing statistics tracker

tests/
    test_detection.py            # 64 unit + integration tests

data_samples/
    README.md                    # Sample CSV format documentation
```

---

## 🧠 Detection Patterns (20 Total)

### Structural Patterns

| #   | Pattern                 | Module                   | Description                                                 |
| --- | ----------------------- | ------------------------ | ----------------------------------------------------------- |
| 1   | `cycle`                 | `cycle_detection.py`     | Circular fund routing (length 3–5, < 72h, CV < 0.25)        |
| 2   | `shell_account`         | `shell_detection.py`     | Shell chains ≥ 3 hops (degree ≤ 3, ≤ 3 txns, holding < 24h) |
| 3   | `large_scc_membership`  | `scc_analysis.py`        | Strongly connected components ≥ 3 nodes                     |
| 4   | `deep_layered_cascade`  | `cascade_depth.py`       | Depth ≥ 3 within 72h window (DFS, max 5 hops)               |
| 5   | `high_local_clustering` | `clustering_analysis.py` | High local clustering coefficient                           |

### Flow & Behavioral Patterns

| #   | Pattern                            | Module                   | Description                                          |
| --- | ---------------------------------- | ------------------------ | ---------------------------------------------------- |
| 6   | `smurfing_aggregator`              | `smurfing.py`            | Fan-in: ≥ 10 senders → 1 receiver in 72h             |
| 7   | `smurfing_disperser`               | `smurfing.py`            | Fan-out: 1 sender → ≥ 10 receivers in 72h            |
| 8   | `high_velocity`                    | `scoring_engine.py`      | Velocity > 2.5× global average                       |
| 9   | `rapid_pass_through`               | `forwarding_latency.py`  | Short holding time + high forward ratio              |
| 10  | `rapid_forwarding`                 | `forwarding_latency.py`  | Median forwarding latency < 2 hours                  |
| 11  | `low_retention_pass_through`       | `retention_analysis.py`  | Retention ratio ≈ 0 (-0.1 to 0.1)                    |
| 12  | `high_throughput_ratio`            | `throughput_analysis.py` | Throughput ratio 0.9–1.1 (near-perfect pass-through) |
| 13  | `balance_oscillation_pass_through` | `balance_oscillation.py` | Repeating inflow/outflow cycles                      |
| 14  | `high_burst_diversity`             | `diversity_analysis.py`  | High sender diversity in short burst (not merchant)  |

### Temporal Patterns

| #   | Pattern                    | Module                    | Description                          |
| --- | -------------------------- | ------------------------- | ------------------------------------ |
| 15  | `sudden_activity_spike`    | `burst_detection.py`      | 5× baseline activity spike           |
| 16  | `irregular_activity_spike` | `activity_consistency.py` | High variance in activity windows    |
| 17  | `dormant_activation_spike` | `dormancy_analysis.py`    | ≥ 30 days dormant → ≥ 10 txns in 48h |
| 18  | `structured_fragmentation` | `amount_structuring.py`   | CV < 0.15 with ≥ 5 txns in 72h       |

### Centrality Patterns

| #   | Pattern                       | Module           | Description                           |
| --- | ----------------------------- | ---------------- | ------------------------------------- |
| 19  | `high_betweenness_centrality` | `betweenness.py` | Top 5th percentile betweenness        |
| 20  | `high_closeness_centrality`   | `closeness.py`   | High closeness in suspicious subgraph |

---

## 🎯 Scoring Model

| Pattern                        | Points              | Cap |
| ------------------------------ | ------------------- | --- |
| Cycle member                   | +40                 | 30  |
| Smurfing aggregator            | +30                 | 30  |
| Smurfing disperser             | +30                 | 30  |
| Shell account                  | +35                 | 30  |
| High velocity                  | +20                 | 30  |
| Rapid pass-through             | +25                 | 30  |
| Rapid forwarding               | +20                 | 30  |
| Sudden activity spike          | +20                 | 30  |
| High betweenness centrality    | +20                 | 30  |
| Low retention pass-through     | +25                 | 30  |
| High throughput ratio          | +20                 | 30  |
| Balance oscillation            | +20                 | 30  |
| High burst diversity           | +20                 | 30  |
| Large SCC membership           | +20                 | 30  |
| Deep layered cascade           | +25                 | 30  |
| Irregular activity spike       | +20                 | 30  |
| High closeness centrality      | +15                 | 30  |
| High local clustering          | +15                 | 30  |
| Dormant activation spike       | +20                 | 30  |
| Structured fragmentation       | +10                 | 30  |
| **Multi-pattern bonus (≥ 2)**  | **+15**             | —   |
| **Nonlinear amplifier (≥ 3)**  | **+10 + 5/pattern** | —   |
| Merchant-like (false positive) | −40                 | —   |
| Payroll-like (false positive)  | −30                 | —   |

Final score normalized to **0–100**, sorted descending. Risk propagation applied post-scoring.

---

## 🛡 False Positive Control

- **Merchant:** > 50 counterparties, > 90-day span, high variance → −40 penalty
- **Payroll:** Monthly bulk transfers, similar amounts (CV < 0.1), ≥ 3 months → −30 penalty

---

## 📊 Performance

| Module           | Time Complexity         | Space      |
| ---------------- | ----------------------- | ---------- |
| Graph build      | O(E)                    | O(V+E)     |
| Cycle detection  | O(V × L^L), L=5         | O(V×L)     |
| Smurfing         | O(n × k)                | O(n)       |
| Shell chains     | O(V × 3^D), D≤8         | O(V)       |
| SCC              | O(V + E)                | O(V)       |
| Cascade depth    | O(V × 5^D) DFS          | O(V)       |
| Risk propagation | O(3 × (V+E))            | O(V)       |
| Scoring          | O(V)                    | O(V)       |
| **Total**        | **< 30 s for 10K txns** | **O(V+E)** |

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
      "detected_patterns": ["cycle", "high_velocity", "rapid_forwarding"],
      "ring_id": "RING_001"
    }
  ],
  "fraud_rings": [
    {
      "ring_id": "RING_001",
      "member_accounts": ["ACC_00123", "ACC_00456"],
      "pattern_type": "cycle",
      "risk_score": 95.3,
      "density_score": 0.67
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

## 🧪 Testing

**64 tests** covering all modules, scoring, normalization, integration pipeline, and false positive control.

```bash
python -m pytest tests/test_detection.py -v
```

Synthetic datasets include: pure mule chains, merchant accounts, payroll distributors, dormant activations, structured bursts, rapid forwarders, deep cascades, and SCC clusters.

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

### Test Upload (curl)

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@transactions.csv"
```

---

## 🐳 Docker Deployment

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
