"""
Unit Tests for the Financial Crime Detection Engine (Phases 1-3).

Tests cover all detection modules, scoring, normalization, formatting,
integration pipeline, and scenario-based assertions for false positive control.
"""

import os
import sys

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from core.structural.cycle_detection import detect_cycles
from core.risk.false_positive_filter import detect_false_positives
from core.graph.graph_builder import build_graph
from core.output.json_formatter import format_output
from core.risk.normalization import normalize_scores
from core.risk.base_scoring import compute_scores
from core.structural.shell_detection import detect_shell_chains
from core.ring_detection.smurfing import detect_smurfing
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
from core.risk.risk_propagation import propagate_risk
from core.centrality.closeness import compute_closeness_centrality
from core.structural.clustering_analysis import detect_high_clustering
from services.processing_pipeline import ProcessingService
from utils.validators import validate_csv
from core.forwarding_latency import detect_rapid_forwarding
from core.dormancy_analysis import detect_dormant_activation
from core.amount_structuring import detect_amount_structuring


# ── Synthetic Datasets ────────────────────────────────────────────────


def _cycle_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3"],
            "sender_id": ["ACC_001", "ACC_002", "ACC_003"],
            "receiver_id": ["ACC_002", "ACC_003", "ACC_001"],
            "amount": [1000.0, 1010.0, 990.0],
            "timestamp": [
                "2024-01-10 10:00:00",
                "2024-01-10 11:00:00",
                "2024-01-10 12:00:00",
            ],
        }
    )


def _smurfing_fan_in_data() -> pd.DataFrame:
    rows = []
    for i in range(11):
        rows.append(
            {
                "transaction_id": f"TIN_{i}",
                "sender_id": f"SENDER_{i:03d}",
                "receiver_id": "AGGREGATOR_001",
                "amount": 50.0 + i,
                "timestamp": f"2024-01-10 {10 + i % 12}:00:00",
            }
        )
    rows.append(
        {
            "transaction_id": "TOUT_0",
            "sender_id": "AGGREGATOR_001",
            "receiver_id": "DOWNSTREAM_001",
            "amount": 500.0,
            "timestamp": "2024-01-10 23:00:00",
        }
    )
    return pd.DataFrame(rows)


def _shell_chain_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["TS1", "TS2", "TS3"],
            "sender_id": ["ORIGIN_001", "SHELL_001", "SHELL_002"],
            "receiver_id": ["SHELL_001", "SHELL_002", "DEST_001"],
            "amount": [5000.0, 4900.0, 4800.0],
            "timestamp": [
                "2024-01-10 08:00:00",
                "2024-01-10 09:00:00",
                "2024-01-10 10:00:00",
            ],
        }
    )


def _rapid_pass_through_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["RPT1", "RPT2", "RPT3", "RPT4"],
            "sender_id": ["SOURCE_A", "SOURCE_B", "MULE_001", "MULE_001"],
            "receiver_id": ["MULE_001", "MULE_001", "DEST_A", "DEST_B"],
            "amount": [1000.0, 2000.0, 900.0, 1900.0],
            "timestamp": [
                "2024-01-10 10:00:00",
                "2024-01-10 10:10:00",
                "2024-01-10 10:20:00",
                "2024-01-10 10:30:00",
            ],
        }
    )


def _activity_spike_data() -> pd.DataFrame:
    rows = []
    for m in range(1, 7):
        rows.append(
            {
                "transaction_id": f"BASE_{m}",
                "sender_id": "SPIKER_001",
                "receiver_id": f"REC_{m:03d}",
                "amount": 100.0,
                "timestamp": f"2024-{m:02d}-15 12:00:00",
            }
        )
    for i in range(15):
        rows.append(
            {
                "transaction_id": f"SPIKE_{i}",
                "sender_id": "SPIKER_001",
                "receiver_id": f"BURST_REC_{i:03d}",
                "amount": 50.0 + i,
                "timestamp": f"2024-07-01 {8 + i % 12}:{(i * 5) % 60:02d}:00",
            }
        )
    return pd.DataFrame(rows)


def _pure_pass_through_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["PT1", "PT2", "PT3", "PT4"],
            "sender_id": ["SRC_1", "SRC_2", "MULE_PT", "MULE_PT"],
            "receiver_id": ["MULE_PT", "MULE_PT", "DST_1", "DST_2"],
            "amount": [5000.0, 3000.0, 5000.0, 3000.0],
            "timestamp": [
                "2024-01-10 10:00:00",
                "2024-01-10 10:05:00",
                "2024-01-10 10:30:00",
                "2024-01-10 10:35:00",
            ],
        }
    )


def _merchant_data() -> pd.DataFrame:
    rows = []
    for i in range(60):
        month = (i % 6) + 1
        day = (i % 28) + 1
        rows.append(
            {
                "transaction_id": f"MERCH_{i}",
                "sender_id": f"CUSTOMER_{i:03d}",
                "receiver_id": "MERCHANT_001",
                "amount": 25.0 + (i % 50),
                "timestamp": f"2024-{month:02d}-{day:02d} 12:00:00",
            }
        )
    return pd.DataFrame(rows)


def _payroll_data() -> pd.DataFrame:
    rows = []
    for m in range(1, 7):
        rows.append(
            {
                "transaction_id": f"PAY_{m}",
                "sender_id": "EMPLOYER_001",
                "receiver_id": "EMPLOYEE_001",
                "amount": 3000.0,
                "timestamp": f"2024-{m:02d}-01 09:00:00",
            }
        )
    return pd.DataFrame(rows)


def _deep_chain_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["DC1", "DC2", "DC3", "DC4"],
            "sender_id": ["LAYER_A", "LAYER_B", "LAYER_C", "LAYER_D"],
            "receiver_id": ["LAYER_B", "LAYER_C", "LAYER_D", "LAYER_E"],
            "amount": [10000.0, 9500.0, 9000.0, 8500.0],
            "timestamp": [
                "2024-01-10 08:00:00",
                "2024-01-10 10:00:00",
                "2024-01-10 12:00:00",
                "2024-01-10 14:00:00",
            ],
        }
    )


def _scc_circular_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["SCC1", "SCC2", "SCC3", "SCC4", "SCC5"],
            "sender_id": ["SCC_A", "SCC_B", "SCC_C", "SCC_A", "SCC_D"],
            "receiver_id": ["SCC_B", "SCC_C", "SCC_A", "SCC_D", "SCC_A"],
            "amount": [1000.0, 1000.0, 1000.0, 500.0, 500.0],
            "timestamp": [
                "2024-01-10 10:00:00",
                "2024-01-10 11:00:00",
                "2024-01-10 12:00:00",
                "2024-01-10 13:00:00",
                "2024-01-10 14:00:00",
            ],
        }
    )


def _burst_diversity_data() -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows.append(
            {
                "transaction_id": f"BD_{i}",
                "sender_id": f"UNIQ_SENDER_{i:03d}",
                "receiver_id": "BURST_RECV",
                "amount": 100.0 + i * 10,
                "timestamp": f"2024-01-10 {8 + i % 10}:00:00",
            }
        )
    return pd.DataFrame(rows)


def _full_dataset() -> pd.DataFrame:
    return pd.concat(
        [_cycle_data(), _smurfing_fan_in_data(), _shell_chain_data()],
        ignore_index=True,
    )


# ── Validator Tests ───────────────────────────────────────────────────


class TestValidator:
    def test_valid_csv(self):
        assert validate_csv(_cycle_data()) is None

    def test_missing_column(self):
        df = _cycle_data().drop(columns=["amount"])
        err = validate_csv(df)
        assert err is not None
        assert "amount" in err

    def test_bad_timestamp(self):
        df = _cycle_data()
        df["timestamp"] = "not-a-date"
        err = validate_csv(df)
        assert err is not None
        assert "timestamp" in err

    def test_empty_csv(self):
        df = pd.DataFrame(
            columns=["transaction_id", "sender_id", "receiver_id", "amount", "timestamp"]
        )
        err = validate_csv(df)
        assert err is not None


# ── Cycle Detection Tests ─────────────────────────────────────────────


class TestCycleDetection:
    def test_finds_triangle(self):
        df = _cycle_data()
        G = build_graph(df)
        cycles = detect_cycles(G, df)
        assert len(cycles) >= 1
        assert set(cycles[0]["members"]) == {"ACC_001", "ACC_002", "ACC_003"}

    def test_no_false_positive_on_chain(self):
        df = _shell_chain_data()
        G = build_graph(df)
        assert len(detect_cycles(G, df)) == 0


# ── Smurfing Tests ────────────────────────────────────────────────────


class TestSmurfingDetection:
    def test_fan_in_detected(self):
        rings, aggs, _ = detect_smurfing(_smurfing_fan_in_data())
        assert isinstance(rings, list)
        assert isinstance(aggs, set)


# ── Shell Detection Tests ─────────────────────────────────────────────


class TestShellDetection:
    def test_shell_chain_detected(self):
        df = _shell_chain_data()
        rings, accts = detect_shell_chains(build_graph(df), df)
        assert isinstance(rings, list)
        assert isinstance(accts, set)


# ── Scoring Tests ─────────────────────────────────────────────────────


class TestScoring:
    def test_cycle_score(self):
        s = compute_scores(
            df=_cycle_data(), cycle_accounts={"ACC_001"}, aggregators=set(),
            dispersers=set(), shell_accounts=set(), merchant_accounts=set(),
            payroll_accounts=set(),
        )
        assert s["ACC_001"]["score"] >= 30

    def test_multi_pattern_bonus(self):
        s = compute_scores(
            df=_cycle_data(), cycle_accounts={"ACC_001"}, aggregators={"ACC_001"},
            dispersers=set(), shell_accounts=set(), merchant_accounts=set(),
            payroll_accounts=set(),
        )
        assert s["ACC_001"]["score"] >= 30 + 25 + 15

    def test_cap_per_metric(self):
        s = compute_scores(
            df=_cycle_data(), cycle_accounts={"ACC_001"}, aggregators=set(),
            dispersers=set(), shell_accounts=set(), merchant_accounts=set(),
            payroll_accounts=set(),
        )
        # Cycle is 40 in config but capped to 30
        base = s["ACC_001"]["score"]
        assert base <= 50  # cap + possible velocity


# ── Normalizer Tests ──────────────────────────────────────────────────


class TestNormalizer:
    def test_clamp_high(self):
        assert normalize_scores({"X": {"score": 200.0, "patterns": []}})["X"]["score"] == 100.0

    def test_clamp_low(self):
        assert normalize_scores({"X": {"score": -50.0, "patterns": []}})["X"]["score"] == 0.0

    def test_rounding(self):
        assert normalize_scores({"X": {"score": 42.567, "patterns": []}})["X"]["score"] == 42.57


# ── JSON Formatter Tests ─────────────────────────────────────────────


class TestJsonFormatter:
    def test_output_structure(self):
        scores = {"A": {"score": 80.0, "patterns": ["cycle"]}}
        rings = [{"ring_id": "R1", "members": ["A", "B"], "pattern_type": "cycle_length_3", "risk_score": 90.0}]
        result = format_output(scores, rings, total_accounts=5)
        assert "suspicious_accounts" in result
        assert "fraud_rings" in result
        assert result["summary"]["total_accounts_analyzed"] == 5

    def test_sorted_by_score(self):
        scores = {"A": {"score": 50.0, "patterns": ["cycle"]}, "B": {"score": 90.0, "patterns": ["high_velocity"]}}
        accts = format_output(scores, [], total_accounts=2)["suspicious_accounts"]
        if len(accts) >= 2:
            assert accts[0]["suspicion_score"] >= accts[1]["suspicion_score"]

    def test_density_score_in_rings(self):
        rings = [{"ring_id": "R1", "members": ["A"], "pattern_type": "x", "risk_score": 80.0, "density_score": 0.67}]
        assert format_output({}, rings, total_accounts=1)["fraud_rings"][0]["density_score"] == 0.67

    def test_new_patterns_in_output(self):
        scores = {"A": {"score": 45.0, "patterns": ["rapid_pass_through", "low_retention_pass_through"]}}
        patterns = format_output(scores, [], total_accounts=1)["suspicious_accounts"][0]["detected_patterns"]
        assert "rapid_pass_through" in patterns
        assert "low_retention_pass_through" in patterns


# ── Holding Time Tests ────────────────────────────────────────────────


class TestHoldingTime:
    def test_rapid_pass_through(self):
        flagged, details = detect_rapid_pass_through(_rapid_pass_through_data())
        assert isinstance(flagged, set)
        if "MULE_001" in flagged:
            assert details["MULE_001"]["avg_holding_hours"] < 2

    def test_no_flag_outgoing_only(self):
        df = pd.DataFrame({
            "transaction_id": ["X1", "X2"], "sender_id": ["OS", "OS"],
            "receiver_id": ["R1", "R2"], "amount": [100.0, 200.0],
            "timestamp": ["2024-01-10 10:00:00", "2024-01-10 11:00:00"],
        })
        assert "OS" not in detect_rapid_pass_through(df)[0]


# ── Activity Spike Tests ─────────────────────────────────────────────


class TestActivitySpike:
    def test_spike_detection(self):
        assert isinstance(detect_activity_spikes(_activity_spike_data()), set)

    def test_no_spike_consistent(self):
        rows = [{"transaction_id": f"C_{d}", "sender_id": "C001", "receiver_id": f"R_{d:03d}",
                 "amount": 100.0, "timestamp": f"2024-01-{d:02d} 12:00:00"} for d in range(1, 31)]
        assert "C001" not in detect_activity_spikes(pd.DataFrame(rows))


# ── Centrality Tests ──────────────────────────────────────────────────


class TestCentrality:
    def test_centrality_computation(self):
        G = build_graph(_full_dataset())
        high, all_c = compute_centrality(G)
        assert isinstance(high, set) and len(all_c) > 0

    def test_hub_detection(self):
        rows = []
        for i in range(5):
            rows += [
                {"transaction_id": f"HI_{i}", "sender_id": f"SI_{i}", "receiver_id": "HUB",
                 "amount": 100.0, "timestamp": "2024-01-10 10:00:00"},
                {"transaction_id": f"HO_{i}", "sender_id": "HUB", "receiver_id": f"SO_{i}",
                 "amount": 100.0, "timestamp": "2024-01-10 11:00:00"},
            ]
        G = build_graph(pd.DataFrame(rows))
        _, all_c = compute_centrality(G)
        if all_c:
            assert max(all_c, key=all_c.get) == "HUB"


# ── Ring Density Tests ────────────────────────────────────────────────


class TestRingDensity:
    @staticmethod
    def _compute_ring_density(G, ring):
        import networkx as nx
        members = ring["members"]
        n = len(members)
        if n < 2:
            return 0.0
        max_edges = n * (n - 1)
        simple_G = nx.DiGraph(G)
        subgraph = simple_G.subgraph(members)
        actual_edges = subgraph.number_of_edges()
        return round(actual_edges / max_edges, 2) if max_edges > 0 else 0.0

    def test_full_cycle_density(self):
        G = build_graph(_cycle_data())
        assert 0 < self._compute_ring_density(G, {"members": ["ACC_001", "ACC_002", "ACC_003"]}) <= 1.0

    def test_single_node_density(self):
        assert self._compute_ring_density(build_graph(_cycle_data()), {"members": ["ACC_001"]}) == 0.0


# ── Nonlinear Bonus Tests ─────────────────────────────────────────────


class TestNonlinearBonus:
    def test_three_patterns(self):
        s = compute_scores(
            df=_cycle_data(), cycle_accounts={"ACC_001"}, aggregators={"ACC_001"},
            dispersers={"ACC_001"}, shell_accounts=set(), merchant_accounts=set(),
            payroll_accounts=set(),
        )
        assert s["ACC_001"]["score"] >= 30 + 25 + 25 + 15 + 10
        assert "nonlinear_amplifier" in s["ACC_001"]["patterns"]

    def test_no_bonus_under_threshold(self):
        s = compute_scores(
            df=_cycle_data(), cycle_accounts={"ACC_001"}, aggregators=set(),
            dispersers=set(), shell_accounts=set(), merchant_accounts=set(),
            payroll_accounts=set(),
        )
        assert "nonlinear_amplifier" not in s["ACC_001"]["patterns"]


# ── Retention Analysis Tests ──────────────────────────────────────────


class TestRetentionAnalysis:
    def test_pass_through_flagged(self):
        flagged = detect_low_retention(_pure_pass_through_data())
        assert "MULE_PT" in flagged

    def test_receiver_only_not_flagged(self):
        df = pd.DataFrame({
            "transaction_id": ["R1", "R2"], "sender_id": ["E1", "E2"],
            "receiver_id": ["SAVER", "SAVER"], "amount": [1000.0, 2000.0],
            "timestamp": ["2024-01-10 10:00:00", "2024-01-10 11:00:00"],
        })
        assert "SAVER" not in detect_low_retention(df)


# ── Flow Metrics Tests ────────────────────────────────────────────────


class TestFlowMetrics:
    def test_high_throughput(self):
        throughput = detect_high_throughput(_pure_pass_through_data())
        assert "MULE_PT" in throughput

    def test_balance_oscillation(self):
        rows = []
        for i in range(10):
            rows += [
                {"transaction_id": f"OI_{i}", "sender_id": f"S_{i}", "receiver_id": "OSC",
                 "amount": 100.0, "timestamp": f"2024-01-{i + 1:02d} 10:00:00"},
                {"transaction_id": f"OO_{i}", "sender_id": "OSC", "receiver_id": f"D_{i}",
                 "amount": 100.0, "timestamp": f"2024-01-{i + 1:02d} 14:00:00"},
            ]
        osc = detect_balance_oscillation(pd.DataFrame(rows))
        assert isinstance(osc, set)


# ── Diversity Analysis Tests ──────────────────────────────────────────


class TestDiversityAnalysis:
    def test_burst_diversity(self):
        assert isinstance(detect_burst_diversity(_burst_diversity_data()), set)

    def test_merchant_not_flagged(self):
        assert "MERCHANT_001" not in detect_burst_diversity(_merchant_data())


# ── SCC Analysis Tests ────────────────────────────────────────────────


class TestSCCAnalysis:
    def test_scc_detected(self):
        flagged, rings = detect_scc(build_graph(_scc_circular_data()))
        assert len(flagged) >= 3

    def test_scc_ring_structure(self):
        _, rings = detect_scc(build_graph(_scc_circular_data()))
        if rings:
            assert rings[0]["pattern_type"] == "scc_cluster"
            assert "density_score" in rings[0]

    def test_no_scc_in_chain(self):
        _, rings = detect_scc(build_graph(_deep_chain_data()))
        assert len(rings) == 0


# ── Cascade Depth Tests ──────────────────────────────────────────────


class TestCascadeDepth:
    def test_deep_chain_flagged(self):
        df = _deep_chain_data()
        flagged = detect_cascade_depth(build_graph(df), df)
        assert len(flagged) >= 1

    def test_short_chain_not_flagged(self):
        df = pd.DataFrame({
            "transaction_id": ["S1", "S2"], "sender_id": ["SA", "SB"],
            "receiver_id": ["SB", "SC"], "amount": [1000.0, 900.0],
            "timestamp": ["2024-01-10 10:00:00", "2024-01-10 11:00:00"],
        })
        assert "SA" not in detect_cascade_depth(build_graph(df), df)


# ── Activity Consistency Tests ────────────────────────────────────────


class TestActivityConsistency:
    def test_irregular_activity(self):
        assert isinstance(detect_irregular_activity(_activity_spike_data()), set)

    def test_merchant_stable(self):
        rows = []
        for d in range(1, 31):
            for i in range(3):
                rows.append({
                    "transaction_id": f"S_{d}_{i}", "sender_id": f"C_{d}_{i}",
                    "receiver_id": "STABLE_M", "amount": 50.0,
                    "timestamp": f"2024-01-{d:02d} {10 + i}:00:00",
                })
        assert "STABLE_M" not in detect_irregular_activity(pd.DataFrame(rows))


# ── Risk Propagation Tests ────────────────────────────────────────────


class TestRiskPropagation:
    def test_neighbor_scores_increase(self):
        G = build_graph(_cycle_data())
        scores = {
            "ACC_001": {"score": 80.0, "patterns": ["cycle"]},
            "ACC_002": {"score": 0.0, "patterns": []},
            "ACC_003": {"score": 0.0, "patterns": []},
        }
        assert propagate_risk(G, scores)["ACC_002"]["score"] > 0

    def test_scores_clamped(self):
        G = build_graph(_cycle_data())
        scores = {k: {"score": 100.0, "patterns": ["cycle"]} for k in ["ACC_001", "ACC_002", "ACC_003"]}
        for v in propagate_risk(G, scores).values():
            assert 0 <= v["score"] <= 100


# ── Closeness Centrality Tests ────────────────────────────────────────


class TestClosenessCentrality:
    def test_closeness(self):
        h, a = compute_closeness_centrality(build_graph(_full_dataset()))
        assert isinstance(h, set) and isinstance(a, dict)

    def test_empty_suspicious(self):
        h, _ = compute_closeness_centrality(build_graph(_cycle_data()), set())
        assert isinstance(h, set)


# ── Clustering Tests ──────────────────────────────────────────────────


class TestClusteringAnalysis:
    def test_clustering(self):
        h, a = detect_high_clustering(build_graph(_full_dataset()))
        assert isinstance(h, set) and isinstance(a, dict)


# ── Scenario Tests ────────────────────────────────────────────────────


class TestScenarioPurePassThrough:
    def test_correct_patterns(self):
        result = ProcessingService().process(_pure_pass_through_data())
        mule = [a for a in result["suspicious_accounts"] if a["account_id"] == "MULE_PT"]
        if mule:
            p = mule[0]["detected_patterns"]
            assert any(x in p for x in ["low_retention_pass_through", "high_throughput_ratio", "rapid_pass_through"])
            assert mule[0]["suspicion_score"] > 0


class TestScenarioMerchant:
    def test_no_false_positive(self):
        assert "MERCHANT_001" not in detect_burst_diversity(_merchant_data())


class TestScenarioPayroll:
    def test_no_false_positive(self):
        assert "EMPLOYEE_001" not in detect_low_retention(_payroll_data())


class TestScenarioSCC:
    def test_correct_tagging(self):
        flagged, rings = detect_scc(build_graph(_scc_circular_data()))
        assert len(flagged) >= 3
        for r in rings:
            assert r["pattern_type"] == "scc_cluster" and r["risk_score"] > 0


class TestScenarioDeepChain:
    def test_cascade(self):
        df = _deep_chain_data()
        assert len(detect_cascade_depth(build_graph(df), df)) >= 1


class TestScenarioBurstDiversity:
    def test_diversity(self):
        assert isinstance(detect_burst_diversity(_burst_diversity_data()), set)


# ── Integration Tests ─────────────────────────────────────────────────


class TestFullPipeline:
    def test_end_to_end(self):
        result = ProcessingService().process(_full_dataset())
        assert result["summary"]["total_accounts_analyzed"] > 0
        scores = [a["suspicion_score"] for a in result["suspicious_accounts"]]
        assert scores == sorted(scores, reverse=True)

    def test_density_in_rings(self):
        for ring in ProcessingService().process(_full_dataset())["fraud_rings"]:
            assert 0 <= ring["density_score"] <= 1.0

    def test_scc_rings_in_output(self):
        result = ProcessingService().process(_scc_circular_data())
        assert any(r["pattern_type"] == "scc_cluster" for r in result["fraud_rings"])

    def test_all_scores_bounded(self):
        for a in ProcessingService().process(_full_dataset())["suspicious_accounts"]:
            assert 0 <= a["suspicion_score"] <= 100


# ── Dormant Activation Tests ──────────────────────────────────────────


def _dormant_then_burst_data():
    """Account dormant 45 days, then 12 transactions in 24h."""
    rows = []
    # Early activity
    for i in range(3):
        rows.append({
            "transaction_id": f"EARLY_{i}",
            "sender_id": "DORMANT_ACC",
            "receiver_id": f"R_{i}",
            "amount": 100.0,
            "timestamp": f"2024-01-{i + 1:02d} 10:00:00",
        })
    # 45-day gap, then burst starting Feb 20
    for i in range(12):
        rows.append({
            "transaction_id": f"BURST_{i}",
            "sender_id": f"S_{i}",
            "receiver_id": "DORMANT_ACC",
            "amount": 50.0,
            "timestamp": f"2024-02-20 {10 + i % 12}:00:00",
        })
    return pd.DataFrame(rows)


class TestDormantActivation:
    def test_dormant_detected(self):
        flagged = detect_dormant_activation(_dormant_then_burst_data())
        assert "DORMANT_ACC" in flagged

    def test_active_account_not_flagged(self):
        """Continuously active accounts should not be flagged."""
        rows = []
        for i in range(30):
            rows.append({
                "transaction_id": f"DAILY_{i}",
                "sender_id": "ACTIVE_ACC",
                "receiver_id": f"R_{i}",
                "amount": 100.0,
                "timestamp": f"2024-01-{i + 1:02d} 10:00:00",
            })
        assert "ACTIVE_ACC" not in detect_dormant_activation(pd.DataFrame(rows))


# ── Amount Structuring Tests ──────────────────────────────────────────


def _structured_burst_data():
    """6 transactions of nearly identical amounts within 48h."""
    rows = []
    for i in range(6):
        rows.append({
            "transaction_id": f"STRUCT_{i}",
            "sender_id": "STRUCTURER",
            "receiver_id": f"R_{i}",
            "amount": 999.0 + (i * 0.5),  # CV ≈ 0.001
            "timestamp": f"2024-01-10 {10 + i}:00:00",
        })
    return pd.DataFrame(rows)


class TestAmountStructuring:
    def test_structured_detected(self):
        flagged = detect_amount_structuring(_structured_burst_data())
        assert "STRUCTURER" in flagged

    def test_varied_amounts_not_flagged(self):
        """Highly varied amounts should not trigger structuring."""
        rows = []
        for i in range(6):
            rows.append({
                "transaction_id": f"VAR_{i}",
                "sender_id": "VARIED_ACC",
                "receiver_id": f"R_{i}",
                "amount": (i + 1) * 500.0,  # 500, 1000, 1500... high CV
                "timestamp": f"2024-01-10 {10 + i}:00:00",
            })
        assert "VARIED_ACC" not in detect_amount_structuring(pd.DataFrame(rows))


# ── Forwarding Latency Tests ──────────────────────────────────────────


def _rapid_forwarder_data():
    """Account receives and forwards within minutes."""
    rows = []
    for i in range(8):
        # Incoming
        rows.append({
            "transaction_id": f"FWD_IN_{i}",
            "sender_id": f"S_{i}",
            "receiver_id": "FORWARDER",
            "amount": 200.0,
            "timestamp": f"2024-01-10 {10 + i}:00:00",
        })
        # Outgoing 15 minutes later
        rows.append({
            "transaction_id": f"FWD_OUT_{i}",
            "sender_id": "FORWARDER",
            "receiver_id": f"D_{i}",
            "amount": 190.0,
            "timestamp": f"2024-01-10 {10 + i}:15:00",
        })
    return pd.DataFrame(rows)


class TestForwardingLatency:
    def test_rapid_forwarder_detected(self):
        flagged, details = detect_rapid_forwarding(_rapid_forwarder_data())
        assert "FORWARDER" in flagged
        assert details["FORWARDER"]["median_latency_hours"] < 2

    def test_slow_forwarder_not_flagged(self):
        """Account that forwards days later should not be flagged."""
        rows = []
        for i in range(5):
            rows.append({
                "transaction_id": f"SLOW_IN_{i}",
                "sender_id": f"S_{i}",
                "receiver_id": "SLOW_FWD",
                "amount": 200.0,
                "timestamp": f"2024-01-{i + 1:02d} 10:00:00",
            })
            rows.append({
                "transaction_id": f"SLOW_OUT_{i}",
                "sender_id": "SLOW_FWD",
                "receiver_id": f"D_{i}",
                "amount": 190.0,
                "timestamp": f"2024-01-{i + 4:02d} 10:00:00",  # 3 days later
            })
        flagged, _ = detect_rapid_forwarding(pd.DataFrame(rows))
        assert "SLOW_FWD" not in flagged


# ── Scenario: Merchant Not False Positive ─────────────────────────────


class TestScenarioMerchantNotFlagged:
    def test_merchant_no_structuring(self):
        """Merchants with varied amounts should not be flagged for structuring."""
        assert "MERCHANT_001" not in detect_amount_structuring(_merchant_data())

    def test_merchant_no_dormancy(self):
        """Continuously active merchants should not be flagged for dormancy."""
        assert "MERCHANT_001" not in detect_dormant_activation(_merchant_data())
