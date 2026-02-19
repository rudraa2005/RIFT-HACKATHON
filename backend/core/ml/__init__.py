"""
ML Scoring Layer.

Provides XGBoost-based risk scoring that integrates with the existing
rule-based scoring engine to produce hybrid risk scores.
"""

from core.ml.feature_vector_builder import build_feature_vectors
from core.ml.ml_model import RiskModel
from core.ml.hybrid_scorer import compute_hybrid_scores

__all__ = ["build_feature_vectors", "RiskModel", "compute_hybrid_scores"]
