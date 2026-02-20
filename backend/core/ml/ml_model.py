"""
ML Risk Model — Logistic Regression Wrapper.

Provides train, save/load, predict, and evaluate capabilities for a
binary Logistic Regression classifier that outputs probability scores [0, 1].

Replaces XGBoost for significantly faster inference and better transparency.

Time Complexity:
    train: O(n_samples × n_features)
    predict: O(n_samples × n_features)  (~<0.1ms for typical batch)
Memory: O(n_features)
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default Logistic Regression hyperparameters
_DEFAULT_PARAMS: Dict[str, Any] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 42,
    "class_weight": "balanced",
}


class RiskModel:
    """
    Logistic Regression-based risk scoring model with built-in scaling.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with optional hyperparameter overrides."""
        if not _ML_AVAILABLE:
            logger.warning("scikit-learn not installed. ML scoring unavailable.")
            self._model = None
            return

        self._params = {**_DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[LogisticRegression] = None
        self._scaler = StandardScaler()

    @property
    def is_available(self) -> bool:
        return _ML_AVAILABLE

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> "RiskModel":
        """Train the classifier with automatic feature scaling."""
        if not _ML_AVAILABLE:
            raise RuntimeError("scikit-learn not installed")

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        self._model = LogisticRegression(**self._params)
        self._model.fit(X_scaled, y)
        
        n_pos = np.sum(y == 1)
        logger.info("Model trained on %d samples (%d positive)", len(y), int(n_pos))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores with scaled features."""
        if self._model is None:
            # No trained model loaded: force caller to fall back to rule engine
            logger.error("RiskModel.predict called without a trained model; returning zeros.")
            return np.zeros(X.shape[0], dtype=float)

        X = np.asarray(X, dtype=np.float32)
        expected_features = getattr(self._scaler, "n_features_in_", None) or getattr(self._model, "n_features_in_", None)
        if expected_features is not None and X.shape[1] != int(expected_features):
            target = int(expected_features)
            if X.shape[1] > target:
                logger.warning("Feature dimension mismatch (got=%d expected=%d). Truncating extra features.", X.shape[1], target)
                X = X[:, :target]
            else:
                logger.warning("Feature dimension mismatch (got=%d expected=%d). Zero-padding missing features.", X.shape[1], target)
                pad = np.zeros((X.shape[0], target - X.shape[1]), dtype=X.dtype)
                X = np.hstack([X, pad])

        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    def save(self, directory: str, version: int = 1) -> str:
        """Save model and scaler to a versioned pickle file."""
        if self._model is None:
            raise RuntimeError("No model to save — train first")

        os.makedirs(directory, exist_ok=True)
        filename = f"risk_model_v{version}.pkl"
        path = os.path.join(directory, filename)
        
        bundle = {
            "model": self._model,
            "scaler": self._scaler,
            "version": version
        }
        
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        # Save metadata alongside
        meta_path = os.path.join(directory, f"risk_model_v{version}_meta.json")
        meta = {
            "version": version,
            "params": self._params,
            "format": "logistic_regression_bundle_v1",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Model bundle saved to %s", path)
        return path

    def load(self, path: str) -> "RiskModel":
        """Load a previously saved model bundle."""
        if not _ML_AVAILABLE:
            raise RuntimeError("scikit-learn not installed")

        if not os.path.exists(path):
            logger.warning("Model bundle not found: %s.", path)
            return self

        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            
            if isinstance(bundle, dict) and "model" in bundle:
                self._model = bundle["model"]
                self._scaler = bundle.get("scaler", StandardScaler())
                logger.info("Model bundle (v%s) loaded from %s", bundle.get("version"), path)
            else:
                # Direct model pickle (legacy)
                self._model = bundle
                self._scaler = StandardScaler() # Fallback
                logger.info("Legacy model loaded from %s (using default scaler)", path)
                
        except Exception as e:
            logger.error("Failed to load model: %s.", str(e))
            # Leave self._model as None so callers can detect missing ML cleanly.
        
        return self

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True binary labels

        Returns:
            Dict with roc_auc, precision_at_top5, confusion_matrix
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        probs = self.predict(X)
        preds = (probs >= 0.5).astype(int)

        # ROC-AUC
        auc = roc_auc_score(y, probs)

        # Precision @ Top 5%
        top_k = max(1, int(len(y) * 0.05))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_true = y[top_indices]
        prec_at_5 = float(np.sum(top_true == 1)) if top_k > 0 else 0
        if top_k > 0: prec_at_5 /= top_k

        # Confusion matrix
        cm = confusion_matrix(y, preds)

        return {
            "roc_auc": round(auc, 4),
            "precision_at_top5_pct": round(prec_at_5, 4),
            "confusion_matrix": cm.tolist(),
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Return feature importance (coefficients) if model is trained."""
        if self._model is None:
            return None
        return self._model.coef_[0]
