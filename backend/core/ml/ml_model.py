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

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default Logistic Regression hyperparameters
_DEFAULT_PARAMS: Dict[str, Any] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced",
}


class RiskModel:
    """
    Logistic Regression-based risk scoring model.

    Usage:
        model = RiskModel()
        model.train(X_train, y_train)
        probs = model.predict(X_test)
        model.save("core/ml/models", version=1)

        # Later...
        model = RiskModel()
        model.load("core/ml/models/risk_model_v1.pkl")
        probs = model.predict(X_new)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with optional hyperparameter overrides."""
        if not _ML_AVAILABLE:
            logger.warning(
                "scikit-learn not installed. "
                "ML scoring will be unavailable."
            )
            self._model = None
            return

        self._params = {**_DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[LogisticRegression] = None

    @property
    def is_available(self) -> bool:
        """Check if the ML libraries are installed."""
        return _ML_AVAILABLE

    @property
    def is_trained(self) -> bool:
        """Check if a model has been trained or loaded."""
        return self._model is not None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> "RiskModel":
        """
        Train the Logistic Regression classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary labels of shape (n_samples,)  — 1=suspicious, 0=clean
            eval_set: Ignored (not used by LogisticRegression)

        Returns:
            self (for chaining)
        """
        if not _ML_AVAILABLE:
            raise RuntimeError("scikit-learn not installed")

        self._model = LogisticRegression(**self._params)
        self._model.fit(X, y)
        
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        logger.info(
            "Model trained on %d samples (%d positive, %d negative)",
            len(y),
            int(n_pos),
            int(n_neg),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of being suspicious (class 1).

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray of shape (n_samples,) with values in [0, 1]
        """
        if self._model is None:
            # Fallback for inference without trained model: simple weighting of features
            # This ensures the pipeline doesn't break if no model is loaded.
            logger.warning("Model not trained or loaded. Using fallback linear combination.")
            # Weighted average of first 20 binary features (patterns)
            # Multiplier of 3.0 ensures that ~7 patterns (out of 20) results in 100% risk.
            return np.clip(np.mean(X[:, :20], axis=1) * 3.0, 0, 1)
            
        return self._model.predict_proba(X)[:, 1]

    def save(self, directory: str, version: int = 1) -> str:
        """
        Save model to a versioned pickle file.

        Args:
            directory: Output directory (created if not exists)
            version: Model version number

        Returns:
            Full path to the saved model file
        """
        if self._model is None:
            raise RuntimeError("No model to save — train first")

        os.makedirs(directory, exist_ok=True)
        filename = f"risk_model_v{version}.pkl"
        path = os.path.join(directory, filename)
        
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

        # Save metadata alongside the model
        meta_path = os.path.join(directory, f"risk_model_v{version}_meta.json")
        meta = {
            "version": version,
            "params": self._params,
            "format": "logistic_regression_pickle",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Model saved to %s", path)
        return path

    def load(self, path: str) -> "RiskModel":
        """
        Load a previously saved model.

        Args:
            path: Path to the .pkl model file (or .json if legacy)

        Returns:
            self (for chaining)
        """
        if not _ML_AVAILABLE:
            raise RuntimeError("scikit-learn not installed")

        if not os.path.exists(path):
            logger.warning("Model file not found: %s. Using fallback.", path)
            return self

        try:
            if path.endswith(".json"):
                 logger.warning("Legacy XGBoost JSON model cannot be loaded by LogisticRegression. Using fallback.")
                 return self
                 
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            logger.info("Model loaded from %s", path)
        except Exception as e:
            logger.error("Failed to load model: %s. Using fallback.", str(e))
        
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
