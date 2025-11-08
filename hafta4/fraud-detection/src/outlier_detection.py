"""Outlier detection utilities for fraud pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


@dataclass
class OutlierDetector:
    """Wrapper around Isolation Forest and LOF with score normalization."""

    contamination: float = 0.05
    n_neighbors: int = 20
    random_state: int = 42
    isolation_forest: Optional[IsolationForest] = field(default=None, init=False)
    lof: Optional[LocalOutlierFactor] = field(default=None, init=False)
    scaler: Optional[object] = field(default=None, init=False)  # populated by pipeline if needed

    def fit_isolation_forest(self, X: np.ndarray) -> IsolationForest:
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X)
        return self.isolation_forest

    def predict_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest model is not fitted")
        raw_scores = -self.isolation_forest.decision_function(X)
        scores = self._normalize_scores(raw_scores)
        labels = (scores >= self._default_threshold(scores)).astype(int)
        return labels, scores

    def fit_lof(self, X: np.ndarray) -> LocalOutlierFactor:
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
        )
        self.lof.fit(X)
        return self.lof

    def predict_lof(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lof is None:
            raise ValueError("Local Outlier Factor model is not fitted")
        raw_scores = -self.lof.score_samples(X)
        scores = self._normalize_scores(raw_scores)
        labels = (scores >= self._default_threshold(scores)).astype(int)
        return labels, scores

    def evaluate_performance(self, y_true: np.ndarray, scores: np.ndarray) -> dict:
        metrics = {
            "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan,
            "pr_auc": average_precision_score(y_true, scores),
        }
        precision, recall, thresh = precision_recall_curve(y_true, scores)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        idx = f1.argmax()
        metrics.update(
            {
                "precision": precision[idx],
                "recall": recall[idx],
                "f1": f1[idx],
                "threshold": thresh[idx - 1] if idx > 0 and len(thresh) > 0 else 0.5,
            }
        )
        return metrics

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_val = scores.min()
        max_val = scores.max()
        if np.isclose(min_val, max_val):
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    @staticmethod
    def _default_threshold(scores: np.ndarray) -> float:
        """Select a threshold at the 95th percentile by default."""
        return float(np.quantile(scores, 0.95))
