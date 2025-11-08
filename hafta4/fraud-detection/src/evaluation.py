"""
Fraud Detection Model Evaluation
Basit evaluation sınıfı
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
)


class FraudEvaluator:
    """Model evaluation için basit sınıf"""
    
    def __init__(self, model=None, model_name="Model"):
        self.model = model
        self.model_name = model_name
        self.results = {}
    
    def evaluate_binary_classification(self, X_test, y_true, y_pred_proba=None, y_pred=None, threshold=0.5):
        """Binary classification evaluation"""

        if y_pred_proba is None:
            if self.model is None:
                raise ValueError("Model or prediction probabilities must be provided")
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = np.asarray(y_pred)

        results = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(set(y_true)) > 1 else float('nan'),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, zero_division=0, output_dict=False),
        }

        self.results = results
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        return results

    def print_evaluation_report(self):
        if not self.results:
            print("⚠️  No evaluation results available")
            return

        print(f"\n=== Evaluation Report: {self.model_name} ===")
        print(f"ROC-AUC      : {self.results.get('roc_auc', float('nan')):.4f}")
        print(f"PR-AUC       : {self.results.get('pr_auc', float('nan')):.4f}")
        print(f"F1 Score     : {self.results.get('f1_score', float('nan')):.4f}")
        print(f"Precision    : {self.results.get('precision', float('nan')):.4f}")
        print(f"Recall       : {self.results.get('recall', float('nan')):.4f}")
        print("Confusion Matrix:")
        cm = self.results.get('confusion_matrix')
        if cm is not None:
            print(np.array(cm))
        print("Classification Report:")
        print(self.results.get('classification_report', 'N/A'))

    def roc_curve_points(self):
        if not hasattr(self, 'y_true') or not hasattr(self, 'y_pred_proba'):
            return None
        return roc_curve(self.y_true, self.y_pred_proba)

    def precision_recall_points(self):
        if not hasattr(self, 'y_true') or not hasattr(self, 'y_pred_proba'):
            return None
        return precision_recall_curve(self.y_true, self.y_pred_proba)