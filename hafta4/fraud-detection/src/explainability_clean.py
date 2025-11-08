"""
Model Explainability - Clean Version
SHAP ve LIME için temiz implementasyon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import logging

# Safe imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Simple Model Explainer"""

    def __init__(self, model, X_train, feature_names=None, class_names=None, y_train=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names or ['Normal', 'Fraud']
        self.shap_explainer = None
        self.shap_values = None
        self.lime_explainer = None
        self._cached_permutation = None

    def initialize_shap(self, explainer_type='tree', **kwargs):
        """Initialize SHAP explainer with optional configuration."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP mevcut değil")
            return False

        try:
            background_size = int(kwargs.get('background_size', 50))
            if explainer_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(
                    self.model,
                    model_output=kwargs.get('model_output', 'probability'),
                )
            else:
                background = shap.sample(
                    self.X_train,
                    min(background_size, len(self.X_train)),
                )
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    background,
                    link=kwargs.get('link', 'logit'),
                )
            logger.info("SHAP explainer hazır")
            return True
        except Exception as exc:
            logger.error(f"SHAP hatası: {exc}")
            return False

    def compute_shap_values(self, X_test, max_samples=50, **kwargs):
        """Compute SHAP values with optional sampling limits."""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None, X_test

        sample_size = int(kwargs.get('max_samples', max_samples))
        X_sample = (
            X_test.head(sample_size)
            if hasattr(X_test, 'head')
            else X_test[:sample_size]
        )

        try:
            shap_kwargs = {}
            if 'max_evals' in kwargs:
                shap_kwargs['max_evals'] = kwargs['max_evals']
            self.shap_values = self.shap_explainer.shap_values(X_sample, **shap_kwargs)
            if isinstance(self.shap_values, list):
                # Fraud class for binary classification
                self.shap_values = self.shap_values[min(1, len(self.shap_values) - 1)]
            return self.shap_values, X_sample
        except Exception as exc:
            logger.error(f"SHAP computation error: {exc}")
            self.shap_values = None
            return None, X_sample

    def plot_shap_summary(self, X_test=None):
        if not SHAP_AVAILABLE or self.shap_values is None:
            logger.warning("SHAP values mevcut değil")
            return

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            X_test,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False,
        )
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show(block=False)
        plt.close()

    def plot_shap_waterfall(self, X_sample=None, index=0):
        if not SHAP_AVAILABLE or self.shap_values is None:
            logger.warning("SHAP mevcut değil, waterfall grafiği atlanıyor")
            return
        if X_sample is None or len(X_sample) == 0:
            logger.warning("Waterfall için örnek veri yok")
            return

        idx = min(index, len(X_sample) - 1)
        try:
            shap.plots.waterfall(self.shap_values[idx], show=False)
            plt.title(f'SHAP Waterfall - Instance {idx}')
            plt.tight_layout()
            plt.show(block=False)
            plt.close()
        except Exception as exc:
            logger.error(f"SHAP waterfall çizim hatası: {exc}")

    def initialize_lime(self):
        if not LIME_AVAILABLE:
            logger.warning("LIME mevcut değil")
            return False

        try:
            self.lime_explainer = LimeTabularExplainer(
                self.X_train.values if hasattr(self.X_train, 'values') else self.X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification',
            )
            logger.info("LIME explainer hazır")
            return True
        except Exception as exc:
            logger.error(f"LIME hatası: {exc}")
            self.lime_explainer = None
            return False

    def explain_instance_lime(self, X_test, instance_idx=0):
        if not LIME_AVAILABLE or self.lime_explainer is None:
            logger.warning("LIME mevcut değil")
            return None

        instance = X_test.iloc[instance_idx] if hasattr(X_test, 'iloc') else X_test[instance_idx]
        try:
            explanation = self.lime_explainer.explain_instance(
                instance.values if hasattr(instance, 'values') else instance,
                self.model.predict_proba,
                num_features=10,
            )
            explanation.as_pyplot_figure(label=1)
            plt.title(f'LIME Explanation - Instance {instance_idx}')
            plt.tight_layout()
            plt.show()
            return explanation
        except Exception as exc:
            logger.error(f"LIME explanation error: {exc}")
            return None

    def permutation_importance(self, X_test, y_test):
        try:
            perm_importance = permutation_importance(
                self.model,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
            )
            sorted_idx = np.argsort(perm_importance.importances_mean)[-10:]
            plt.figure(figsize=(10, 6))
            plt.barh(
                [self.feature_names[i] for i in sorted_idx],
                perm_importance.importances_mean[sorted_idx],
            )
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance (Permutation-based)')
            plt.tight_layout()
            plt.show()
            self._cached_permutation = perm_importance
            return perm_importance
        except Exception as exc:
            logger.error(f"Permutation importance error: {exc}")
            return None

    def global_feature_importance(self, X_sample, y_true=None):
        if SHAP_AVAILABLE and self.shap_values is not None:
            importances = np.abs(self.shap_values).mean(axis=0)
            return dict(zip(self.feature_names, importances))

        logger.info("SHAP bulunamadı, permutation importance ile devam ediliyor")
        labels = None
        features = None

        if y_true is not None:
            labels = np.asarray(y_true)
            features = X_sample
        elif self._cached_permutation is not None:
            features = None
        elif self.y_train is not None:
            labels = np.asarray(self.y_train)
            features = self.X_train
        else:
            logger.warning("Permutation importance için etiket bulunamadı, sonuç üretilemiyor")
            return {}

        if self._cached_permutation is None and features is not None:
            self._cached_permutation = self.permutation_importance(
                features,
                labels,
            )

        if self._cached_permutation is None:
            return {}
        return dict(
            zip(
                self.feature_names,
                self._cached_permutation.importances_mean,
            )
        )

    def analyze_fraud_patterns(self, X_sample, y_true):
        if X_sample is None or len(X_sample) == 0:
            return {}

        y_true = np.asarray(y_true)
        fraud_mask = y_true == 1
        normal_mask = y_true == 0

        fraud_means = np.asarray(
            X_sample[fraud_mask].mean(axis=0)
            if fraud_mask.any()
            else np.zeros(X_sample.shape[1])
        )
        normal_means = np.asarray(
            X_sample[normal_mask].mean(axis=0)
            if normal_mask.any()
            else np.zeros(X_sample.shape[1])
        )
        deltas = (fraud_means - normal_means).reshape(-1)

        if SHAP_AVAILABLE and self.shap_values is not None and fraud_mask.any():
            fraud_shap = np.abs(self.shap_values[fraud_mask]).mean(axis=0)
            fraud_shap = np.asarray(fraud_shap).reshape(-1)
            return {
                feature: {
                    'fraud_minus_normal': float(deltas[idx]),
                    'mean_abs_shap': float(fraud_shap[idx]),
                }
                for idx, feature in enumerate(self.feature_names)
            }

        return {
            feature: float(deltas[idx])
            for idx, feature in enumerate(self.feature_names)
        }


def demo_explainability():
    """Clean explainability demo"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    logger.info("Explainability demo başlatılıyor...")
    
    # Create dataset
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=6,
        weights=[0.9, 0.1], random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(8)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")
    
    # Create explainer
    explainer = ModelExplainer(
        model,
        X_train,
        feature_names,
        ['Normal', 'Fraud'],
        y_train=y_train,
    )
    
    # Try SHAP
    if explainer.initialize_shap():
        shap_values, X_sample = explainer.compute_shap_values(X_test)
        if shap_values is not None:
            explainer.plot_shap_summary(X_sample)
    
    # Try LIME
    if explainer.initialize_lime():
        explainer.explain_instance_lime(X_test, 0)
    
    # Permutation importance (always works)
    explainer.permutation_importance(X_test, y_test)
    
    print("✅ Explainability demo tamamlandı!")
    return explainer


if __name__ == "__main__":
    demo_explainability()