#!/usr/bin/env python3
"""Interactive demo helpers for the fraud detection pipeline."""

import sys
import warnings

sys.path.append('src')

from pipeline import FraudDetectionPipeline
from preprocessing import demo_preprocessing as preprocessing_demo
from evaluation import FraudEvaluator
from outlier_detection import OutlierDetector

try:
    from explainability_clean import ModelExplainer  # noqa: F401
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    ModelExplainer = None
    EXPLAINABILITY_AVAILABLE = False

warnings.filterwarnings('ignore')


def demo_preprocessing():
    print("\n" + "=" * 60)
    print("🔧 PREPROCESSING DEMO")
    print("=" * 60)
    return preprocessing_demo()


def demo_outlier_detection():
    print("\n" + "=" * 60)
    print("🎯 OUTLIER DETECTION DEMO")
    print("=" * 60)

    pipeline = FraudDetectionPipeline()
    pipeline.load_data(synthetic=True)
    pipeline.preprocess_data()

    detector = OutlierDetector(contamination=0.05)
    detector.fit_isolation_forest(pipeline.X_train_processed.values)
    labels, scores = detector.predict_isolation_forest(pipeline.X_test_processed.values)

    print(f"Test set size             : {len(labels)}")
    print(f"Detected fraud candidates : {int(labels.sum())}")
    print(f"Average anomaly score     : {scores.mean():.4f}")

    return labels, scores


def demo_evaluation():
    print("\n" + "=" * 60)
    print("📊 EVALUATION DEMO")
    print("=" * 60)

    pipeline = FraudDetectionPipeline()
    pipeline.run_full_pipeline(data_path=None, save_models=False)

    best_model = pipeline._find_best_model()
    evaluator = pipeline.evaluators[best_model]
    evaluator.print_evaluation_report()

    return evaluator.results


def demo_explainability():
    print("\n" + "=" * 60)
    print("🔍 EXPLAINABILITY DEMO")
    print("=" * 60)

    if not EXPLAINABILITY_AVAILABLE:
        print("⚠️  SHAP/LIME explainability opsiyonel paketler yüklenmediği için atlandı.")
        return None

    pipeline = FraudDetectionPipeline()
    pipeline.run_full_pipeline(data_path=None, save_models=False)
    pipeline.explain_models('random_forest')

    return pipeline.explainer


def demo_full_pipeline():
    """Run the full pipeline end-to-end and display key results."""
    print("\n" + "=" * 60)
    print("🚀 FULL PIPELINE DEMO")
    print("=" * 60)

    pipeline = FraudDetectionPipeline()
    print("Pipeline configuration loaded:")
    print(f"- Models: {list(pipeline.config.get('models', {}).keys())}")
    print(f"- Preprocessing: {pipeline.config.get('preprocessing', {}).get('scaling_method', 'default')}")
    print(f"- Evaluation thresholds: ROC-AUC >= {pipeline.config.get('evaluation', {}).get('min_roc_auc', 0.7)}")

    print("\n🏃‍♂️ Running full pipeline...")
    success = pipeline.run_full_pipeline(data_path=None, save_models=True)
    if not success:
        print("\n❌ Pipeline failed!")
        return None

    print("\n✅ Pipeline completed successfully!")
    best_model = pipeline._find_best_model()
    if best_model and best_model in pipeline.evaluators:
        results = pipeline.evaluators[best_model].results
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   ROC-AUC : {results['roc_auc']:.4f}")
        print(f"   PR-AUC  : {results['pr_auc']:.4f}")
        print(f"   F1      : {results['f1_score']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall   : {results['recall']:.4f}")

    print("\n🔮 Demo Predictions:")
    test_sample = pipeline.X_test.head(3)
    predictions, probabilities = pipeline.predict(test_sample, best_model)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label = 'FRAUD' if int(pred) == 1 else 'NORMAL'
        print(f"   Transaction {i+1}: {label} (Probability: {prob:.3f})")

    return pipeline


def interactive_demo():
    """Interactive demo menu"""
    while True:
        print("\n" + "=" * 60)
        print("🎭 FRAUD DETECTION INTERACTIVE DEMO")
        print("=" * 60)
        print("1. 🔧 Preprocessing Demo")
        print("2. 🎯 Outlier Detection Demo")
        print("3. 📊 Evaluation Demo")
        print("4. 🔍 Explainability Demo")
        print("5. 🚀 Full Pipeline Demo")
        print("6. ❌ Exit")
        print("-" * 60)

        choice = input("Enter your choice (1-6): ").strip()

        try:
            if choice == '1':
                demo_preprocessing()
            elif choice == '2':
                demo_outlier_detection()
            elif choice == '3':
                demo_evaluation()
            elif choice == '4':
                demo_explainability()
            elif choice == '5':
                demo_full_pipeline()
            elif choice == '6':
                print("👋 Demo finished. Thank you!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print("\n⏸️  Demo interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error in demo: {e}")
            print("Continuing to menu...")

        input("\nPress Enter to continue...")


def main():
    """Main function"""
    print("🎭 Welcome to Fraud Detection Demo!")
    print("This demo showcases all components of the fraud detection system.")
    
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        print("\n🔄 Running automated demo...")
        demo_full_pipeline()
    else:
        interactive_demo()


if __name__ == "__main__":
    main()