# Model Evaluation Script
# Multi-UE Traffic Classification Project

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

# Import our modules
import sys
sys.path.append('src')

from traffic_classifier.models.xgb_classifier import MultiUETrafficClassifier
from traffic_classifier.evaluation.metrics import ModelEvaluator
from traffic_classifier.utils.logging import get_logger
from traffic_classifier.utils.io import save_results

logger = get_logger("evaluate_model")

def evaluate_model(model_path, test_data_path, output_dir="artifacts"):
    """Comprehensive model evaluation."""

    logger.info(f"Loading model from {model_path}")
    classifier = MultiUETrafficClassifier.load(model_path)

    logger.info(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # Prepare test data
    if 'Application_Class' not in test_df.columns:
        raise ValueError("Test data must have 'Application_Class' column")

    # Extract features (use same feature columns as trained model)
    from traffic_classifier.features.flow_features import FlowFeatureExtractor

    feature_extractor = FlowFeatureExtractor()
    test_features = feature_extractor.extract_all_features(test_df)

    # Use only features that were used during training
    if classifier.feature_names:
        available_features = [f for f in classifier.feature_names if f in test_features.columns]
        X_test = test_features[available_features]
        logger.info(f"Using {len(available_features)} features for evaluation")
    else:
        X_test = test_features.select_dtypes(include=[np.number])
        logger.warning("No feature names found in model. Using all numeric columns.")

    y_test = test_df['Application_Class']

    logger.info(f"Evaluating on {len(X_test)} test samples")

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Run comprehensive evaluation
    results = evaluator.evaluate_model(classifier, X_test, y_test, include_timing=True)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save detailed results
    results_file = output_path / "evaluation_results.json"
    save_results(results, results_file)

    # Generate confusion matrix plot
    try:
        confusion_plot_path = output_path / "confusion_matrix.png"
        evaluator.plot_confusion_matrix(y_test, classifier.predict(X_test), 
                                       save_path=str(confusion_plot_path))
        logger.info(f"Confusion matrix saved to {confusion_plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")

    # Print summary
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    basic_metrics = results['basic_metrics']
    print(f"Accuracy: {basic_metrics['accuracy']:.3f}")
    print(f"Macro F1: {basic_metrics['macro_f1']:.3f}")
    print(f"Micro F1: {basic_metrics['micro_f1']:.3f}")

    if 'top2_accuracy' in basic_metrics:
        print(f"Top-2 Accuracy: {basic_metrics['top2_accuracy']:.3f}")

    # Timing results
    if 'timing' in results:
        timing = results['timing']
        print(f"\nPerformance:")
        print(f"P95 Latency: {timing['single_prediction_p95_ms']:.2f} ms")
        print(f"P99 Latency: {timing['single_prediction_p99_ms']:.2f} ms")

    # Per-class metrics
    print(f"\nPer-Class F1 Scores:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"  {class_name:<15}: {metrics['f1_score']:.3f} ({metrics['support']} samples)")

    # QoS metrics
    if 'qos_metrics' in results:
        qos = results['qos_metrics']
        print(f"\nQoS-Aware Metrics:")
        print(f"Critical Class Accuracy: {qos['critical_class_accuracy']:.3f}")
        print(f"Priority-Weighted Accuracy: {qos['priority_weighted_accuracy']:.3f}")

    # Calibration metrics
    if 'calibration_metrics' in results:
        cal = results['calibration_metrics']
        print(f"\nCalibration Metrics:")
        print(f"Expected Calibration Error: {cal['expected_calibration_error']:.4f}")
        print(f"Average Confidence: {cal['average_confidence']:.3f}")

    print(f"\nDetailed results saved to: {results_file}")

    # Performance assessment
    macro_f1 = basic_metrics['macro_f1']
    if macro_f1 >= 0.80:
        print("\nEXCELLENT: Model meets performance targets!")
    elif macro_f1 >= 0.70:
        print("\nGOOD: Model performance is acceptable")
    else:
        print("\nNEEDS IMPROVEMENT: Consider retraining or feature engineering")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-UE Traffic Classifier")
    parser.add_argument("--model", default="models/traffic_classifier.joblib",
                       help="Path to trained model")
    parser.add_argument("--test-data", default="data/splits/test_data.csv",
                       help="Test data CSV file")
    parser.add_argument("--output-dir", default="artifacts",
                       help="Output directory for results")

    args = parser.parse_args()

    try:
        results = evaluate_model(args.model, args.test_data, args.output_dir)
        print("\nEvaluation complete!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    main()