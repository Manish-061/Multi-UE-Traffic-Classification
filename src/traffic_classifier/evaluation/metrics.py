"""Comprehensive model evaluation and metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import time

class ModelEvaluator:
    """Comprehensive evaluation for traffic classification models."""

    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize evaluator.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names or [
            'gaming', 'audio_calls', 'video_calls', 'video_streaming',
            'browsing', 'video_uploads', 'texting'
        ]

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      include_timing: bool = True) -> Dict[str, Any]:
        """Comprehensive model evaluation.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            include_timing: Whether to include timing measurements

        Returns:
            Comprehensive evaluation results
        """
        results = {}

        # Timing measurements
        if include_timing:
            timing_results = self._measure_inference_timing(model, X_test)
            results['timing'] = timing_results

        # Predictions
        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_proba, model_class_names = model.predict_proba(X_test)
            class_names = np.array(model_class_names)
            results['probabilities_available'] = True
        else:
            y_proba = None
            class_names = np.array(self.class_names)
            results['probabilities_available'] = False

        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(y_test, y_pred, y_proba, class_names)
        results['basic_metrics'] = basic_metrics

        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(y_test, y_pred)
        results['per_class_metrics'] = per_class_metrics

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.class_names)
        results['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'class_names': self.class_names
        }

        # QoS-aware metrics
        qos_metrics = self._calculate_qos_metrics(y_test, y_pred)
        results['qos_metrics'] = qos_metrics

        # Calibration metrics (if probabilities available)
        if y_proba is not None:
            calibration_metrics = self._calculate_calibration_metrics(y_test, y_proba, class_names)
            results['calibration_metrics'] = calibration_metrics

        return results

    def _measure_inference_timing(self, model, X_test: pd.DataFrame, 
                                n_runs: int = 100) -> Dict[str, float]:
        """Measure model inference timing.

        Args:
            model: Trained model
            X_test: Test features
            n_runs: Number of timing runs

        Returns:
            Timing statistics
        """
        # Warm up
        for _ in range(5):
            _ = model.predict(X_test[:1])

        # Single prediction timing
        single_times = []
        for _ in range(n_runs):
            start_time = time.time()
            _ = model.predict(X_test[:1])
            end_time = time.time()
            single_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Batch prediction timing
        batch_times = []
        batch_sizes = [1, 10, 100, min(1000, len(X_test))]

        for batch_size in batch_sizes:
            if batch_size <= len(X_test):
                batch_data = X_test[:batch_size]
                times = []

                for _ in range(min(10, n_runs)):
                    start_time = time.time()
                    _ = model.predict(batch_data)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)

                batch_times.append({
                    'batch_size': batch_size,
                    'mean_time_ms': np.mean(times),
                    'p95_time_ms': np.percentile(times, 95),
                    'throughput_samples_per_sec': batch_size / (np.mean(times) / 1000)
                })

        return {
            'single_prediction_mean_ms': np.mean(single_times),
            'single_prediction_p95_ms': np.percentile(single_times, 95),
            'single_prediction_p99_ms': np.percentile(single_times, 99),
            'batch_timing': batch_times
        }

    def _calculate_basic_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray], 
                               class_names: List[str]) -> Dict[str, float]:
        """Calculate basic classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            class_names: Class names

        Returns:
            Basic metrics dictionary
        """
        metrics = {}

        # Accuracy
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        # F1 scores
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=self.class_names, zero_division=0
        )

        metrics['macro_f1'] = float(np.mean(f1))
        metrics['micro_f1'] = float(accuracy_score(y_true, y_pred))  # For multiclass
        metrics['weighted_f1'] = float(np.average(f1, weights=support))

        # Top-K accuracy
        if y_proba is not None:
            metrics['top2_accuracy'] = self._calculate_topk_accuracy(y_true, y_proba, class_names, k=2)
            metrics['top3_accuracy'] = self._calculate_topk_accuracy(y_true, y_proba, class_names, k=3)

        return metrics

    def _calculate_per_class_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Per-class metrics dictionary
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=self.class_names, zero_division=0
        )

        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(precision):
                per_class_metrics[class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i]) if i < len(support) else 0
                }

        return per_class_metrics

    def _calculate_qos_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate QoS-aware metrics based on traffic priorities.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            QoS metrics dictionary
        """
        # QoS priority mapping
        qos_priorities = {
            'gaming': 1, 'audio_calls': 2, 'video_calls': 3,
            'video_streaming': 4, 'browsing': 5, 'video_uploads': 6, 'texting': 7
        }

        # Critical classes (priority 1-3)
        critical_classes = ['gaming', 'audio_calls', 'video_calls']

        # Calculate metrics for critical classes
        critical_mask = y_true.isin(critical_classes)

        if critical_mask.sum() > 0:
            critical_accuracy = accuracy_score(y_true[critical_mask], y_pred[critical_mask])

            # Critical class recall (important for QoS)
            critical_recall = {}
            for class_name in critical_classes:
                class_mask = y_true == class_name
                if class_mask.sum() > 0:
                    critical_recall[class_name] = float(
                        (y_pred[class_mask] == class_name).mean()
                    )
        else:
            critical_accuracy = 0.0
            critical_recall = {}

        # Priority-weighted accuracy
        priority_weights = np.array([1.0 / qos_priorities.get(label, 7) for label in y_true])
        weighted_correct = (y_true == y_pred) * priority_weights
        priority_weighted_accuracy = weighted_correct.sum() / priority_weights.sum()

        return {
            'critical_class_accuracy': float(critical_accuracy),
            'critical_class_recall': critical_recall,
            'priority_weighted_accuracy': float(priority_weighted_accuracy),
            'critical_classes': critical_classes
        }

    def _calculate_calibration_metrics(self, y_true: pd.Series, y_proba: np.ndarray,
                                     class_names: np.ndarray) -> Dict[str, float]:
        """Calculate probability calibration metrics.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            class_names: Class names

        Returns:
            Calibration metrics dictionary
        """
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_proba, class_names)

        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_true, y_proba, class_names)

        # Reliability (average confidence when correct)
        y_pred_classes = class_names[np.argmax(y_proba, axis=1)]
        confidences = np.max(y_proba, axis=1)
        correct_mask = y_true == y_pred_classes

        reliability = float(confidences[correct_mask].mean()) if correct_mask.sum() > 0 else 0.0

        return {
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'reliability': reliability,
            'average_confidence': float(confidences.mean())
        }

    def _calculate_ece(self, y_true: pd.Series, y_proba: np.ndarray,
                      class_names: List[str], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        y_pred_classes = class_names[np.argmax(y_proba, axis=1)]
        confidences = np.max(y_proba, axis=1)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (y_true[in_bin] == y_pred_classes[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _calculate_mce(self, y_true: pd.Series, y_proba: np.ndarray,
                      class_names: List[str], n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        y_pred_classes = class_names[np.argmax(y_proba, axis=1)]
        confidences = np.max(y_proba, axis=1)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (y_true[in_bin] == y_pred_classes[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)

        return max_error

    def _calculate_topk_accuracy(self, y_true: pd.Series, y_proba: np.ndarray,
                               class_names: List[str], k: int = 2) -> float:
        """Calculate top-k accuracy."""
        correct = 0
        class_names_np = np.array(class_names)

        for i, true_label in enumerate(y_true):
            if i < len(y_proba):
                sample_proba = y_proba[i]
                top_k_indices = np.argsort(sample_proba)[::-1][:k]
                top_k_predictions = [class_names_np[idx] for idx in top_k_indices]

                if true_label in top_k_predictions:
                    correct += 1

        return correct / len(y_true) if len(y_true) > 0 else 0.0

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray,
                            save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save plot
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Multi-UE Traffic Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()
