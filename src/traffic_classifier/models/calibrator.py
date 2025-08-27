"""Probability calibration for reliable confidence scores."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

class ProbabilityCalibrator:
    """Calibrate model probabilities for reliable confidence scores."""

    def __init__(self, base_classifier, method: str = 'platt', cv: int = 5):
        """Initialize calibrator.

        Args:
            base_classifier: Base classifier to calibrate
            method: Calibration method ('platt' or 'isotonic')
            cv: Number of cross-validation folds
        """
        self.base_classifier = base_classifier
        self.method = method
        self.cv = cv
        self.calibrated_classifier = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ProbabilityCalibrator':
        """Fit calibrator on validation data.

        Args:
            X: Validation features
            y: Validation labels

        Returns:
            Self for method chaining
        """
        print(f"Calibrating probabilities using {self.method} method...")

        self.calibrated_classifier = CalibratedClassifierCV(
            self.base_classifier, 
            method=self.method, 
            cv=self.cv
        )

        self.calibrated_classifier.fit(X, y)
        self.is_fitted = True

        print("Probability calibration complete")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Calibrated probability array
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")

        return self.calibrated_classifier.predict_proba(X)

    def evaluate_calibration(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate calibration quality.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Calibration evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before evaluation")

        # Get probabilities
        y_proba_uncalibrated = self.base_classifier.predict_proba(X)
        y_proba_calibrated = self.calibrated_classifier.predict_proba(X)

        # Convert labels to binary format for Brier score
        from sklearn.preprocessing import label_binarize

        unique_labels = np.unique(y)
        y_binary = label_binarize(y, classes=unique_labels)

        # Calculate Brier scores
        brier_uncalibrated = brier_score_loss(y_binary.ravel(), y_proba_uncalibrated.ravel())
        brier_calibrated = brier_score_loss(y_binary.ravel(), y_proba_calibrated.ravel())

        # Expected Calibration Error (ECE)
        ece_uncalibrated = self._calculate_ece(y, y_proba_uncalibrated, unique_labels)
        ece_calibrated = self._calculate_ece(y, y_proba_calibrated, unique_labels)

        evaluation_results = {
            'brier_score_uncalibrated': float(brier_uncalibrated),
            'brier_score_calibrated': float(brier_calibrated),
            'brier_score_improvement': float(brier_uncalibrated - brier_calibrated),
            'ece_uncalibrated': float(ece_uncalibrated),
            'ece_calibrated': float(ece_calibrated),
            'ece_improvement': float(ece_uncalibrated - ece_calibrated)
        }

        return evaluation_results

    def _calculate_ece(self, y_true: pd.Series, y_proba: np.ndarray, 
                      classes: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            classes: Unique class labels
            n_bins: Number of bins for calibration

        Returns:
            Expected Calibration Error
        """
        # Get predicted classes and confidence scores
        y_pred_classes = classes[np.argmax(y_proba, axis=1)]
        confidences = np.max(y_proba, axis=1)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                accuracy_in_bin = (y_true[in_bin] == y_pred_classes[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def plot_reliability_diagram(self, X: pd.DataFrame, y: pd.Series, 
                                save_path: Optional[str] = None) -> None:
        """Plot reliability diagram for calibration assessment.

        Args:
            X: Test features
            y: Test labels
            save_path: Optional path to save plot
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before plotting")

        # Get probabilities
        y_proba_uncalibrated = self.base_classifier.predict_proba(X)
        y_proba_calibrated = self.calibrated_classifier.predict_proba(X)

        # Get predicted classes and confidence scores
        unique_labels = np.unique(y)

        confidences_uncal = np.max(y_proba_uncalibrated, axis=1)
        pred_classes_uncal = unique_labels[np.argmax(y_proba_uncalibrated, axis=1)]

        confidences_cal = np.max(y_proba_calibrated, axis=1)
        pred_classes_cal = unique_labels[np.argmax(y_proba_calibrated, axis=1)]

        # Create reliability diagrams
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        self._plot_single_reliability_diagram(
            y, pred_classes_uncal, confidences_uncal, 
            ax1, "Uncalibrated"
        )

        self._plot_single_reliability_diagram(
            y, pred_classes_cal, confidences_cal, 
            ax2, "Calibrated"
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reliability diagram saved to {save_path}")

        plt.show()

    def _plot_single_reliability_diagram(self, y_true: pd.Series, y_pred: np.ndarray,
                                       confidences: np.ndarray, ax, title: str,
                                       n_bins: int = 10) -> None:
        """Plot single reliability diagram.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidences: Confidence scores
            ax: Matplotlib axis
            title: Plot title
            n_bins: Number of bins
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                bin_centers.append(avg_confidence_in_bin)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(in_bin.sum())

        # Plot
        ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
               edgecolor='black', linewidth=1)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Calibration')

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{title} Reliability Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
