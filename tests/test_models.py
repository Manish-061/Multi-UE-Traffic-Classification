#!/usr/bin/env python3
# Test Model Training and Prediction

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traffic_classifier.models.xgb_classifier import MultiUETrafficClassifier

class TestMultiUEClassifier(unittest.TestCase):

    def setUp(self):
        """Set up test data and classifier."""
        self.classifier = MultiUETrafficClassifier(n_estimators=10, random_state=42)

        # Create synthetic training data
        np.random.seed(42)
        n_samples = 100

        self.X_train = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples)
        })

        # Create labels with some structure
        labels = []
        for i in range(n_samples):
            if self.X_train.iloc[i]['feature_1'] > 0.5:
                labels.append('gaming')
            elif self.X_train.iloc[i]['feature_2'] > 0.3:
                labels.append('video_streaming')
            else:
                labels.append('browsing')

        self.y_train = pd.Series(labels)

        # Test data
        self.X_test = self.X_train.iloc[:20].copy()
        self.y_test = self.y_train.iloc[:20].copy()

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertFalse(self.classifier.is_trained)
        self.assertEqual(len(self.classifier.classes), 7)  # Default classes
        self.assertIsNone(self.classifier.calibrator)

    def test_model_training(self):
        """Test model training."""
        results = self.classifier.train(self.X_train, self.y_train)

        # Check training results
        self.assertIsInstance(results, dict)
        self.assertIn('training_time_seconds', results)
        self.assertIn('n_features', results)
        self.assertTrue(self.classifier.is_trained)

        # Check feature names are stored
        self.assertEqual(len(self.classifier.feature_names), self.X_train.shape[1])

    def test_model_prediction(self):
        """Test model prediction."""
        # Train first
        self.classifier.train(self.X_train, self.y_train)

        # Make predictions
        predictions = self.classifier.predict(self.X_test)

        # Check predictions
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in ['gaming', 'video_streaming', 'browsing'] for pred in predictions))

    def test_model_probabilities(self):
        """Test probability predictions."""
        # Train first
        self.classifier.train(self.X_train, self.y_train)

        # Get probabilities
        probabilities, class_names = self.classifier.predict_proba(self.X_test)

        # Check probabilities
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertEqual(probabilities.shape[1], len(class_names))

        # Probabilities should sum to 1
        prob_sums = np.sum(probabilities, axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(self.X_test)), decimal=5)

    def test_model_evaluation(self):
        """Test model evaluation."""
        # Train first
        self.classifier.train(self.X_train, self.y_train)

        # Evaluate
        results = self.classifier.evaluate(self.X_test, self.y_test)

        # Check evaluation results
        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)
        self.assertIn('macro_f1', results)
        self.assertIn('per_class_metrics', results)

        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(results['accuracy'], 0.0)
        self.assertLessEqual(results['accuracy'], 1.0)

if __name__ == '__main__':
    unittest.main()
