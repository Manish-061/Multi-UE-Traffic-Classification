#!/usr/bin/env python3
# Test Feature Extraction

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traffic_classifier.features.flow_features import FlowFeatureExtractor

class TestFlowFeatures(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        self.extractor = FlowFeatureExtractor()

        # Create sample flow data
        self.sample_data = pd.DataFrame({
            'Flow Duration': [1.5, 2.0, 0.5],
            'Total Fwd Packets': [10, 20, 5],
            'Total Backward Packets': [8, 15, 3],
            'Total Length of Fwd Packets': [1200, 2400, 600],
            'Total Length of Bwd Packets': [800, 1500, 300],
            'Flow Bytes/s': [1333.33, 1950.0, 1800.0],
            'Flow Packets/s': [12.0, 17.5, 16.0]
        })

    def test_basic_features(self):
        """Test basic feature extraction."""
        result = self.extractor.extract_basic_features(self.sample_data)

        # Check that new features are created
        self.assertIn('Total_Packets', result.columns)
        self.assertIn('Total_Bytes', result.columns)
        self.assertIn('Avg_Packet_Size', result.columns)

        # Check calculations
        self.assertEqual(result['Total_Packets'].iloc[0], 18)  # 10 + 8
        self.assertEqual(result['Total_Bytes'].iloc[0], 2000)  # 1200 + 800

    def test_feature_selection(self):
        """Test feature selection."""
        features = self.extractor.extract_all_features(self.sample_data)
        selected = self.extractor.select_features(features)

        # Should return numeric columns only
        self.assertTrue(all(selected.dtypes.apply(lambda x: np.issubdtype(x, np.number))))

        # Should store feature names
        self.assertIsNotNone(self.extractor.feature_columns)
        self.assertGreater(len(self.extractor.feature_columns), 0)

    def test_scaler_fitting(self):
        """Test scaler fitting and transformation."""
        features = self.extractor.extract_all_features(self.sample_data)
        selected = self.extractor.select_features(features)

        # Fit scaler
        self.extractor.fit_scaler(selected)
        self.assertTrue(self.extractor.is_fitted)

        # Transform features
        transformed = self.extractor.transform_features(selected)

        # Check that transformation preserves shape
        self.assertEqual(transformed.shape, selected.shape)

if __name__ == '__main__':
    unittest.main()
