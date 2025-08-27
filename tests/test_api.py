# Test API Endpoints

import unittest
import sys
import os
import json
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from fastapi.testclient import TestClient
    from traffic_classifier.serving.api import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestAPI(unittest.TestCase):

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(app.router.startup())

        # Sample flow data
        self.sample_flows = [
            {
                "Flow Duration": 1.5,
                "Total Fwd Packets": 10,
                "Total Backward Packets": 8,
                "Flow Bytes/s": 1333.33,
                "Flow Packets/s": 12.0
            },
            {
                "Flow Duration": 2.0,
                "Total Fwd Packets": 20,
                "Total Backward Packets": 15,
                "Flow Bytes/s": 1950.0,
                "Flow Packets/s": 17.5
            }
        ]

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("uptime_seconds", data)

    def test_classes_endpoint(self):
        """Test classes endpoint."""
        response = self.client.get("/classes")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("application_classes", data)
        self.assertIn("qos_mapping", data)
        self.assertEqual(len(data["application_classes"]), 7)

    def test_predict_endpoint(self):
        """Test prediction endpoint."""
        request_data = {
            "flows": self.sample_flows,
            "ue_id": "test_ue",
            "include_probabilities": True
        }

        response = self.client.post("/predict", json=request_data)

        # Should work even without trained model (uses heuristics)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("predictions", data)
        self.assertIn("processing_time_ms", data)
        self.assertEqual(data["ue_id"], "test_ue")
        self.assertEqual(len(data["predictions"]), 2)

        # Check prediction structure
        prediction = data["predictions"][0]
        self.assertIn("predicted_class", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("qos_priority", prediction)

    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = self.client.get("/model/info")

        # Should return 503 if no model is loaded, or 200 if model exists
        self.assertIn(response.status_code, [200, 503])

        if response.status_code == 200:
            data = response.json()
            self.assertIn("model_type", data)
            self.assertIn("is_trained", data)

if __name__ == '__main__':
    unittest.main()
