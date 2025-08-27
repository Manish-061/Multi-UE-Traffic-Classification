# 🚀 Multi-UE Traffic Classifier: API Reference

Welcome to the API for the Multi-UE Traffic Classifier! This guide provides detailed information on how to interact with the available endpoints. Our API is built with FastAPI, ensuring high performance and automatic interactive documentation.

**Base URL**: `http://localhost:8000`

**Interactive Docs**:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🩺 Health Check

Check the operational status of the API and the loaded model.

- **Endpoint**: `GET /health`
- **Success Response**: `200 OK`

### Response Body

A JSON object indicating the API's health.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 120.5,
  "version": "1.0.0",
  "timestamp": 1678886400.0
}
```

### Example Usage

**cURL**
```bash
curl -X GET http://localhost:8000/health
```

**Python (`requests`)**
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

---

## 🔮 Predict Traffic Class

This is the core endpoint for classifying network flows. It accepts a list of flows and returns predictions for each.

- **Endpoint**: `POST /predict`
- **Success Response**: `200 OK`
- **Error Responses**:
    - `400 Bad Request`: If the input data is invalid or missing.
    - `503 Service Unavailable`: If the model is not loaded.

### Request Body

A JSON object containing a list of flows and a UE identifier.

```json
{
  "flows": [
    {
      "Flow Duration": 1500000,
      "Total Fwd Packets": 10,
      "Total Backward Packets": 8,
      "Total Length of Fwd Packets": 1200,
      "Total Length of Bwd Packets": 980,
      "Flow Bytes/s": 1453.3,
      "Flow Packets/s": 12.0,
      "Flow IAT Mean": 187500.0,
      "Flow IAT Std": 25000.0
    }
  ],
  "ue_id": "user_phone_123",
  "include_probabilities": true
}
```

### Response Body

Returns the predicted class, confidence, QoS parameters, and (optionally) the full probability distribution for each flow.

```json
{
  "ue_id": "user_phone_123",
  "predictions": [
    {
      "predicted_class": "video_streaming",
      "confidence": 0.92,
      "probabilities": {
        "gaming": 0.01,
        "audio_calls": 0.01,
        "video_calls": 0.03,
        "video_streaming": 0.92,
        "browsing": 0.01,
        "video_uploads": 0.01,
        "texting": 0.01
      },
      "qos_priority": 4,
      "qci": 6,
      "delay_budget_ms": 300
    }
  ],
  "processing_time_ms": 15.7,
  "model_version": "1.0.0",
  "timestamp": 1678886500.0
}
```

### Example Usage

**cURL**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "flows": [{"Flow Duration": 1500000, "Total Fwd Packets": 10}],
  "ue_id": "user_phone_123"
}'
```

**Python (`requests`)**
```python
import requests

flow_data = {
    "flows": [{
        "Flow Duration": 1500000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 8,
        "Total Length of Fwd Packets": 1200,
        "Total Length of Bwd Packets": 980
    }],
    "ue_id": "user_phone_123"
}

response = requests.post("http://localhost:8000/predict", json=flow_data)
print(response.json())
```

---

## ℹ️ Get Model Information

Retrieve details about the currently loaded classification model.

- **Endpoint**: `GET /model/info`
- **Success Response**: `200 OK`

### Response Body

```json
{
  "model_type": "XGBoost Classifier",
  "is_trained": true,
  "classes": [
    "gaming", "audio_calls", "video_calls", "video_streaming", 
    "browsing", "video_uploads", "texting"
  ],
  "feature_count": 50,
  "version": "1.0.0"
}
```

### Example Usage

**cURL**
```bash
curl -X GET http://localhost:8000/model/info
```

---

## 📊 Get API Metrics

Get internal metrics for monitoring the API's performance and state.

- **Endpoint**: `GET /metrics`
- **Success Response**: `200 OK`

### Response Body

```json
{
  "uptime_seconds": 3600.0,
  "model_loaded": true,
  "active_sessions": 150,
  "api_version": "1.0.0",
  "status": "healthy"
}
```

### Example Usage

**cURL**
```bash
curl -X GET http://localhost:8000/metrics
```
