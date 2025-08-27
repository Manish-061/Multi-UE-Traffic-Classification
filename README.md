# Multi-UE Traffic Classification

AI-powered network traffic classification system for Quality of Service (QoS) optimization in multi-User Equipment (UE) scenarios.

## 🎯 Features

- **7-Class Classification**: Gaming, Audio Calls, Video Calls, Video Streaming, Browsing, Video Uploads, Texting
- **Real-time Inference**: <20ms latency for network control integration
- **QoS Integration**: Direct mapping to 5G QCI values and delay budgets  
- **UE-aware Processing**: Handles multiple concurrent users
- **Calibrated Probabilities**: Reliable confidence scores for policy decisions

## 🚀 Quick Start

```bash
# Setup project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Generate synthetic data  
python scripts/generate_synthetic_data.py

# Train XGBoost model
python scripts/train_model.py --config configs/model_xgb.yaml

# Evaluate model performance
python scripts/evaluate_model.py --model models/traffic_classifier.joblib

# Start API server
python scripts/serve_api.py --port 8000

# Run demo prediction
python -c "from src.traffic_classifier.serving.cli import demo; demo()"
```

## 📊 Performance Targets

- **Macro F1-Score**: ≥ 0.80
- **Per-Class F1**: ≥ 0.70 (all classes)
- **Inference Latency**: P95 ≤ 20ms  
- **Memory Usage**: ≤ 512MB

## 🏗️ Architecture

The system processes network flows through feature extraction, XGBoost classification, and probability calibration to provide real-time traffic classification with QoS policy recommendations.

## 🔧 API Usage

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "flows": [{"Flow Duration": 1.5, "Total Fwd Packets": 10, ...}],
    "ue_id": "UE_001"  
})

predictions = response.json()["predictions"]
```

## 📚 Project Structure

```
multi_ue_traffic_classifier/
├── src/traffic_classifier/     # Core package
├── scripts/                    # Automation scripts
├── configs/                    # YAML configurations
├── data/                      # Datasets (raw/processed/splits)
├── models/                    # Trained models
├── notebooks/                 # Jupyter notebooks for analysis
├── dashboards/                # Streamlit dashboard
└── tests/                     # Unit tests
```

## 🎮 Application Classes & QoS Mapping

| Class | Priority | QCI | Delay Budget | Use Case |
|-------|----------|-----|--------------|----------|
| Gaming | 1 | 3 | 50ms | Ultra-low latency |
| Audio Calls | 2 | 1 | 100ms | Voice quality |
| Video Calls | 3 | 2 | 150ms | Video conferencing |
| Video Streaming | 4 | 6/7 | 300ms | Entertainment |
| Browsing | 5 | 6/8 | 300ms | Web traffic |
| Video Uploads | 6 | 8/9 | 300ms | Content creation |
| Texting | 7 | 9 | 300ms | Messaging |

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_models.py -v
```

## 📈 Monitoring

- Streamlit dashboard: `streamlit run dashboards/streamlit_app.py`
- API health: `curl http://localhost:8000/health`
- Model metrics: Check `artifacts/metrics/`

## 📄 License

MIT License - see LICENSE file for details.
