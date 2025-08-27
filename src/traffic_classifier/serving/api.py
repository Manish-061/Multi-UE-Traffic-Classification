"""FastAPI application for real-time traffic classification."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import time
import logging
from pathlib import Path

# Import our classifier
from ..models.xgb_classifier import MultiUETrafficClassifier
from ..features.window_features import SlidingWindowFeatures
from ..utils.logging import get_logger

# Setup logging
logger = get_logger("traffic_classifier_api")

# Initialize FastAPI app
app = FastAPI(
    title="Multi-UE Traffic Classifier API",
    description="Real-time network traffic classification for 5G QoS optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and state
classifier = None
sliding_window = None
model_loaded = False

# Request/Response models
class FlowData(BaseModel):
    """Single flow data model."""
    flow_duration: Optional[float] = Field(None, alias="Flow Duration")
    total_fwd_packets: Optional[int] = Field(None, alias="Total Fwd Packets")
    total_backward_packets: Optional[int] = Field(None, alias="Total Backward Packets")
    total_length_fwd: Optional[float] = Field(None, alias="Total Length of Fwd Packets")
    total_length_bwd: Optional[float] = Field(None, alias="Total Length of Bwd Packets")
    flow_bytes_s: Optional[float] = Field(None, alias="Flow Bytes/s")
    flow_packets_s: Optional[float] = Field(None, alias="Flow Packets/s")
    flow_iat_mean: Optional[float] = Field(None, alias="Flow IAT Mean")
    flow_iat_std: Optional[float] = Field(None, alias="Flow IAT Std")

    class Config:
        allow_population_by_field_name = True

class PredictionRequest(BaseModel):
    """Prediction request model."""
    flows: List[Dict[str, Any]]
    ue_id: str = "default"
    include_probabilities: bool = True
    confidence_threshold: float = 0.7

class FlowPrediction(BaseModel):
    """Single flow prediction."""
    predicted_class: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    qos_priority: int
    qci: int
    delay_budget_ms: int

class PredictionResponse(BaseModel):
    """Prediction response model."""
    ue_id: str
    predictions: List[FlowPrediction]
    processing_time_ms: float
    model_version: str = "1.0.0"
    timestamp: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str
    timestamp: float

# QoS mapping
QOS_MAPPING = {
    'gaming': {'priority': 1, 'qci': 3, 'delay_budget': 50},
    'audio_calls': {'priority': 2, 'qci': 1, 'delay_budget': 100},
    'video_calls': {'priority': 3, 'qci': 2, 'delay_budget': 150},
    'video_streaming': {'priority': 4, 'qci': 6, 'delay_budget': 300},
    'browsing': {'priority': 5, 'qci': 8, 'delay_budget': 300},
    'video_uploads': {'priority': 6, 'qci': 9, 'delay_budget': 300},
    'texting': {'priority': 7, 'qci': 9, 'delay_budget': 300}
}

# Startup time
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global classifier, sliding_window, model_loaded

    logger.info("Starting Multi-UE Traffic Classifier API...")
    logger.debug(f"Initial model_loaded: {model_loaded}, classifier: {classifier is not None}")

    try:
        # Try to load pre-trained model
        model_path = Path("models/traffic_classifier.joblib")
        logger.debug(f"Model path exists: {model_path.exists()}")

        if model_path.exists():
            classifier = MultiUETrafficClassifier.load(str(model_path))
            logger.info(f"Loaded model from {model_path}")
        else:
            # Create empty classifier for demo purposes
            classifier = MultiUETrafficClassifier()
            logger.warning("No pre-trained model found. Using empty classifier.")

        # Initialize sliding window features
        sliding_window = SlidingWindowFeatures(window_size=25)
        logger.debug(f"Sliding window initialized: {sliding_window is not None}")

        model_loaded = True
        logger.info("API startup complete!")
        logger.debug(f"Final model_loaded: {model_loaded}, classifier: {classifier is not None}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False
        logger.debug(f"Exception caught. model_loaded: {model_loaded}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        uptime_seconds=uptime,
        version="1.0.0",
        timestamp=time.time()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_traffic(request: PredictionRequest):
    """Predict traffic classes for flows."""
    if not model_loaded or classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Convert flows to DataFrame
        df = pd.DataFrame(request.flows)

        if df.empty:
            raise HTTPException(status_code=400, detail="No flow data provided")

        logger.info(f"Processing {len(df)} flows for UE {request.ue_id}")

        # Make predictions
        predictions = []

        if classifier.is_trained:
            # Use trained model
            try:
                y_pred = classifier.predict(df)

                if request.include_probabilities:
                    y_proba, class_names = classifier.predict_proba(df)
                else:
                    y_proba = None
                    class_names = classifier.classes

                # Process each prediction
                for i, pred_class in enumerate(y_pred):
                    # Get probabilities if available
                    if y_proba is not None and i < len(y_proba):
                        probs = {class_names[j]: float(y_proba[i][j]) for j in range(len(class_names))}
                        confidence = float(max(y_proba[i]))
                    else:
                        probs = None
                        confidence = 0.5  # Default confidence

                    # Get QoS mapping
                    qos_info = QOS_MAPPING.get(pred_class, QOS_MAPPING['browsing'])

                    predictions.append(FlowPrediction(
                        predicted_class=pred_class,
                        confidence=confidence,
                        probabilities=probs,
                        qos_priority=qos_info['priority'],
                        qci=qos_info['qci'],
                        delay_budget_ms=qos_info['delay_budget']
                    ))

            except Exception as e:
                logger.warning(f"Model prediction failed: {e}. Using fallback.")
                # Fallback to heuristic predictions
                predictions = _heuristic_predictions(df, request.include_probabilities)

        else:
            # Use heuristic predictions for demo
            logger.info("Using heuristic predictions (model not trained)")
            predictions = _heuristic_predictions(df, request.include_probabilities)

        processing_time = (time.time() - start_time) * 1000

        response = PredictionResponse(
            ue_id=request.ue_id,
            predictions=predictions,
            processing_time_ms=processing_time,
            timestamp=time.time()
        )

        logger.info(f"Processed {len(predictions)} predictions in {processing_time:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def _heuristic_predictions(df: pd.DataFrame, include_probabilities: bool) -> List[FlowPrediction]:
    """Generate heuristic predictions for demo purposes."""
    predictions = []

    for _, row in df.iterrows():
        # Simple heuristic based on available features
        total_packets = row.get('Total Fwd Packets', 0) + row.get('Total Backward Packets', 0)
        flow_duration = row.get('Flow Duration', 1)

        # Heuristic classification
        if total_packets > 1000 and flow_duration > 60:
            pred_class = 'video_streaming'
            confidence = 0.8
        elif total_packets < 50:
            pred_class = 'texting'
            confidence = 0.7
        elif flow_duration < 5:
            pred_class = 'gaming'
            confidence = 0.6
        else:
            pred_class = 'browsing'
            confidence = 0.5

        # Generate mock probabilities
        if include_probabilities:
            probs = {cls: 0.1 for cls in QOS_MAPPING.keys()}
            probs[pred_class] = confidence
            # Normalize
            total_prob = sum(probs.values())
            probs = {k: v/total_prob for k, v in probs.items()}
        else:
            probs = None

        # Get QoS info
        qos_info = QOS_MAPPING[pred_class]

        predictions.append(FlowPrediction(
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=probs,
            qos_priority=qos_info['priority'],
            qci=qos_info['qci'],
            delay_budget_ms=qos_info['delay_budget']
        ))

    return predictions

@app.get("/classes")
async def get_classes():
    """Get available application classes and QoS mapping."""
    return {
        "application_classes": list(QOS_MAPPING.keys()),
        "qos_mapping": QOS_MAPPING,
        "total_classes": len(QOS_MAPPING)
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    if not model_loaded or classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = {
        "model_type": "XGBoost Classifier",
        "is_trained": classifier.is_trained,
        "classes": classifier.classes,
        "feature_count": len(classifier.feature_names) if classifier.feature_names else 0,
        "version": "1.0.0"
    }

    if classifier.is_trained:
        info["feature_names"] = classifier.feature_names[:10]  # First 10 features

    return info

@app.get("/metrics")
async def get_metrics():
    """Get API metrics (for monitoring)."""
    uptime = time.time() - startup_time

    return {
        "uptime_seconds": uptime,
        "model_loaded": model_loaded,
        "active_sessions": sliding_window.get_session_count() if sliding_window else 0,
        "api_version": "1.0.0",
        "status": "healthy" if model_loaded else "degraded"
    }

# Background task to cleanup old sessions
@app.post("/admin/cleanup")
async def cleanup_sessions(background_tasks: BackgroundTasks):
    """Cleanup old sessions (admin endpoint)."""
    if sliding_window:
        background_tasks.add_task(sliding_window.cleanup_old_sessions, 3600)
        return {"message": "Session cleanup scheduled"}
    else:
        raise HTTPException(status_code=503, detail="Sliding window not initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)