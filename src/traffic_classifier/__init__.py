"""
Multi-UE Traffic Classifier

AI-powered network traffic classification for 5G QoS optimization.
"""

__version__ = "1.0.0"
__author__ = "Multi-UE Traffic Classification Team"
__email__ = "contact@traffic-classifier.com"

from .utils.logging import get_logger
from .data.labeler import TrafficLabeler
from .models.xgb_classifier import MultiUETrafficClassifier

__all__ = [
    "get_logger",
    "TrafficLabeler", 
    "MultiUETrafficClassifier"
]
