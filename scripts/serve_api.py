#!/usr/bin/env python3
# API Server Launcher
# Multi-UE Traffic Classification Project

import uvicorn
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from traffic_classifier.serving.api import app
from traffic_classifier.utils.logging import get_logger

logger = get_logger("api_server")

def main():
    parser = argparse.ArgumentParser(description="Start Multi-UE Traffic Classifier API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    print("🚀 Starting Multi-UE Traffic Classifier API...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API documentation: http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to stop the server\n")

    # Check if model exists
    model_path = Path("models/traffic_classifier.joblib")
    if not model_path.exists():
        print("⚠️  Warning: No trained model found at models/traffic_classifier.joblib")
        print("   The API will run in demo mode with heuristic predictions.")

    try:
        uvicorn.run(
            "traffic_classifier.serving.api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level,
            app_dir="src"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"❌ Server failed to start: {e}")

if __name__ == "__main__":
    main()
