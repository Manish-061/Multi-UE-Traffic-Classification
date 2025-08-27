#!/usr/bin/env python3
# Model Training Script
# Multi-UE Traffic Classification Project

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import argparse
import time
from sklearn.model_selection import train_test_split

# Import our modules
import sys
sys.path.append('src')

from traffic_classifier.models.xgb_classifier import MultiUETrafficClassifier
from traffic_classifier.data.splitter import UEBasedDataSplitter
from traffic_classifier.features.flow_features import FlowFeatureExtractor
from traffic_classifier.utils.logging import get_logger
from traffic_classifier.utils.io import load_config, save_results

logger = get_logger("train_model")

def load_training_data(data_path):
    """Load and prepare training data."""
    logger.info(f"Loading training data from {data_path}")

    data_path = Path(data_path)

    if data_path.is_file():
        # Single file
        df = pd.read_csv(data_path)
    elif data_path.is_dir():
        # Directory with multiple files
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        dfs = []
        for csv_file in csv_files:
            df_temp = pd.read_csv(csv_file)
            dfs.append(df_temp)

        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(csv_files)} files")
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    logger.info(f"Loaded {len(df)} flows with {df.shape[1]} columns")

    # Check for required columns
    if 'Application_Class' not in df.columns:
        raise ValueError("Dataset must have 'Application_Class' column")

    return df

def train_model(config_path="configs/model_xgb.yaml", data_path="data/synthetic/all_synthetic_flows.csv"):
    """Train the traffic classification model."""

    # Load configuration
    config = load_config(config_path)
    model_params = config.get('model_params', {})
    training_config = config.get('training', {})

    logger.info(f"Training configuration loaded from {config_path}")

    # Load data
    df = load_training_data(data_path)

    # Create UE-based splits if UE_ID doesn't exist
    if 'UE_ID' not in df.columns:
        # Create simple UE IDs based on index
        df['UE_ID'] = 'UE_' + (df.index // 100).astype(str).str.zfill(3)
        logger.info("Created synthetic UE IDs")

    # Split data using UE-based approach
    splitter = UEBasedDataSplitter(random_state=42)
    splits = splitter.ue_based_split(df, test_size=0.15, val_size=0.15)

    train_df = splits['train']
    val_df = splits['val']
    test_df = splits['test']

    logger.info(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Save splits to disk
    splitter.save_splits(splits)

    # Feature extraction
    feature_extractor = FlowFeatureExtractor()

    # Extract features
    logger.info("Extracting features...")
    train_features = feature_extractor.extract_all_features(train_df)
    val_features = feature_extractor.extract_all_features(val_df)
    test_features = feature_extractor.extract_all_features(test_df)

    # Select features
    X_train = feature_extractor.select_features(train_features)
    feature_extractor.fit_scaler(X_train)

    # Apply same feature selection to val/test
    X_val = feature_extractor.transform_features(val_features[feature_extractor.feature_columns])
    X_test = feature_extractor.transform_features(test_features[feature_extractor.feature_columns])

    # Target variables
    y_train = train_df['Application_Class']
    y_val = val_df['Application_Class']
    y_test = test_df['Application_Class']

    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Feature columns: {len(feature_extractor.feature_columns)}")

    # Initialize classifier
    classifier = MultiUETrafficClassifier(**model_params)

    # Get class weights if specified
    class_weights = training_config.get('class_weights')

    # Train model
    logger.info("Starting model training...")
    start_time = time.time()

    training_results = classifier.train(
        X_train, y_train,
        X_val, y_val,
        class_weights=class_weights
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Calibrate probabilities if enabled
    calibration_config = config.get('calibration', {})
    if calibration_config.get('enabled', True):
        logger.info("Calibrating probabilities...")
        classifier.calibrate_probabilities(X_val, y_val, method=calibration_config.get('method', 'sigmoid'))

    # Evaluate on test set
    logger.info("Evaluating model...")
    evaluation_results = classifier.evaluate(X_test, y_test)

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "traffic_classifier.joblib"

    classifier.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training results
    results = {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'training_time_seconds': training_time,
        'data_splits': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        },
        'feature_count': len(feature_extractor.feature_columns),
        'model_path': str(model_path)
    }

    results_path = Path("artifacts/training_results.json")
    results_path.parent.mkdir(exist_ok=True)
    save_results(results, results_path)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved: {model_path}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Features used: {len(feature_extractor.feature_columns)}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {evaluation_results['accuracy']:.3f}")
    print(f"  Macro F1: {evaluation_results['macro_f1']:.3f}")
    print(f"  Top-2 Accuracy: {evaluation_results['top2_accuracy']:.3f}")

    print(f"\nPer-class F1 scores:")
    for class_name, metrics in evaluation_results['per_class_metrics'].items():
        print(f"  {class_name:<15}: {metrics['f1_score']:.3f}")

    # Check if targets are met
    targets = config.get('performance_targets', {})
    macro_f1_target = targets.get('macro_f1', 0.80)

    if evaluation_results['macro_f1'] >= macro_f1_target:
        print(f"\nTarget achieved! Macro F1 ({evaluation_results['macro_f1']:.3f}) >= {macro_f1_target}")
    else:
        print(f"\nTarget not met. Macro F1 ({evaluation_results['macro_f1']:.3f}) < {macro_f1_target}")

    return classifier, results

def main():
    parser = argparse.ArgumentParser(description="Train Multi-UE Traffic Classifier")
    parser.add_argument("--config", default="configs/model_xgb.yaml", 
                       help="Model configuration file")
    parser.add_argument("--data", default="data/synthetic/all_synthetic_flows.csv",
                       help="Training data path (file or directory)")
    parser.add_argument("--output-model", default="models/traffic_classifier.joblib",
                       help="Output model path")

    args = parser.parse_args()

    try:
        classifier, results = train_model(args.config, args.data)

        # Save model to specified path if different
        if args.output_model != "models/traffic_classifier.joblib":
            classifier.save(args.output_model)
            print(f"\nModel also saved to: {args.output_model}")

        print("\nReady for inference! Try:")
        print("  python scripts/serve_api.py --port 8000    # Start API server")
        print("  python -c \"from src.traffic_classifier.serving.cli import demo; demo()\"     # Run CLI demo")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    main()