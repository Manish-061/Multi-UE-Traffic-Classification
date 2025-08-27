# Feature Extraction Script
# Multi-UE Traffic Classification Project

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Add src to path
sys.path.append('src')

from traffic_classifier.features.flow_features import FlowFeatureExtractor
from traffic_classifier.utils.logging import get_logger

logger = get_logger("extract_features")

def extract_features(input_dir, output_dir):
    """Extract features from raw flow data."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {input_path}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Initialize feature extractor
    feature_extractor = FlowFeatureExtractor()
    
    all_features = []
    
    for csv_file in csv_files:
        logger.info(f"Processing {csv_file.name}")
        
        try:
            # Load data
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} flows from {csv_file.name}")
            
            # Extract features
            features = feature_extractor.extract_all_features(df)
            
            # Add filename for tracking
            features['source_file'] = csv_file.name
            
            all_features.append(features)
            
        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
            continue
    
    if all_features:
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Save features
        output_file = output_path / "extracted_features.csv"
        combined_features.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(combined_features)} flows with features to {output_file}")
        logger.info(f"Feature columns: {list(combined_features.columns)}")
        
        return output_file
    else:
        logger.error("No features extracted from any files")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract features from flow data")
    parser.add_argument("--input", "-i", required=True, help="Input directory with CSV files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for features")
    
    args = parser.parse_args()
    
    try:
        output_file = extract_features(args.input, args.output)
        if output_file:
            print(f"✅ Feature extraction complete! Output: {output_file}")
        else:
            print("❌ Feature extraction failed")
            return 1
            
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        print(f"❌ Feature extraction failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
