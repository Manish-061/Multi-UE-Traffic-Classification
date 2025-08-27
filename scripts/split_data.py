#!/usr/bin/env python3
# Data Splitting Script
# Multi-UE Traffic Classification Project

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Add src to path
sys.path.append('src')

from traffic_classifier.data.splitter import UEBasedDataSplitter
from traffic_classifier.utils.logging import get_logger

logger = get_logger("split_data")

def split_data(input_file, output_dir, test_size=0.15, val_size=0.15):
    """Split synthetic data into train/val/test sets."""
    
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info(f"Loaded {len(df)} flows with {df.shape[1]} columns")
    
    # Check for required columns
    if 'Application_Class' not in df.columns:
        raise ValueError("Dataset must have 'Application_Class' column")
    
    # Create UE-based splits
    splitter = UEBasedDataSplitter(random_state=42)
    
    # Add UE IDs if not present
    if 'UE_ID' not in df.columns:
        logger.info("Creating synthetic UE IDs...")
        df['UE_ID'] = 'UE_' + (df.index // 100).astype(str).str.zfill(3)
    
    # Split data
    logger.info("Creating UE-based splits...")
    splits = splitter.ue_based_split(df, test_size=test_size, val_size=val_size)
    
    # Save splits
    train_file = output_path / "train_data.csv"
    val_file = output_path / "val_data.csv"
    test_file = output_path / "test_data.csv"
    
    splits['train'].to_csv(train_file, index=False)
    splits['val'].to_csv(val_file, index=False)
    splits['test'].to_csv(test_file, index=False)
    
    # Save UE assignments
    ue_assignments = {
        'train_ues': splits['train_ues'],
        'val_ues': splits['val_ues'],
        'test_ues': splits['test_ues']
    }
    
    ue_file = output_path / "ue_assignments.json"
    import json
    with open(ue_file, 'w') as f:
        json.dump(ue_assignments, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SPLITTING RESULTS")
    print("="*60)
    print(f"Train set: {len(splits['train'])} flows ({len(splits['train_ues'])} UEs)")
    print(f"Validation set: {len(splits['val'])} flows ({len(splits['val_ues'])} UEs)")
    print(f"Test set: {len(splits['test'])} flows ({len(splits['test_ues'])} UEs)")
    
    # Class distribution
    print(f"\nClass distribution:")
    for split_name, split_df in [('Train', splits['train']), ('Val', splits['val']), ('Test', splits['test'])]:
        class_counts = split_df['Application_Class'].value_counts()
        print(f"\n{split_name}:")
        for class_name, count in class_counts.items():
            percentage = (count / len(split_df)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\nFiles saved to: {output_path}")
    print(f"UE assignments: {ue_file}")
    
    return splits

def main():
    parser = argparse.ArgumentParser(description="Split synthetic data into train/val/test sets")
    parser.add_argument("--input", "-i", default="data/synthetic/all_synthetic_flows.csv",
                       help="Input CSV file with synthetic data")
    parser.add_argument("--output", "-o", default="data/splits",
                       help="Output directory for splits")
    parser.add_argument("--test-size", type=float, default=0.15,
                       help="Fraction for test set")
    parser.add_argument("--val-size", type=float, default=0.15,
                       help="Fraction for validation set")
    
    args = parser.parse_args()
    
    try:
        splits = split_data(args.input, args.output, args.test_size, args.val_size)
        print("\n✅ Data splitting complete!")
        
    except Exception as e:
        logger.error(f"Data splitting failed: {e}")
        print(f"❌ Data splitting failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
