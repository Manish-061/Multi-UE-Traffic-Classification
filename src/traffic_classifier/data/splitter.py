"""UE-based data splitting to prevent temporal leakage."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
import json

class UEBasedDataSplitter:
    """Split data by UE to prevent temporal leakage between train/val/test."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def create_ue_sessions(self, df: pd.DataFrame, 
                          src_ip_col: str = 'Source IP', 
                          dst_ip_col: str = 'Destination IP') -> pd.DataFrame:
        """Create UE session identifiers from flow data.

        Args:
            df: Input DataFrame
            src_ip_col: Source IP column name
            dst_ip_col: Destination IP column name

        Returns:
            DataFrame with UE_ID and Session_ID columns
        """
        df = df.copy()

        # Create UE identifiers (in real scenarios, this would be more sophisticated)
        if src_ip_col in df.columns and dst_ip_col in df.columns:
            df['UE_ID'] = df[src_ip_col].astype(str) + '_' + df[dst_ip_col].astype(str)
        else:
            # Fallback: create synthetic UE IDs
            df['UE_ID'] = 'UE_' + (df.index // 100).astype(str).str.zfill(3)

        df['Session_ID'] = df['UE_ID'] + '_' + df.index.astype(str)
        return df

    def ue_based_split(self, df: pd.DataFrame, 
                      test_size: float = 0.15, 
                      val_size: float = 0.15) -> Dict[str, Any]:
        """Split data by UE to prevent temporal leakage.

        Args:
            df: Input DataFrame with UE_ID column
            test_size: Fraction for test set
            val_size: Fraction for validation set

        Returns:
            Dictionary with train/val/test DataFrames and UE assignments
        """
        if 'UE_ID' not in df.columns:
            raise ValueError("DataFrame must have 'UE_ID' column")

        # Get unique UEs and their statistics
        ue_stats = df.groupby('UE_ID').agg({
            'Application_Class': ['count', lambda x: list(x.unique())] if 'Application_Class' in df.columns else 'count'
        }).round(2)

        if 'Application_Class' in df.columns:
            ue_stats.columns = ['flow_count', 'app_classes']
        else:
            ue_stats.columns = ['flow_count']
            ue_stats['app_classes'] = [['unknown']] * len(ue_stats)

        ue_stats = ue_stats.reset_index()

        # Filter UEs with minimum flow requirements
        min_flows_per_ue = 30
        valid_ues = ue_stats[ue_stats['flow_count'] >= min_flows_per_ue]['UE_ID'].tolist()

        if len(valid_ues) < 10:
            print(f"Warning: Only {len(valid_ues)} UEs meet minimum flow requirements")
            if len(valid_ues) < 3:
                # Create synthetic UEs if too few
                print("Creating synthetic UE splits...")
                all_ues = df['UE_ID'].unique().tolist()
                valid_ues = all_ues[:max(10, len(all_ues))]

        # Split UEs (not flows) into train/val/test
        train_ues, temp_ues = train_test_split(
            valid_ues, 
            test_size=(test_size + val_size), 
            random_state=self.random_state, 
            shuffle=True
        )

        if len(temp_ues) >= 2:
            val_ues, test_ues = train_test_split(
                temp_ues, 
                test_size=(test_size/(test_size + val_size)), 
                random_state=self.random_state, 
                shuffle=True
            )
        else:
            # Fallback for small datasets
            val_ues = temp_ues[:len(temp_ues)//2] if len(temp_ues) > 0 else []
            test_ues = temp_ues[len(temp_ues)//2:] if len(temp_ues) > 0 else []

        # Create splits based on UE assignment
        train_df = df[df['UE_ID'].isin(train_ues)].copy()
        val_df = df[df['UE_ID'].isin(val_ues)].copy()
        test_df = df[df['UE_ID'].isin(test_ues)].copy()

        return {
            'train': train_df,
            'val': val_df, 
            'test': test_df,
            'ue_assignment': {
                'train_ues': train_ues,
                'val_ues': val_ues,
                'test_ues': test_ues
            }
        }

    def validate_splits(self, splits: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data splits and generate quality report.

        Args:
            splits: Dictionary with train/val/test splits

        Returns:
            Validation report
        """
        report = {}

        for split_name, df in splits.items():
            if split_name == 'ue_assignment':
                continue

            if 'Application_Class' in df.columns:
                class_dist = df['Application_Class'].value_counts()
            else:
                class_dist = pd.Series({'unknown': len(df)})

            ue_count = df['UE_ID'].nunique()

            report[split_name] = {
                'total_flows': len(df),
                'unique_ues': ue_count,
                'class_distribution': class_dist.to_dict(),
                'flows_per_ue': len(df) / ue_count if ue_count > 0 else 0
            }

        return report

    def save_splits(self, splits: Dict[str, Any], output_dir: str = 'data/splits') -> Dict[str, Any]:
        """Save train/val/test splits to separate files.

        Args:
            splits: Dictionary with splits
            output_dir: Output directory

        Returns:
            Validation report
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, data in splits.items():
            if split_name == 'ue_assignment':
                # Save UE assignments as JSON
                with open(output_path / 'ue_assignments.json', 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                # Save dataframes as CSV
                output_file = output_path / f'{split_name}_data.csv'
                data.to_csv(output_file, index=False)
                print(f"Saved {len(data)} flows to {output_file}")

        # Generate and save validation report
        report = self.validate_splits(splits)
        with open(output_path / 'split_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        return report
