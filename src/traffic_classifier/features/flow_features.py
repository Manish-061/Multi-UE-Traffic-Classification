"""Flow-level feature extraction and engineering."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

class FlowFeatureExtractor:
    """Extract and engineer features from network flow data."""

    def __init__(self):
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic flow statistics.

        Args:
            df: Input DataFrame with flow data

        Returns:
            DataFrame with basic features
        """
        features_df = df.copy()

        # Packet count features
        if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
            features_df['Total_Packets'] = features_df['Total Fwd Packets'] + features_df['Total Backward Packets']
            features_df['Fwd_Bwd_Packet_Ratio'] = features_df['Total Fwd Packets'] / (features_df['Total Backward Packets'] + 1e-10)

        # Bytes features
        if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
            features_df['Total_Bytes'] = features_df['Total Length of Fwd Packets'] + features_df['Total Length of Bwd Packets']
            features_df['Fwd_Bwd_Bytes_Ratio'] = features_df['Total Length of Fwd Packets'] / (features_df['Total Length of Bwd Packets'] + 1e-10)

        # Packet size features
        if 'Total_Packets' in features_df.columns and 'Total_Bytes' in features_df.columns:
            features_df['Avg_Packet_Size'] = features_df['Total_Bytes'] / (features_df['Total_Packets'] + 1e-10)

        # Throughput features
        if 'Flow Duration' in df.columns and 'Total_Bytes' in features_df.columns:
            features_df['Bytes_Per_Second'] = features_df['Total_Bytes'] / (features_df['Flow Duration'] + 1e-10)

        if 'Flow Duration' in df.columns and 'Total_Packets' in features_df.columns:
            features_df['Packets_Per_Second'] = features_df['Total_Packets'] / (features_df['Flow Duration'] + 1e-10)

        return features_df

    def extract_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract timing-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with timing features
        """
        features_df = df.copy()

        # Inter-arrival time ratios
        if 'Flow IAT Mean' in df.columns and 'Flow IAT Std' in df.columns:
            features_df['IAT_Coefficient_Variation'] = features_df['Flow IAT Std'] / (features_df['Flow IAT Mean'] + 1e-10)

        # Forward vs Backward timing
        fwd_iat_cols = ['Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min']
        bwd_iat_cols = ['Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min']

        for fwd_col, bwd_col in zip(fwd_iat_cols, bwd_iat_cols):
            if fwd_col in df.columns and bwd_col in df.columns:
                ratio_col = f"{fwd_col.replace(' ', '_')}_to_{bwd_col.replace(' ', '_')}_Ratio"
                features_df[ratio_col] = features_df[fwd_col] / (features_df[bwd_col] + 1e-10)

        return features_df

    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral pattern features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with behavioral features
        """
        features_df = df.copy()

        # Flag-based features
        flag_columns = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
                       'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count']

        available_flags = [col for col in flag_columns if col in df.columns]

        if available_flags:
            features_df['Total_Flags'] = features_df[available_flags].sum(axis=1)

            for flag_col in available_flags:
                flag_ratio_col = f"{flag_col.replace(' ', '_')}_Ratio"
                features_df[flag_ratio_col] = features_df[flag_col] / (features_df['Total_Flags'] + 1e-10)

        # Burstiness indicators
        if 'Flow IAT Std' in df.columns and 'Flow IAT Mean' in df.columns:
            features_df['Burstiness_Index'] = features_df['Flow IAT Std'] / (features_df['Flow IAT Mean'] + 1e-10)

        # Protocol patterns
        if 'Protocol' in df.columns:
            features_df['Is_TCP'] = (features_df['Protocol'] == 6).astype(int)
            features_df['Is_UDP'] = (features_df['Protocol'] == 17).astype(int)
            features_df['Is_ICMP'] = (features_df['Protocol'] == 1).astype(int)

        return features_df

    def create_sliding_window_features(self, df: pd.DataFrame, window_sizes: List[int] = [10, 25, 50]) -> pd.DataFrame:
        """Create sliding window features for early classification.

        Args:
            df: Input DataFrame
            window_sizes: List of window sizes in packets

        Returns:
            DataFrame with sliding window features
        """
        features_df = df.copy()

        # This is a simplified version - in practice, you'd need packet-level data
        # For now, we'll create rolling statistics on flow-level data

        if 'Total Fwd Packets' in df.columns:
            for window_size in window_sizes:
                col_name = f'Fwd_Packets_Rolling_Mean_{window_size}'
                features_df[col_name] = features_df['Total Fwd Packets'].rolling(window=window_size, min_periods=1).mean()

                col_name = f'Fwd_Packets_Rolling_Std_{window_size}'
                features_df[col_name] = features_df['Total Fwd Packets'].rolling(window=window_size, min_periods=1).std().fillna(0)

        if 'Flow Duration' in df.columns:
            for window_size in window_sizes:
                col_name = f'Duration_Rolling_Mean_{window_size}'
                features_df[col_name] = features_df['Flow Duration'].rolling(window=window_size, min_periods=1).mean()

        return features_df

    def select_features(self, df: pd.DataFrame, feature_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Select and filter features based on configuration.

        Args:
            df: Input DataFrame
            feature_config: Feature selection configuration

        Returns:
            DataFrame with selected features
        """
        if feature_config is None:
            feature_config = {'max_features': 50}

        # Get numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target column if present
        if 'Application_Class' in numeric_columns:
            numeric_columns.remove('Application_Class')

        # Remove identifier columns
        id_columns = ['UE_ID', 'Session_ID', 'Source IP', 'Destination IP']
        numeric_columns = [col for col in numeric_columns if col not in id_columns]

        # Limit number of features
        max_features = feature_config.get('max_features', 50)
        if len(numeric_columns) > max_features:
            # Simple feature selection based on variance
            feature_variance = df[numeric_columns].var().sort_values(ascending=False)
            selected_features = feature_variance.head(max_features).index.tolist()
        else:
            selected_features = numeric_columns

        self.feature_columns = selected_features
        return df[selected_features]

    def fit_scaler(self, df: pd.DataFrame) -> 'FlowFeatureExtractor':
        """Fit the feature scaler on training data.

        Args:
            df: Training DataFrame

        Returns:
            Self for method chaining
        """
        if self.feature_columns:
            feature_data = df[self.feature_columns]
        else:
            feature_data = df.select_dtypes(include=[np.number])

        self.scaler.fit(feature_data)
        self.is_fitted = True
        return self

    def transform_features(self, df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        """Transform features with scaling.

        Args:
            df: Input DataFrame
            scale: Whether to apply scaling

        Returns:
            Transformed DataFrame
        """
        if self.feature_columns:
            feature_data = df[self.feature_columns]
        else:
            feature_data = df.select_dtypes(include=[np.number])

        if scale and self.is_fitted:
            scaled_data = self.scaler.transform(feature_data)
            scaled_df = pd.DataFrame(scaled_data, columns=feature_data.columns, index=feature_data.index)
            return scaled_df
        else:
            return feature_data

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all available features from flow data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all extracted features
        """
        # Start with basic features
        features_df = self.extract_basic_features(df)

        # Add timing features
        features_df = self.extract_timing_features(features_df)

        # Add behavioral features
        features_df = self.extract_behavioral_features(features_df)

        # Add sliding window features
        features_df = self.create_sliding_window_features(features_df)

        return features_df
