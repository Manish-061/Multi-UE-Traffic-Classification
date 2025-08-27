"""Traffic classification labeling with QoS mapping."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class TrafficLabeler:
    """Label traffic flows with application classes and QoS priorities."""

    def __init__(self):
        self.app_classes = [
            'video_streaming', 'audio_calls', 'video_calls', 
            'gaming', 'video_uploads', 'browsing', 'texting'
        ]

        self.qos_mapping = {
            'gaming': {'priority': 1, 'qci': 3, 'delay_budget': 50},
            'audio_calls': {'priority': 2, 'qci': 1, 'delay_budget': 100},
            'video_calls': {'priority': 3, 'qci': 2, 'delay_budget': 150},
            'video_streaming': {'priority': 4, 'qci': 6, 'delay_budget': 300},
            'browsing': {'priority': 5, 'qci': 8, 'delay_budget': 300},
            'video_uploads': {'priority': 6, 'qci': 9, 'delay_budget': 300},
            'texting': {'priority': 7, 'qci': 9, 'delay_budget': 300}
        }

    def label_synthetic_data(self, file_path: Path, confidence: float = 0.95) -> Dict[str, Any]:
        """Label synthetic data based on filename.

        Args:
            file_path: Path to synthetic data file
            confidence: Labeling confidence score

        Returns:
            Label information dictionary
        """
        file_path = Path(file_path)

        # Extract application type from filename
        for app_class in self.app_classes:
            if app_class in file_path.stem.lower():
                return {
                    'application_class': app_class,
                    'confidence': confidence,
                    'labeling_method': 'filename_synthetic',
                    'qos_priority': self.qos_mapping[app_class]['priority'],
                    'qci': self.qos_mapping[app_class]['qci'],
                    'delay_budget': self.qos_mapping[app_class]['delay_budget']
                }

        return {
            'application_class': 'unknown',
            'confidence': 0.0,
            'labeling_method': 'filename_unknown',
            'qos_priority': 9,
            'qci': 9,
            'delay_budget': 300
        }

    def label_from_port_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Heuristic labeling based on port patterns.

        Args:
            df: DataFrame with flow data

        Returns:
            List of label dictionaries
        """
        labels = []

        for _, row in df.iterrows():
            src_port = row.get('Source Port', 0)
            dst_port = row.get('Destination Port', 0)
            protocol = row.get('Protocol', 0)
            total_packets = row.get('Total Fwd Packets', 0) + row.get('Total Backward Packets', 0)

            # Gaming ports (common ranges)
            if any(port in range(27000, 28000) for port in [src_port, dst_port]):
                app_class = 'gaming'
                confidence = 0.7
            # Video streaming (HTTP/HTTPS with high packet count)
            elif any(port in [80, 443, 8080] for port in [src_port, dst_port]) and total_packets > 100:
                app_class = 'video_streaming'
                confidence = 0.6
            # VoIP/calls (RTP/SIP ports or UDP with consistent small packets)
            elif (any(port in range(5060, 5070) for port in [src_port, dst_port]) or 
                  (protocol == 17 and total_packets > 50 and total_packets < 500)):
                app_class = 'audio_calls'
                confidence = 0.7
            # Video calls (WebRTC, Zoom ports)
            elif any(port in [3478, 5349, 8801, 8802] for port in [src_port, dst_port]):
                app_class = 'video_calls' 
                confidence = 0.8
            # Web browsing (HTTP/HTTPS with moderate activity)
            elif any(port in [80, 443] for port in [src_port, dst_port]):
                app_class = 'browsing'
                confidence = 0.5
            # FTP/Upload patterns
            elif any(port in [20, 21] for port in [src_port, dst_port]):
                app_class = 'video_uploads'
                confidence = 0.6
            # Default fallback
            else:
                app_class = 'browsing'
                confidence = 0.3

            labels.append({
                'application_class': app_class,
                'confidence': confidence,
                'labeling_method': 'port_heuristic',
                'qos_priority': self.qos_mapping[app_class]['priority'],
                'qci': self.qos_mapping[app_class]['qci'],
                'delay_budget': self.qos_mapping[app_class]['delay_budget']
            })

        return labels

    def create_metadata_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive metadata for labeled dataset.

        Args:
            df: Labeled DataFrame

        Returns:
            Metadata dictionary
        """
        metadata = {
            'dataset_info': {
                'total_flows': len(df),
                'unique_ues': df['UE_ID'].nunique() if 'UE_ID' in df.columns else 'unknown',
                'labeling_timestamp': datetime.now().isoformat(),
                'classes': list(df['Application_Class'].value_counts().to_dict().keys()) if 'Application_Class' in df.columns else []
            },
            'application_classes': self.app_classes,
            'qos_mapping': self.qos_mapping,
            'labeling_confidence': {
                'high': '≥ 0.9 (synthetic, controlled)',
                'medium': '0.6-0.9 (heuristic, port-based)',
                'low': '< 0.6 (uncertain, requires review)'
            },
            'validation_rules': {
                'required_columns': ['UE_ID', 'Application_Class', 'Confidence', 'Labeling_Method'],
                'class_validation': f'Must be one of: {self.app_classes}',
                'confidence_range': '[0.0, 1.0]'
            }
        }

        return metadata

    def generate_labeling_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate quality report for labeled dataset.

        Args:
            df: Labeled DataFrame

        Returns:
            Quality report dictionary
        """
        if 'Application_Class' not in df.columns:
            return {'error': 'No Application_Class column found'}

        class_distribution = df['Application_Class'].value_counts()
        confidence_stats = df['Confidence'].describe() if 'Confidence' in df.columns else {}

        report = {
            'summary': {
                'total_flows': len(df),
                'unique_classes': len(class_distribution),
                'most_common_class': class_distribution.index[0] if len(class_distribution) > 0 else None,
                'least_common_class': class_distribution.index[-1] if len(class_distribution) > 0 else None
            },
            'class_distribution': class_distribution.to_dict(),
            'class_percentages': (class_distribution / len(df) * 100).round(2).to_dict() if len(df) > 0 else {},
            'confidence_statistics': confidence_stats,
            'quality_indicators': {
                'class_balance_score': self._calculate_balance_score(class_distribution),
                'labeling_coverage': (df['Application_Class'] != 'unknown').mean() * 100 if len(df) > 0 else 0,
                'high_confidence_percentage': (df['Confidence'] >= 0.9).mean() * 100 if 'Confidence' in df.columns and len(df) > 0 else 0
            }
        }

        return report

    def _calculate_balance_score(self, class_distribution: pd.Series) -> float:
        """Calculate class balance score (0=perfectly balanced, 1=completely imbalanced).

        Args:
            class_distribution: Value counts of classes

        Returns:
            Balance score between 0 and 1
        """
        if len(class_distribution) <= 1:
            return 1.0

        proportions = class_distribution / class_distribution.sum()
        ideal_proportion = 1.0 / len(proportions)
        balance_score = np.sum(np.abs(proportions - ideal_proportion)) / 2
        return round(balance_score, 3)
