"""Sliding window feature extraction for real-time classification."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import time

class SlidingWindowFeatures:
    """Extract features over sliding windows for early classification."""

    def __init__(self, window_size: int = 25, feature_columns: Optional[List[str]] = None):
        self.window_size = window_size
        self.feature_columns = feature_columns or [
            'packet_size', 'inter_arrival_time', 'direction', 'protocol'
        ]
        self.windows = {}  # session_id -> window data

    def update_window(self, session_id: str, packet_data: Dict[str, Any]) -> Dict[str, float]:
        """Update sliding window with new packet and compute features.

        Args:
            session_id: Unique session identifier
            packet_data: Dictionary with packet information

        Returns:
            Dictionary of computed window features
        """
        if session_id not in self.windows:
            self.windows[session_id] = {
                'packets': deque(maxlen=self.window_size),
                'timestamps': deque(maxlen=self.window_size),
                'last_update': time.time()
            }

        window = self.windows[session_id]

        # Add new packet
        window['packets'].append(packet_data)
        window['timestamps'].append(time.time())
        window['last_update'] = time.time()

        # Compute features
        features = self._compute_window_features(window['packets'])
        return features

    def _compute_window_features(self, packets: deque) -> Dict[str, float]:
        """Compute features from current window of packets.

        Args:
            packets: Deque of packet data

        Returns:
            Dictionary of computed features
        """
        if not packets:
            return self._get_default_features()

        packet_list = list(packets)
        features = {}

        # Packet size statistics
        sizes = [p.get('packet_size', 0) for p in packet_list]
        features.update(self._compute_stats('packet_size', sizes))

        # Inter-arrival time statistics
        if len(packet_list) > 1:
            iats = []
            for i in range(1, len(packet_list)):
                if 'timestamp' in packet_list[i] and 'timestamp' in packet_list[i-1]:
                    iat = packet_list[i]['timestamp'] - packet_list[i-1]['timestamp']
                    iats.append(iat)

            if iats:
                features.update(self._compute_stats('iat', iats))
            else:
                features.update(self._compute_stats('iat', [0]))
        else:
            features.update(self._compute_stats('iat', [0]))

        # Direction-based features
        directions = [p.get('direction', 0) for p in packet_list]  # 0=forward, 1=backward
        features['fwd_packets_ratio'] = sum(1 for d in directions if d == 0) / len(directions)
        features['bwd_packets_ratio'] = sum(1 for d in directions if d == 1) / len(directions)

        # Protocol features
        protocols = [p.get('protocol', 6) for p in packet_list]  # Default TCP
        features['tcp_packets_ratio'] = sum(1 for p in protocols if p == 6) / len(protocols)
        features['udp_packets_ratio'] = sum(1 for p in protocols if p == 17) / len(protocols)

        # Throughput features
        total_bytes = sum(sizes)
        if len(packet_list) > 1 and 'timestamp' in packet_list[0] and 'timestamp' in packet_list[-1]:
            duration = packet_list[-1]['timestamp'] - packet_list[0]['timestamp']
            features['throughput_bps'] = total_bytes / (duration + 1e-10)
            features['packet_rate_pps'] = len(packet_list) / (duration + 1e-10)
        else:
            features['throughput_bps'] = 0.0
            features['packet_rate_pps'] = 0.0

        # Burstiness features
        if len(sizes) > 1:
            size_variance = np.var(sizes)
            size_mean = np.mean(sizes)
            features['burstiness_index'] = size_variance / (size_mean + 1e-10)
        else:
            features['burstiness_index'] = 0.0

        return features

    def _compute_stats(self, prefix: str, values: List[float]) -> Dict[str, float]:
        """Compute statistical features for a list of values.

        Args:
            prefix: Feature name prefix
            values: List of values

        Returns:
            Dictionary of statistical features
        """
        if not values:
            return {
                f'{prefix}_mean': 0.0,
                f'{prefix}_std': 0.0,
                f'{prefix}_min': 0.0,
                f'{prefix}_max': 0.0,
                f'{prefix}_median': 0.0
            }

        values_array = np.array(values)
        return {
            f'{prefix}_mean': float(np.mean(values_array)),
            f'{prefix}_std': float(np.std(values_array)),
            f'{prefix}_min': float(np.min(values_array)),
            f'{prefix}_max': float(np.max(values_array)),
            f'{prefix}_median': float(np.median(values_array))
        }

    def _get_default_features(self) -> Dict[str, float]:
        """Get default features for empty windows.

        Returns:
            Dictionary with default feature values
        """
        features = {}

        # Default packet size stats
        features.update(self._compute_stats('packet_size', []))

        # Default IAT stats
        features.update(self._compute_stats('iat', []))

        # Default ratios
        features.update({
            'fwd_packets_ratio': 0.0,
            'bwd_packets_ratio': 0.0,
            'tcp_packets_ratio': 1.0,  # Assume TCP by default
            'udp_packets_ratio': 0.0,
            'throughput_bps': 0.0,
            'packet_rate_pps': 0.0,
            'burstiness_index': 0.0
        })

        return features

    def get_feature_vector(self, session_id: str) -> np.ndarray:
        """Get feature vector for a session.

        Args:
            session_id: Session identifier

        Returns:
            Numpy array of features
        """
        if session_id not in self.windows:
            features = self._get_default_features()
        else:
            window = self.windows[session_id]
            features = self._compute_window_features(window['packets'])

        # Convert to ordered array
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])

        return feature_vector

    def cleanup_old_sessions(self, max_age_seconds: int = 3600):
        """Remove old sessions to prevent memory leaks.

        Args:
            max_age_seconds: Maximum age for sessions in seconds
        """
        current_time = time.time()
        sessions_to_remove = []

        for session_id, window in self.windows.items():
            if current_time - window['last_update'] > max_age_seconds:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.windows[session_id]

        if sessions_to_remove:
            print(f"Cleaned up {len(sessions_to_remove)} old sessions")

    def get_session_count(self) -> int:
        """Get number of active sessions.

        Returns:
            Number of active sessions
        """
        return len(self.windows)
