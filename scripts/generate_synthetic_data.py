# Synthetic Traffic Generation Script
# Multi-UE Traffic Classification Project

import subprocess
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

class SyntheticTrafficGenerator:
    def __init__(self, output_dir="data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.app_classes = {
            "video_streaming": {
                "duration": 1800,  # 30 minutes
                "description": "Netflix, YouTube streaming",
                "instructions": "Open YouTube/Netflix and start streaming video"
            },
            "gaming": {
                "duration": 1200,  # 20 minutes
                "description": "Online gaming sessions",
                "instructions": "Launch online game and play actively"
            },
            "video_calls": {
                "duration": 1200,  # 20 minutes
                "description": "Zoom, Teams video calls",
                "instructions": "Start video call on Zoom/Teams/Skype"
            },
            "audio_calls": {
                "duration": 900,   # 15 minutes
                "description": "Voice calls, VoIP",
                "instructions": "Make voice call via WhatsApp/Phone"
            },
            "browsing": {
                "duration": 1800,  # 30 minutes
                "description": "Web browsing, social media",
                "instructions": "Browse websites and social media actively"
            },
            "video_uploads": {
                "duration": 900,   # 15 minutes
                "description": "Upload videos to social platforms",
                "instructions": "Upload videos to Instagram/TikTok/YouTube"
            },
            "texting": {
                "duration": 600,   # 10 minutes
                "description": "Text messaging, chat apps",
                "instructions": "Send messages via WhatsApp/SMS/Telegram"
            }
        }

    def generate_synthetic_csv(self, app_class, num_flows=500):
        """Generate synthetic flow data without network capture."""
        print(f"Generating synthetic {app_class} data ({num_flows} flows)...")

        np.random.seed(42)
        flows = []

        for i in range(num_flows):
            if app_class == "gaming":
                flow_duration = max(0.1, np.random.exponential(2))
                total_fwd_packets = max(1, int(np.random.poisson(50)))
                total_bwd_packets = max(1, int(np.random.poisson(30)))
                avg_packet_size = 80 + np.random.normal(0, 20)
            elif app_class == "video_streaming":
                flow_duration = max(1, np.random.exponential(180))
                total_fwd_packets = max(1, int(np.random.poisson(1000)))
                total_bwd_packets = max(1, int(np.random.poisson(200)))
                avg_packet_size = 1200 + np.random.normal(0, 200)
            elif app_class == "audio_calls":
                flow_duration = max(1, np.random.exponential(300))
                total_fwd_packets = max(1, int(np.random.poisson(800)))
                total_bwd_packets = max(1, int(np.random.poisson(800)))
                avg_packet_size = 160 + np.random.normal(0, 30)
            elif app_class == "video_calls":
                flow_duration = max(1, np.random.exponential(600))
                total_fwd_packets = max(1, int(np.random.poisson(1500)))
                total_bwd_packets = max(1, int(np.random.poisson(1200)))
                avg_packet_size = 1000 + np.random.normal(0, 150)
            elif app_class == "browsing":
                flow_duration = max(0.5, np.random.exponential(30))
                total_fwd_packets = max(1, int(np.random.poisson(100)))
                total_bwd_packets = max(1, int(np.random.poisson(300)))
                avg_packet_size = 800 + np.random.normal(0, 200)
            elif app_class == "video_uploads":
                flow_duration = max(5, np.random.exponential(120))
                total_fwd_packets = max(1, int(np.random.poisson(2000)))
                total_bwd_packets = max(1, int(np.random.poisson(100)))
                avg_packet_size = 1400 + np.random.normal(0, 300)
            else:  # texting
                flow_duration = max(0.1, np.random.exponential(1))
                total_fwd_packets = max(1, int(np.random.poisson(5)))
                total_bwd_packets = max(1, int(np.random.poisson(3)))
                avg_packet_size = 100 + np.random.normal(0, 30)

            # Calculate derived features
            total_fwd_bytes = int(total_fwd_packets * max(50, avg_packet_size))
            total_bwd_bytes = int(total_bwd_packets * max(50, avg_packet_size * 0.8))

            flow_bytes_s = (total_fwd_bytes + total_bwd_bytes) / flow_duration
            flow_packets_s = (total_fwd_packets + total_bwd_packets) / flow_duration

            # Inter-arrival time features
            iat_mean = flow_duration / max(1, total_fwd_packets + total_bwd_packets) * 1000000  # microseconds
            iat_std = iat_mean * (0.1 + np.random.exponential(0.3))

            flow = {
                'Flow Duration': flow_duration,
                'Total Fwd Packets': total_fwd_packets,
                'Total Backward Packets': total_bwd_packets,
                'Total Length of Fwd Packets': total_fwd_bytes,
                'Total Length of Bwd Packets': total_bwd_bytes,
                'Flow Bytes/s': flow_bytes_s,
                'Flow Packets/s': flow_packets_s,
                'Flow IAT Mean': iat_mean,
                'Flow IAT Std': iat_std,
                'Flow IAT Max': iat_mean + 3 * iat_std,
                'Flow IAT Min': max(0, iat_mean - iat_std),
                'Fwd Packet Length Mean': avg_packet_size,
                'Bwd Packet Length Mean': avg_packet_size * 0.8,
                'Packet Length Mean': (avg_packet_size + avg_packet_size * 0.8) / 2,
                'Application_Class': app_class,
                'UE_ID': f'synthetic_UE_{(i // 50):03d}',  # 50 flows per UE
                'Session_ID': f'synthetic_{app_class}_{i:04d}'
            }

            flows.append(flow)

        # Create DataFrame and save
        df = pd.DataFrame(flows)
        output_file = self.output_dir / f"{app_class}_flows.csv"
        df.to_csv(output_file, index=False)

        print(f"Generated {len(df)} {app_class} flows -> {output_file}")
        return df

    def generate_all_synthetic(self, flows_per_class=500):
        """Generate synthetic data for all application classes."""
        print("Generating synthetic traffic data for all classes...")
        print("This creates labeled flow data without requiring network capture.\n")

        all_flows = []

        for app_class in self.app_classes.keys():
            df = self.generate_synthetic_csv(app_class, flows_per_class)
            all_flows.append(df)

        # Combine all flows
        combined_df = pd.concat(all_flows, ignore_index=True)
        combined_file = self.output_dir / "all_synthetic_flows.csv"
        combined_df.to_csv(combined_file, index=False)

        print(f"\nCombined dataset: {len(combined_df)} total flows -> {combined_file}")

        # Print statistics
        print("\nDataset Statistics:")
        print("-" * 40)
        class_counts = combined_df['Application_Class'].value_counts()
        for app_class, count in class_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"{app_class:<15}: {count:>4} flows ({percentage:>5.1f}%)")

        ue_count = combined_df['UE_ID'].nunique()
        print(f"\nUnique UEs: {ue_count}")
        print(f"Avg flows per UE: {len(combined_df) / ue_count:.1f}")

        return combined_df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traffic data")
    parser.add_argument("--flows-per-class", type=int, default=500, 
                       help="Number of flows per application class")
    parser.add_argument("--output-dir", default="data/synthetic",
                       help="Output directory for synthetic data")
    parser.add_argument("--classes", nargs="+", 
                       help="Specific classes to generate (default: all)")

    args = parser.parse_args()

    generator = SyntheticTrafficGenerator(args.output_dir)

    if args.classes:
        # Generate specific classes
        all_flows = []
        for app_class in args.classes:
            if app_class in generator.app_classes:
                df = generator.generate_synthetic_csv(app_class, args.flows_per_class)
                all_flows.append(df)
            else:
                print(f"Unknown class: {app_class}")

        if all_flows:
            combined_df = pd.concat(all_flows, ignore_index=True)
            combined_file = Path(args.output_dir) / "selected_synthetic_flows.csv"
            combined_df.to_csv(combined_file, index=False)
            print(f"Generated {len(combined_df)} flows -> {combined_file}")
    else:
        # Generate all classes
        generator.generate_all_synthetic(args.flows_per_class)

    print("\nSynthetic data generation complete!")
    print("\nNext steps:")
    print("1. Use generated data for training: python scripts/train_model.py --config configs/model_xgb.yaml")
    print("2. Or create UE-based splits: python scripts/split_data.py")

if __name__ == "__main__":
    main()