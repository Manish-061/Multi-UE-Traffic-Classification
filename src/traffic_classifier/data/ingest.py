"""PCAP and flow data ingestion utilities."""

import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import os

class PCAPProcessor:
    """Process PCAP files and extract flow features using CICFlowMeter."""

    def __init__(self, cicflowmeter_path: Optional[str] = None):
        self.cicflowmeter_path = cicflowmeter_path or "cicflowmeter"

    def pcap_to_flows(self, pcap_path: str, output_path: str) -> pd.DataFrame:
        """Convert PCAP file to flow CSV using CICFlowMeter.

        Args:
            pcap_path: Path to PCAP file
            output_path: Output CSV path

        Returns:
            DataFrame with extracted flows
        """
        pcap_path = Path(pcap_path)
        output_path = Path(output_path)

        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run CICFlowMeter
        cmd = [self.cicflowmeter_path, "-f", str(pcap_path), "-c", str(output_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"CICFlowMeter output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"CICFlowMeter error: {e.stderr}")
            raise RuntimeError(f"Failed to process PCAP: {e}")
        except FileNotFoundError:
            raise RuntimeError(f"CICFlowMeter not found. Install with: pip install cicflowmeter")

        # Load and return the generated CSV
        if output_path.exists():
            df = pd.read_csv(output_path)
            print(f"Extracted {len(df)} flows from {pcap_path.name}")
            return df
        else:
            raise RuntimeError(f"No flow file generated at {output_path}")

    def batch_process_pcaps(self, pcap_dir: str, output_dir: str) -> Dict[str, pd.DataFrame]:
        """Process multiple PCAP files in a directory.

        Args:
            pcap_dir: Directory containing PCAP files
            output_dir: Output directory for CSV files

        Returns:
            Dictionary mapping filenames to DataFrames
        """
        pcap_dir = Path(pcap_dir)
        output_dir = Path(output_dir)

        if not pcap_dir.exists():
            raise FileNotFoundError(f"PCAP directory not found: {pcap_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        pcap_files = list(pcap_dir.glob("*.pcap")) + list(pcap_dir.glob("*.pcapng"))

        if not pcap_files:
            print(f"No PCAP files found in {pcap_dir}")
            return {}

        results = {}

        for pcap_file in pcap_files:
            output_file = output_dir / f"{pcap_file.stem}_flows.csv"

            try:
                df = self.pcap_to_flows(str(pcap_file), str(output_file))
                results[pcap_file.name] = df
                print(f"✅ Processed: {pcap_file.name} -> {len(df)} flows")
            except Exception as e:
                print(f"❌ Failed to process {pcap_file.name}: {e}")
                continue

        return results

    def capture_live_traffic(self, interface: str, output_path: str, 
                           duration: int = 60, filter_expr: str = "") -> pd.DataFrame:
        """Capture live traffic and convert to flows.

        Args:
            interface: Network interface to capture from
            output_path: Output CSV path
            duration: Capture duration in seconds
            filter_expr: Optional tcpdump filter expression

        Returns:
            DataFrame with captured flows
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary PCAP file
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp_pcap:
            temp_pcap_path = tmp_pcap.name

        try:
            # Capture with tcpdump
            cmd = ["tcpdump", "-i", interface, "-w", temp_pcap_path, "-s", "0"]
            if filter_expr:
                cmd.append(filter_expr)

            print(f"Capturing traffic on {interface} for {duration} seconds...")
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            import time
            time.sleep(duration)
            process.terminate()
            process.wait()

            print("Capture complete. Converting to flows...")

            # Convert to flows
            df = self.pcap_to_flows(temp_pcap_path, str(output_path))
            return df

        finally:
            # Clean up temporary file
            if os.path.exists(temp_pcap_path):
                os.unlink(temp_pcap_path)

    def validate_flow_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate flow data quality and completeness.

        Args:
            df: Flow DataFrame

        Returns:
            Validation report
        """
        required_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        # Basic statistics
        stats = {
            'total_flows': len(df),
            'columns_count': len(df.columns),
            'missing_columns': missing_columns,
            'missing_values_per_column': df.isnull().sum().to_dict(),
            'duplicate_flows': df.duplicated().sum(),
            'zero_duration_flows': (df.get('Flow Duration', pd.Series([1])) == 0).sum(),
            'negative_values': {
                col: (df[col] < 0).sum() for col in df.select_dtypes(include=['number']).columns
            }
        }

        # Data quality score
        quality_issues = (
            len(missing_columns) +
            sum(df.isnull().sum()) +
            stats['duplicate_flows'] +
            stats['zero_duration_flows'] +
            sum(stats['negative_values'].values())
        )

        stats['quality_score'] = max(0, 1 - (quality_issues / len(df)))
        stats['quality_level'] = (
            'High' if stats['quality_score'] > 0.9 else
            'Medium' if stats['quality_score'] > 0.7 else
            'Low'
        )

        return stats
