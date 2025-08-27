"""Command-line interface for traffic classification."""

import pandas as pd
import numpy as np
import click
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

from ..models.xgb_classifier import MultiUETrafficClassifier
from ..data.labeler import TrafficLabeler
from ..utils.logging import get_logger
from ..utils.io import load_data, save_results

logger = get_logger("traffic_classifier_cli")

@click.group()
def cli():
    """Multi-UE Traffic Classifier CLI."""
    pass

@cli.command()
@click.option('--input', '-i', required=True, help='Input CSV file with flow data')
@click.option('--model', '-m', default='models/traffic_classifier.joblib', help='Path to trained model')
@click.option('--output', '-o', help='Output file for predictions (optional)')
@click.option('--confidence-threshold', '-c', default=0.7, help='Confidence threshold for predictions')
@click.option('--include-probabilities', is_flag=True, help='Include probability scores in output')
def predict(input: str, model: str, output: Optional[str], 
           confidence_threshold: float, include_probabilities: bool):
    """Predict traffic classes from flow CSV file."""

    logger.info(f"Loading flow data from {input}")

    try:
        # Load data
        df = load_data(input)
        logger.info(f"Loaded {len(df)} flows")

        # Load model
        classifier = MultiUETrafficClassifier.load(model)
        logger.info(f"Loaded model from {model}")

        # Make predictions
        start_time = time.time()

        if classifier.is_trained:
            predictions = classifier.predict(df)

            if include_probabilities:
                probabilities, class_names = classifier.predict_proba(df)
            else:
                probabilities = None
                class_names = classifier.classes
        else:
            logger.warning("Model is not trained. Using heuristic predictions.")
            predictions = _heuristic_predict(df)
            probabilities = None
            class_names = ['gaming', 'browsing', 'video_streaming']

        prediction_time = time.time() - start_time

        # Prepare results
        results = {
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            'processing_time_seconds': prediction_time,
            'confidence_threshold': confidence_threshold,
            'model_path': model,
            'total_flows': len(df)
        }

        if include_probabilities and probabilities is not None:
            results['probabilities'] = probabilities.tolist()
            results['class_names'] = class_names

        # Display results
        click.echo(f"\n=== PREDICTION RESULTS ===")
        click.echo(f"Total flows: {len(df)}")
        click.echo(f"Processing time: {prediction_time:.3f} seconds")

        # Class distribution
        if hasattr(predictions, 'value_counts'):
            class_dist = predictions.value_counts()
        else:
            unique, counts = np.unique(predictions, return_counts=True)
            class_dist = dict(zip(unique, counts))

        click.echo(f"\nClass distribution:")
        for class_name, count in class_dist.items():
            percentage = (count / len(predictions)) * 100
            click.echo(f"  {class_name}: {count} ({percentage:.1f}%)")

        # Save results if output specified
        if output:
            save_results(results, output)
            click.echo(f"\nResults saved to {output}")

        # Sample predictions
        click.echo(f"\nSample predictions:")
        for i in range(min(5, len(predictions))):
            pred = predictions[i]
            if probabilities is not None:
                conf = max(probabilities[i])
                click.echo(f"  Flow {i+1}: {pred} (confidence: {conf:.3f})")
            else:
                click.echo(f"  Flow {i+1}: {pred}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        click.echo(f"Error: {e}", err=True)
        return 1

def _heuristic_predict(df: pd.DataFrame) -> np.ndarray:
    """Simple heuristic predictions for demo."""
    predictions = []

    for _, row in df.iterrows():
        total_packets = row.get('Total Fwd Packets', 0) + row.get('Total Backward Packets', 0)
        flow_duration = row.get('Flow Duration', 1)

        if total_packets > 1000 and flow_duration > 60:
            pred = 'video_streaming'
        elif total_packets < 50:
            pred = 'texting'
        elif flow_duration < 5:
            pred = 'gaming'
        else:
            pred = 'browsing'

        predictions.append(pred)

    return np.array(predictions)

@cli.command()
@click.option('--ue-count', '-n', default=3, help='Number of UEs to simulate')
@click.option('--flows-per-ue', '-f', default=50, help='Flows per UE')
def demo(ue_count: int, flows_per_ue: int):
    """Run a demo prediction with synthetic data."""

    click.echo(f"🚀 Multi-UE Traffic Classification Demo")
    click.echo(f"Simulating {ue_count} UEs with {flows_per_ue} flows each\n")

    # Generate synthetic flow data
    np.random.seed(42)

    synthetic_data = []
    app_classes = ['gaming', 'video_streaming', 'browsing', 'audio_calls', 'video_calls']

    for ue_id in range(ue_count):
        for flow_id in range(flows_per_ue):
            # Random application class
            app_class = np.random.choice(app_classes)

            # Generate synthetic features based on app class
            if app_class == 'gaming':
                flow_duration = np.random.exponential(2)
                total_packets = np.random.poisson(100)
                bytes_per_packet = 80
            elif app_class == 'video_streaming':
                flow_duration = np.random.exponential(180)
                total_packets = np.random.poisson(2000)
                bytes_per_packet = 1200
            elif app_class == 'browsing':
                flow_duration = np.random.exponential(30)
                total_packets = np.random.poisson(200)
                bytes_per_packet = 800
            elif app_class == 'audio_calls':
                flow_duration = np.random.exponential(300)
                total_packets = np.random.poisson(1500)
                bytes_per_packet = 160
            else:  # video_calls
                flow_duration = np.random.exponential(600)
                total_packets = np.random.poisson(3000)
                bytes_per_packet = 1000

            total_bytes = total_packets * bytes_per_packet

            synthetic_data.append({
                'UE_ID': f'UE_{ue_id:03d}',
                'Flow_ID': flow_id,
                'Flow Duration': max(0.1, flow_duration),
                'Total Fwd Packets': max(1, int(total_packets * 0.6)),
                'Total Backward Packets': max(1, int(total_packets * 0.4)),
                'Total Length of Fwd Packets': int(total_bytes * 0.7),
                'Total Length of Bwd Packets': int(total_bytes * 0.3),
                'Flow Bytes/s': total_bytes / max(0.1, flow_duration),
                'Flow Packets/s': total_packets / max(0.1, flow_duration),
                'True_Class': app_class  # For demo comparison
            })

    df = pd.DataFrame(synthetic_data)

    # Run heuristic predictions
    predictions = _heuristic_predict(df)

    # Compare with true classes
    true_classes = df['True_Class'].values
    accuracy = (predictions == true_classes).mean()

    # Display results by UE
    click.echo("📊 Results by UE:")
    click.echo("-" * 50)

    for ue_id in range(ue_count):
        ue_mask = df['UE_ID'] == f'UE_{ue_id:03d}'
        ue_predictions = predictions[ue_mask]
        ue_true = true_classes[ue_mask]
        ue_accuracy = (ue_predictions == ue_true).mean()

        # Class distribution for this UE
        unique, counts = np.unique(ue_predictions, return_counts=True)
        class_dist = dict(zip(unique, counts))

        click.echo(f"UE_{ue_id:03d}: Accuracy {ue_accuracy:.2f} | Classes: {class_dist}")

    # Overall statistics
    click.echo("\n📈 Overall Statistics:")
    click.echo("-" * 30)
    click.echo(f"Total flows: {len(df)}")
    click.echo(f"Overall accuracy: {accuracy:.3f}")

    # Class-wise results
    from collections import Counter
    pred_dist = Counter(predictions)
    true_dist = Counter(true_classes)

    click.echo(f"\n🎯 Class Distribution:")
    click.echo("Class          | Predicted | Actual")
    click.echo("-" * 35)
    for class_name in app_classes:
        pred_count = pred_dist.get(class_name, 0)
        true_count = true_dist.get(class_name, 0)
        click.echo(f"{class_name:<14} | {pred_count:>9} | {true_count:>6}")

    click.echo(f"\n✅ Demo completed successfully!")

    # QoS priority mapping
    qos_mapping = {
        'gaming': 1, 'audio_calls': 2, 'video_calls': 3,
        'video_streaming': 4, 'browsing': 5
    }

    click.echo(f"\n🚦 QoS Priority Distribution:")
    priority_dist = Counter([qos_mapping.get(pred, 5) for pred in predictions])
    for priority in sorted(priority_dist.keys()):
        count = priority_dist[priority]
        percentage = (count / len(predictions)) * 100
        click.echo(f"  Priority {priority}: {count} flows ({percentage:.1f}%)")

def predict_from_csv(csv_path: str, model_path: str = "models/traffic_classifier.joblib") -> Dict[str, Any]:
    """Predict from CSV file (programmatic interface)."""

    try:
        # Load data and model
        df = load_data(csv_path)
        classifier = MultiUETrafficClassifier.load(model_path)

        # Make predictions
        if classifier.is_trained:
            predictions = classifier.predict(df)
            probabilities, class_names = classifier.predict_proba(df)
        else:
            predictions = _heuristic_predict(df)
            probabilities = None
            class_names = None

        return {
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'class_names': class_names,
            'total_flows': len(df)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == '__main__':
    cli()
