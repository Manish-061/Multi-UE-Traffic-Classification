"""Input/Output utilities."""

import json
import yaml
import pickle
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Union

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def save_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save results to JSON file.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_model(model_path: Union[str, Path]) -> Any:
    """Load trained model from file.

    Args:
        model_path: Path to model file

    Returns:
        Loaded model
    """
    model_path = Path(model_path)

    if model_path.suffix == '.joblib':
        return joblib.load(model_path)
    elif model_path.suffix == '.pkl':
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

def save_model(model: Any, model_path: Union[str, Path]) -> None:
    """Save model to file.

    Args:
        model: Trained model
        model_path: Output path
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.suffix == '.joblib':
        joblib.dump(model, model_path)
    elif model_path.suffix == '.pkl':
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from various formats.

    Args:
        data_path: Path to data file

    Returns:
        Loaded DataFrame
    """
    data_path = Path(data_path)

    if data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    elif data_path.suffix == '.json':
        return pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
