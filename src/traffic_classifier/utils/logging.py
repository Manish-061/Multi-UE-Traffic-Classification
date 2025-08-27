"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

def get_logger(
    name: str, 
    level: str = "INFO", 
    log_file: Optional[str] = None
) -> logging.Logger:
    """Get configured logger instance.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Don't add handlers if they already exist
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger

# Default logger
logger = get_logger("traffic_classifier")
