"""Timing utilities for performance measurement."""

import time
from contextlib import contextmanager
from typing import Optional

class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

        if self.name:
            print(f"{self.name}: {self.elapsed_time:.4f} seconds")

    @property
    def elapsed_ms(self) -> Optional[float]:
        """Get elapsed time in milliseconds."""
        if self.elapsed_time is not None:
            return self.elapsed_time * 1000
        return None

@contextmanager
def time_function(name: str):
    """Context manager to time function execution.

    Args:
        name: Name for the timer
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"{name}: {elapsed:.4f} seconds ({elapsed*1000:.2f} ms)")
