"""
utils/config_loader.py
----------------------
Loads config.yaml from the project root (or the path in CONFIG_PATH env var).
Returns a plain dict; callers access nested keys with .get().
"""

import os
import yaml
from pathlib import Path

_CONFIG_CACHE: dict = {}


def load_config(path: str = None) -> dict:
    """
    Load and cache the YAML configuration file.

    Args:
        path: Optional override path.  Defaults to CONFIG_PATH env var,
              then ./config.yaml relative to CWD.

    Returns:
        Parsed config as a nested dict.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE:
        return _CONFIG_CACHE

    config_path = Path(
        path
        or os.getenv("CONFIG_PATH", "config.yaml")
    )

    if not config_path.exists():
        # Return sensible defaults when file is missing (e.g. during tests)
        _CONFIG_CACHE = _defaults()
        return _CONFIG_CACHE

    with config_path.open("r", encoding="utf-8") as fh:
        _CONFIG_CACHE = yaml.safe_load(fh) or {}

    return _CONFIG_CACHE


def _defaults() -> dict:
    return {
        "orchestrator": {
            "host": "0.0.0.0",
            "port": 8000,
            "heartbeat_timeout_s": 15,
            "scheduler_type": "adaptive",
        },
        "scheduler": {
            "adaptive": {"alpha": 0.5, "beta": 0.4, "gamma": 0.1}
        },
        "agent": {
            "heartbeat_interval_s": 5,
            "orchestrator_url": "http://127.0.0.1:8000",
        },
        "training": {
            "default_dataset": "MNIST",
            "default_epochs": 3,
            "default_batch_size": 64,
            "checkpoint_dir": "./checkpoints",
        },
        "logging": {"level": "INFO"},
    }
