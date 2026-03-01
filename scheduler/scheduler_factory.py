"""
scheduler/scheduler_factory.py
-------------------------------
Factory function that returns the appropriate scheduler instance
based on a string key read from config.yaml.
"""

import logging
from scheduler.base import BaseScheduler
from scheduler.round_robin import RoundRobinScheduler
from scheduler.least_loaded import LeastLoadedScheduler
from scheduler.adaptive import AdaptiveScheduler

logger = logging.getLogger(__name__)

_REGISTRY = {
    "round_robin":   RoundRobinScheduler,
    "least_loaded":  LeastLoadedScheduler,
    "adaptive":      AdaptiveScheduler,
}


def get_scheduler(scheduler_type: str = "adaptive") -> BaseScheduler:
    """
    Return an instantiated scheduler.

    Args:
        scheduler_type: one of "round_robin", "least_loaded", "adaptive"

    Returns:
        Concrete BaseScheduler implementation.

    Raises:
        ValueError: if scheduler_type is not registered.
    """
    key = scheduler_type.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown scheduler '{key}'. "
            f"Valid options: {list(_REGISTRY.keys())}"
        )
    instance = _REGISTRY[key]()
    logger.info("Scheduler selected: %s", instance.name)
    return instance
