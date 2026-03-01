"""
scheduler/round_robin.py
------------------------
Round-Robin Scheduler: assigns jobs to nodes in circular order.
State is maintained via a class-level counter so it persists across
multiple calls within the same process lifetime.
"""

import logging
from typing import List, Optional
from database.models import GPUNode
from scheduler.base import BaseScheduler

logger = logging.getLogger(__name__)


class RoundRobinScheduler(BaseScheduler):
    """
    Distributes jobs evenly across all available nodes by cycling
    through them in order. No load or reliability weighting is applied.
    """

    _counter: int = 0          # class-level index shared across instances

    def select_node(self, nodes: List[GPUNode], job_id: int) -> Optional[GPUNode]:
        if not nodes:
            logger.warning("RoundRobin: no online nodes available for job %d", job_id)
            return None

        index = RoundRobinScheduler._counter % len(nodes)
        RoundRobinScheduler._counter += 1

        selected = nodes[index]
        logger.info(
            "RoundRobin: job=%d → node=%s (index %d / %d)",
            job_id, selected.id, index, len(nodes)
        )
        return selected

    @property
    def name(self) -> str:
        return "round_robin"
