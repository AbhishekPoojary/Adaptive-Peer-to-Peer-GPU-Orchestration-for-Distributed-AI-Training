"""
scheduler/least_loaded.py
--------------------------
Least-Loaded Scheduler: selects the node with the lowest current_load.
Ties are broken by reliability score (higher is better).
"""

import logging
from typing import List, Optional
from database.models import GPUNode
from scheduler.base import BaseScheduler

logger = logging.getLogger(__name__)


class LeastLoadedScheduler(BaseScheduler):
    """
    Picks the node with the minimum current_load (0.0 – 1.0).
    In case of ties, higher reliability wins.
    """

    def select_node(self, nodes: List[GPUNode], job_id: int) -> Optional[GPUNode]:
        if not nodes:
            logger.warning("LeastLoaded: no online nodes available for job %d", job_id)
            return None

        # Primary sort: ascending load; secondary (tie-break): descending reliability
        selected = min(nodes, key=lambda n: (n.current_load, -n.reliability_score))

        logger.info(
            "LeastLoaded: job=%d → node=%s (load=%.2f, reliability=%.2f)",
            job_id, selected.id, selected.current_load, selected.reliability_score
        )
        return selected

    @property
    def name(self) -> str:
        return "least_loaded"
