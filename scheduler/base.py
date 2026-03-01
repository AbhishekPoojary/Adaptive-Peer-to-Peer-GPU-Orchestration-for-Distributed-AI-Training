"""
scheduler/base.py
-----------------
Abstract base class that all scheduler implementations must extend.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from database.models import GPUNode


class BaseScheduler(ABC):
    """Common interface for all scheduling algorithms."""

    @abstractmethod
    def select_node(self, nodes: List[GPUNode], job_id: int) -> Optional[GPUNode]:
        """
        Given a list of *online* candidate nodes, select the best one
        for the provided job.

        Args:
            nodes   : List of GPUNode ORM objects (already filtered to ONLINE).
            job_id  : The ID of the job being scheduled (for logging).

        Returns:
            The chosen GPUNode, or None if no suitable node exists.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
