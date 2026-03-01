"""
scheduler/adaptive.py
---------------------
Reliability-Aware Adaptive Scheduler.

Scoring formula (lower is better):
    Score_i = alpha * Load_i - beta * Reliability_i + gamma * Latency_i

Where Latency_i is approximated as seconds since last heartbeat.
alpha, beta, gamma are configurable via config.yaml (loaded at import time).
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

from database.models import GPUNode
from scheduler.base import BaseScheduler
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

_cfg = load_config()
_sched_cfg = _cfg.get("scheduler", {}).get("adaptive", {})

DEFAULT_ALPHA = float(_sched_cfg.get("alpha", 0.5))
DEFAULT_BETA  = float(_sched_cfg.get("beta",  0.4))
DEFAULT_GAMMA = float(_sched_cfg.get("gamma", 0.1))


class AdaptiveScheduler(BaseScheduler):
    """
    Selects the node with the **minimum** composite score:

        Score_i = alpha * Load_i - beta * Reliability_i + gamma * Latency_i

    - Load_i        : current_load (0.0–1.0)
    - Reliability_i : reliability_score (0.0–1.0, negated → lower score for reliable nodes)
    - Latency_i     : seconds since last heartbeat (normalised by dividing by 60)

    Parameters
    ----------
    alpha, beta, gamma : float
        Weighting coefficients; defaults sourced from config.yaml.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta:  float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        logger.info(
            "AdaptiveScheduler initialised: alpha=%.2f beta=%.2f gamma=%.2f",
            alpha, beta, gamma,
        )

    def _score(self, node: GPUNode) -> float:
        """Compute composite score for a single node (lower = better)."""
        now = datetime.utcnow()
        # Convert last_heartbeat to offset-naive if necessary
        lhb = node.last_heartbeat
        if lhb.tzinfo is not None:
            lhb = lhb.replace(tzinfo=None)
        latency_s = (now - lhb).total_seconds()
        latency_norm = latency_s / 60.0   # normalise to [0, ~1] range

        score = (
            self.alpha * node.current_load
            - self.beta  * node.reliability_score
            + self.gamma * latency_norm
        )
        return score

    def select_node(self, nodes: List[GPUNode], job_id: int) -> Optional[GPUNode]:
        if not nodes:
            logger.warning("Adaptive: no online nodes available for job %d", job_id)
            return None

        scored = [(node, self._score(node)) for node in nodes]
        scored.sort(key=lambda x: x[1])   # ascending → best node first

        selected, best_score = scored[0]

        # Log all candidates for academic inspection
        for node, sc in scored:
            logger.debug(
                "Adaptive candidate: node=%s load=%.2f rel=%.2f score=%.4f",
                node.id, node.current_load, node.reliability_score, sc,
            )
        logger.info(
            "Adaptive: job=%d → node=%s (score=%.4f, load=%.2f, rel=%.2f)",
            job_id, selected.id, best_score,
            selected.current_load, selected.reliability_score,
        )
        return selected

    @property
    def name(self) -> str:
        return "adaptive"
