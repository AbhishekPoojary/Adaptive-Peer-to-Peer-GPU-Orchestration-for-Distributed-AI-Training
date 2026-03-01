"""
orchestrator/heartbeat_monitor.py
----------------------------------
Background asyncio task that periodically sweeps for nodes that have
stopped sending heartbeats and triggers job recovery.

Production changes:
  - Calls recover_orphaned_jobs() on first sweep (handles orchestrator restart)
  - Sweep interval configurable via HEARTBEAT_SWEEP_S env var
"""

import asyncio
import logging

from database.db import SessionLocal
from database.models import JobStatus
from orchestrator.node_manager import NodeManager
from utils.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


async def heartbeat_monitor_loop(job_manager_instance) -> None:
    """
    Infinite async loop – runs as a FastAPI background task (lifespan).
    On first iteration: recovers any orphaned jobs from previous orchestrator run.
    """
    timeout  = _settings.heartbeat_timeout_s
    interval = _settings.heartbeat_sweep_s

    logger.info(
        "HeartbeatMonitor started (timeout=%.0fs, sweep=%.0fs)",
        timeout, interval,
    )

    first_sweep = True
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(_run_sweep, job_manager_instance, timeout, first_sweep)
            first_sweep = False
        except Exception as exc:
            logger.exception("HeartbeatMonitor sweep error: %s", exc)


def _run_sweep(job_manager, timeout_s: float, recover_orphans: bool = False) -> None:
    """One monitoring sweep – executed in a thread pool (SQLAlchemy is sync)."""
    db = SessionLocal()
    try:
        # On first sweep: recover jobs left running from before last restart
        if recover_orphans:
            recovered = job_manager.recover_orphaned_jobs(db)
            if recovered:
                logger.info("Startup recovery: %d orphaned job(s) rescheduled.", recovered)

        # Regular heartbeat timeout detection
        timed_out = NodeManager.get_timed_out_nodes(db, timeout_s, grace_s=_settings.registration_grace_s)
        for node in timed_out:
            logger.warning(
                "HeartbeatMonitor: node %s timed out (last seen %s)",
                node.id, node.last_heartbeat.isoformat(),
            )
            NodeManager.mark_node_failed(db, node)

            # Recover all running jobs on this failed node
            from database.models import TrainingJob
            running_jobs = db.query(TrainingJob).filter(
                TrainingJob.assigned_node == node.id,
                TrainingJob.status        == JobStatus.RUNNING,
            ).all()

            for job in running_jobs:
                logger.warning(
                    "HeartbeatMonitor: recovering job %d from failed node %s",
                    job.id, node.id,
                )
                job_manager.recover_job(db, job.id, node.id)
    finally:
        db.close()
