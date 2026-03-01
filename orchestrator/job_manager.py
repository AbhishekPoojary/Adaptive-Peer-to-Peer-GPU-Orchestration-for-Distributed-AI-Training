"""
orchestrator/job_manager.py
----------------------------
Job queuing, scheduling dispatch, and fault-tolerant job reassignment.

Production changes:
  - Async dispatch via asyncio.to_thread (non-blocking event loop)
  - Exponential backoff retry on agent dispatch (3 attempts)
  - On startup recovery: detects RUNNING jobs from before restart
"""

import asyncio
import logging
import time
import httpx
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from database.models import TrainingJob, JobStatus, SchedulingLog
from orchestrator.node_manager import NodeManager
from scheduler.scheduler_factory import get_scheduler
from utils.settings import get_settings
from utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)
_settings = get_settings()
_scheduler_type = _settings.scheduler_type


class JobManager:
    """Assigns jobs to nodes and handles recovery when nodes fail."""

    def __init__(self):
        self._scheduler = get_scheduler(_scheduler_type)
        self._metrics   = MetricsCollector.get()
        logger.info("JobManager initialised with scheduler='%s'", _scheduler_type)

    # ── Job submission ─────────────────────────────────────────────────────────

    def submit_job(
        self, db: Session, job_name: str,
        dataset: str = "MNIST", epochs: int = 3, batch_size: int = 64,
    ) -> TrainingJob:
        """Create a new job record and immediately attempt scheduling."""
        job = TrainingJob(
            job_name=job_name,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            status=JobStatus.QUEUED,
            submitted_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        logger.info("Job submitted: id=%d name='%s'", job.id, job.job_name)
        self._schedule_job(db, job)
        return job

    # ── Scheduling ────────────────────────────────────────────────────────────

    def _schedule_job(
        self, db: Session, job: TrainingJob,
        exclude_node: Optional[str] = None,
    ) -> bool:
        """Select a node and dispatch the job. Returns True on success."""
        online_nodes = NodeManager.get_online_nodes(db)
        if exclude_node:
            online_nodes = [n for n in online_nodes if n.id != exclude_node]

        selected = self._scheduler.select_node(online_nodes, job.id)
        if selected is None:
            job.status        = JobStatus.FAILED
            job.error_message = "No online nodes available at scheduling time."
            db.commit()
            logger.error("Job %d could not be scheduled – no online nodes.", job.id)
            return False

        log_entry = SchedulingLog(
            job_id=job.id,
            node_id=selected.id,
            scheduler_type=self._scheduler.name,
            score=None,
            decision_at=datetime.utcnow(),
            reason=f"Scheduled by {self._scheduler.name}",
        )
        db.add(log_entry)

        job.assigned_node = selected.id
        job.status        = JobStatus.RUNNING
        job.started_at    = datetime.utcnow()
        db.commit()

        self._metrics.record_submission(job.id, job.job_name, self._scheduler.name, selected.id)
        self._metrics.record_start(job.id)

        # Non-blocking dispatch (runs in a thread pool to avoid blocking the event loop)
        asyncio.get_event_loop().run_in_executor(
            None, self._dispatch_with_retry, selected.host, selected.port, job
        )
        return True

    def _dispatch_with_retry(self, host: str, port: int, job: TrainingJob) -> None:
        """
        POST training request to the GPU agent with exponential backoff retry.
        Attempts: 1s → 2s → 4s delays between retries.
        """
        url = f"http://{host}:{port}/run_job"
        payload = {
            "job_id":          job.id,
            "job_name":        job.job_name,
            "dataset":         job.dataset,
            "epochs":          job.epochs,
            "batch_size":      job.batch_size,
            "checkpoint_path": job.checkpoint_path,
            "resume_epoch":    job.resume_epoch,
        }
        delays = [1, 2, 4]
        for attempt, delay in enumerate(delays, start=1):
            try:
                resp = httpx.post(
                    url, json=payload,
                    timeout=10.0,
                    headers={"X-API-Key": _settings.orchestrator_api_key},
                )
                resp.raise_for_status()
                logger.info(
                    "Job %d dispatched to %s:%d (attempt %d)",
                    job.id, host, port, attempt,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Dispatch attempt %d/%d for job %d to %s:%d failed: %s",
                    attempt, len(delays), job.id, host, port, exc,
                )
                if attempt < len(delays):
                    time.sleep(delay)

        logger.error(
            "All %d dispatch attempts failed for job %d to %s:%d.",
            len(delays), job.id, host, port,
        )

    # ── Completion callback ───────────────────────────────────────────────────

    def handle_completion(
        self, db: Session, job_id: int, node_id: str,
        success: bool, checkpoint_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            logger.warning("Completion for unknown job_id=%d", job_id)
            return
        job.status        = JobStatus.COMPLETED if success else JobStatus.FAILED
        job.completed_at  = datetime.utcnow()
        job.error_message = error
        if checkpoint_path:
            job.checkpoint_path = checkpoint_path
        db.commit()
        NodeManager.record_job_outcome(db, node_id, success)
        if success:
            self._metrics.record_completion(job.id)
        logger.info("Job %d completion: success=%s node=%s", job_id, success, node_id)

    # ── Fault recovery ────────────────────────────────────────────────────────

    def recover_job(self, db: Session, job_id: int, failed_node_id: str) -> bool:
        """
        Reschedule a job from its last checkpoint after node failure.
        Called by HeartbeatMonitor and on orchestrator restart.
        """
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job or job.status not in (JobStatus.RUNNING, JobStatus.RECOVERING):
            return False

        t_start = time.time()
        job.status = JobStatus.RECOVERING
        db.commit()
        logger.warning("Recovering job %d (was on %s)", job_id, failed_node_id)

        success = self._schedule_job(db, job, exclude_node=failed_node_id)
        recovery_time = time.time() - t_start

        if success:
            job.recovery_time_s = recovery_time
            db.commit()
            self._metrics.record_recovery(job_id, recovery_time, job.assigned_node or "")
            logger.info("Job %d recovered in %.2fs", job_id, recovery_time)
        else:
            job.status = JobStatus.FAILED
            db.commit()
            logger.error("Job %d recovery FAILED – no eligible nodes.", job_id)

        return success

    def get_online_schedulable_nodes(self, db: Session) -> list[GPUNode]:
        """Return a list of ONLINE nodes that have actually connected their agent."""
        return db.query(GPUNode).filter(
            GPUNode.status == NodeStatus.ONLINE,
            GPUNode.agent_connected == 1
        ).all()

    def recover_orphaned_jobs(self, db: Session) -> int:
        """
        Called at orchestrator startup: finds jobs left RUNNING/RECOVERING
        from before the last restart and triggers recovery.
        Returns the number of jobs recovered.
        """
        orphans = db.query(TrainingJob).filter(
            TrainingJob.status.in_([JobStatus.RUNNING, JobStatus.RECOVERING])
        ).all()
        if not orphans:
            return 0
        logger.warning(
            "Found %d orphaned jobs from previous orchestrator run. Recovering...",
            len(orphans),
        )
        count = 0
        for job in orphans:
            failed_node = job.assigned_node or ""
            if self.recover_job(db, job.id, failed_node):
                count += 1
        return count

    @staticmethod
    def get_running_jobs_on_node(db: Session, node_id: str):
        return db.query(TrainingJob).filter(
            TrainingJob.assigned_node == node_id,
            TrainingJob.status        == JobStatus.RUNNING,
        ).all()
