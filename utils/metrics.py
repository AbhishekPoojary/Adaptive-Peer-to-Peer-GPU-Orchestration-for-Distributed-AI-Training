"""
utils/metrics.py
----------------
In-memory metrics collector for academic measurement.
Tracks total training time, recovery time, and scheduler decisions.
Metrics are also written to logs/metrics.jsonl (newline-delimited JSON).
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

METRICS_FILE = Path("./logs/metrics.jsonl")


@dataclass
class JobMetric:
    job_id:          int
    job_name:        str
    scheduler:       str
    assigned_node:   str
    submitted_at:    float       # epoch seconds
    started_at:      Optional[float] = None
    completed_at:    Optional[float] = None
    recovery_time_s: Optional[float] = None
    total_time_s:    Optional[float] = None
    recovered:       bool        = False


class MetricsCollector:
    """Singleton-style in-process metrics store."""

    _instance: Optional["MetricsCollector"] = None

    def __init__(self):
        self._jobs: Dict[int, JobMetric] = {}
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Job lifecycle helpers
    # ------------------------------------------------------------------

    def record_submission(self, job_id: int, job_name: str,
                          scheduler: str, node_id: str) -> None:
        m = JobMetric(
            job_id=job_id,
            job_name=job_name,
            scheduler=scheduler,
            assigned_node=node_id,
            submitted_at=time.time(),
        )
        self._jobs[job_id] = m
        self._flush(m)
        logger.info("METRIC | submission | job=%d node=%s scheduler=%s",
                    job_id, node_id, scheduler)

    def record_start(self, job_id: int) -> None:
        m = self._jobs.get(job_id)
        if m:
            m.started_at = time.time()
            self._flush(m)

    def record_completion(self, job_id: int) -> None:
        m = self._jobs.get(job_id)
        if m:
            m.completed_at = time.time()
            if m.started_at:
                m.total_time_s = m.completed_at - m.started_at
            self._flush(m)
            logger.info("METRIC | completion | job=%d total_time=%.1fs",
                        job_id, m.total_time_s or 0)

    def record_recovery(self, job_id: int, recovery_time_s: float,
                        new_node_id: str) -> None:
        m = self._jobs.get(job_id)
        if m:
            m.recovery_time_s = recovery_time_s
            m.assigned_node   = new_node_id
            m.recovered       = True
            self._flush(m)
            logger.info("METRIC | recovery   | job=%d time=%.1fs new_node=%s",
                        job_id, recovery_time_s, new_node_id)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        completed = [m for m in self._jobs.values() if m.completed_at]
        recovered = [m for m in self._jobs.values() if m.recovered]
        return {
            "total_jobs":           len(self._jobs),
            "completed_jobs":       len(completed),
            "recovered_jobs":       len(recovered),
            "avg_total_time_s":     (
                sum(m.total_time_s or 0 for m in completed) / len(completed)
                if completed else 0
            ),
            "avg_recovery_time_s":  (
                sum(m.recovery_time_s or 0 for m in recovered) / len(recovered)
                if recovered else 0
            ),
        }

    # ------------------------------------------------------------------

    def _flush(self, m: JobMetric) -> None:
        """Append metric snapshot to JSONL file for offline analysis."""
        try:
            row = asdict(m)
            row["_ts"] = datetime.utcnow().isoformat()
            with METRICS_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception as exc:
            logger.warning("MetricsCollector flush error: %s", exc)
