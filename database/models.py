"""
database/models.py
------------------
SQLAlchemy ORM models for all persistent entities:
  - GPUNode            : Registered peer node metadata + reliability tracking
  - TrainingJob        : Training job lifecycle (queued → running → completed/failed)
  - SchedulingLog      : Per-assignment scheduling decisions (immutable audit trail)
  - ReliabilityHistory : Time-series reliability snapshots for each node
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Text, Enum, Index
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


class NodeStatus(str, enum.Enum):
    ONLINE  = "online"
    OFFLINE = "offline"
    FAILED  = "failed"


class JobStatus(str, enum.Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    RECOVERING = "recovering"


class GPUNode(Base):
    """Represents a registered GPU peer node."""
    __tablename__ = "gpu_nodes"

    id                = Column(String, primary_key=True)          # e.g. "node-1"
    host              = Column(String, nullable=False)
    port              = Column(Integer, nullable=False)
    gpu_memory_mb     = Column(Float, default=0.0)                # total GPU VRAM in MB
    current_load      = Column(Float, default=0.0)                # 0.0 – 1.0
    reliability_score = Column(Float, default=1.0)                # 0.0 – 1.0
    total_jobs        = Column(Integer, default=0)
    successful_jobs   = Column(Integer, default=0)
    status            = Column(String, default=NodeStatus.ONLINE)
    last_heartbeat    = Column(DateTime, default=datetime.utcnow)
    registered_at     = Column(DateTime, default=datetime.utcnow)
    api_key_hint      = Column(String(8), nullable=True)          # first 8 chars of registering key
    agent_connected   = Column(Integer, default=0)               # 0 = pending, 1 = agent has sent heartbeat

    jobs              = relationship("TrainingJob", back_populates="node")
    reliability_hist  = relationship("ReliabilityHistory", back_populates="node")
    scheduling_logs   = relationship("SchedulingLog", back_populates="node")

    def __repr__(self):
        return (f"<GPUNode id={self.id} host={self.host}:{self.port} "
                f"load={self.current_load:.2f} rel={self.reliability_score:.2f}>")


class TrainingJob(Base):
    """Represents a submitted training job and its lifecycle."""
    __tablename__ = "training_jobs"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    job_name        = Column(String, nullable=False)
    dataset         = Column(String, default="MNIST")
    epochs          = Column(Integer, default=3)
    batch_size      = Column(Integer, default=64)
    checkpoint_path = Column(String, nullable=True)              # path to latest checkpoint
    resume_epoch    = Column(Integer, default=0)                 # epoch to resume from
    status          = Column(String, default=JobStatus.QUEUED)
    assigned_node   = Column(String, ForeignKey("gpu_nodes.id"), nullable=True)
    submitted_at    = Column(DateTime, default=datetime.utcnow)
    started_at      = Column(DateTime, nullable=True)
    completed_at    = Column(DateTime, nullable=True)
    recovery_time_s = Column(Float, nullable=True)               # seconds to recover
    error_message   = Column(Text, nullable=True)

    node            = relationship("GPUNode", back_populates="jobs")
    scheduling_logs = relationship("SchedulingLog", back_populates="job")

    # ── Indexes for common query patterns ────────────────────────────────────
    __table_args__ = (
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_assigned_node", "assigned_node"),
        Index("ix_jobs_status_node", "status", "assigned_node"),
        Index("ix_jobs_submitted_at", "submitted_at"),
    )

    def __repr__(self):
        return (f"<TrainingJob id={self.id} name={self.job_name} "
                f"status={self.status} node={self.assigned_node}>")


class SchedulingLog(Base):
    """Immutable audit log of each scheduling decision."""
    __tablename__ = "scheduling_logs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    job_id         = Column(Integer, ForeignKey("training_jobs.id"))
    node_id        = Column(String, ForeignKey("gpu_nodes.id"))
    scheduler_type = Column(String, nullable=False)             # "round_robin" | "least_loaded" | "adaptive"
    score          = Column(Float, nullable=True)               # computed score (adaptive only)
    decision_at    = Column(DateTime, default=datetime.utcnow)
    reason         = Column(Text, nullable=True)

    job  = relationship("TrainingJob", back_populates="scheduling_logs")
    node = relationship("GPUNode", back_populates="scheduling_logs")


class ReliabilityHistory(Base):
    """Time-series snapshots of a node's reliability score."""
    __tablename__ = "reliability_history"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    node_id           = Column(String, ForeignKey("gpu_nodes.id"))
    reliability_score = Column(Float, nullable=False)
    total_jobs        = Column(Integer, default=0)
    successful_jobs   = Column(Integer, default=0)
    recorded_at       = Column(DateTime, default=datetime.utcnow)

    node = relationship("GPUNode", back_populates="reliability_hist")

    __table_args__ = (
        Index("ix_rel_history_node_id", "node_id"),
        Index("ix_rel_history_recorded_at", "recorded_at"),
    )
