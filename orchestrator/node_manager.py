"""
orchestrator/node_manager.py
-----------------------------
Manages GPU node registration, state updates, heartbeat tracking,
reliability scoring, and failure detection helpers.

Production changes:
  - register_node accepts optional api_key_hint for audit trail
  - get_timed_out_nodes: FAILED nodes excluded from timeout sweep
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session
from database.models import GPUNode, NodeStatus, ReliabilityHistory

logger = logging.getLogger(__name__)


class NodeManager:
    """CRUD + business logic for GPUNode entities."""

    @staticmethod
    def register_node(
        db: Session,
        node_id: str,
        host: str,
        port: int,
        gpu_memory_mb: float,
        api_key_hint: Optional[str] = None,
    ) -> GPUNode:
        """
        Upsert a node registration record.
        On re-registration: restores ONLINE status and refreshes all fields.
        """
        existing = db.query(GPUNode).filter(GPUNode.id == node_id).first()
        if existing:
            existing.host           = host
            existing.port           = port
            existing.gpu_memory_mb  = gpu_memory_mb
            existing.status         = NodeStatus.ONLINE
            existing.last_heartbeat = datetime.utcnow()
            if api_key_hint:
                existing.api_key_hint = api_key_hint
            db.commit()
            db.refresh(existing)
            logger.info("Node re-registered: %s @ %s:%d", node_id, host, port)
            return existing

        node = GPUNode(
            id=node_id,
            host=host,
            port=port,
            gpu_memory_mb=gpu_memory_mb,
            status=NodeStatus.ONLINE,
            last_heartbeat=datetime.utcnow(),
            registered_at=datetime.utcnow(),
            api_key_hint=api_key_hint,
        )
        db.add(node)
        db.commit()
        db.refresh(node)
        logger.info(
            "Node registered: %s @ %s:%d (%.0f MB GPU) key_hint=%s",
            node_id, host, port, gpu_memory_mb, api_key_hint or "n/a",
        )
        return node

    @staticmethod
    def update_heartbeat(
        db: Session, node_id: str, current_load: float
    ) -> Optional[GPUNode]:
        """Record a heartbeat and update the node's current load."""
        node = db.query(GPUNode).filter(GPUNode.id == node_id).first()
        if not node:
            logger.warning("Heartbeat from unknown node: %s", node_id)
            return None
        node.last_heartbeat = datetime.utcnow()
        node.current_load   = max(0.0, min(1.0, current_load))
        # Mark agent as connected on first real heartbeat
        if not node.agent_connected:
            node.agent_connected = 1
            logger.info("Node '%s' agent connected for the first time.", node_id)
        # Auto-recover a node that comes back online after transient failure
        if node.status == NodeStatus.FAILED:
            node.status = NodeStatus.ONLINE
            logger.info("Node '%s' came back online after failure.", node_id)
        db.commit()
        return node

    @staticmethod
    def get_timed_out_nodes(db: Session, timeout_s: float, grace_s: float = 120.0) -> List[GPUNode]:
        """Return ONLINE nodes whose last heartbeat is older than timeout_s.
        Newly registered nodes are exempt for grace_s seconds after registration."""
        now = datetime.utcnow()
        nodes = db.query(GPUNode).filter(
            GPUNode.status == NodeStatus.ONLINE
        ).all()
        timed_out = []
        for n in nodes:
            # Grace period: skip nodes that registered recently
            reg_at = n.registered_at
            if reg_at.tzinfo is not None:
                reg_at = reg_at.replace(tzinfo=None)
            if (now - reg_at).total_seconds() < grace_s:
                continue  # still within grace window – don't time out yet

            lhb = n.last_heartbeat
            if lhb.tzinfo is not None:
                lhb = lhb.replace(tzinfo=None)
            if (now - lhb).total_seconds() > timeout_s:
                timed_out.append(n)
        return timed_out

    @staticmethod
    def mark_node_failed(db: Session, node: GPUNode) -> None:
        node.status = NodeStatus.FAILED
        db.commit()
        logger.warning("Node marked FAILED: %s", node.id)

    @staticmethod
    def mark_node_offline(db: Session, node_id: str) -> None:
        """Graceful shutdown: mark OFFLINE (not FAILED) so no recovery is triggered."""
        node = db.query(GPUNode).filter(GPUNode.id == node_id).first()
        if node:
            node.status = NodeStatus.OFFLINE
            db.commit()
            logger.info("Node marked OFFLINE (graceful shutdown): %s", node_id)

    @staticmethod
    def record_job_outcome(db: Session, node_id: str, success: bool) -> None:
        """Update reliability score after a job finishes."""
        node = db.query(GPUNode).filter(GPUNode.id == node_id).first()
        if not node:
            return
        node.total_jobs += 1
        if success:
            node.successful_jobs += 1
        if node.total_jobs > 0:
            node.reliability_score = node.successful_jobs / node.total_jobs

        hist = ReliabilityHistory(
            node_id=node_id,
            reliability_score=node.reliability_score,
            total_jobs=node.total_jobs,
            successful_jobs=node.successful_jobs,
        )
        db.add(hist)
        db.commit()
        logger.info(
            "Reliability updated: node=%s rel=%.3f (%d/%d)",
            node_id, node.reliability_score,
            node.successful_jobs, node.total_jobs,
        )

    @staticmethod
    def get_online_nodes(db: Session) -> List[GPUNode]:
        return db.query(GPUNode).filter(GPUNode.status == NodeStatus.ONLINE).all()

    @staticmethod
    def get_all_nodes(db: Session) -> List[GPUNode]:
        return db.query(GPUNode).all()

    @staticmethod
    def get_node(db: Session, node_id: str) -> Optional[GPUNode]:
        return db.query(GPUNode).filter(GPUNode.id == node_id).first()
