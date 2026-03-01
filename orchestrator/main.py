"""
orchestrator/main.py
---------------------
Central Orchestrator Server – production-grade FastAPI application.

API Routes (prefix: /api/v1):
  GET  /health                → liveness + DB check (public)
  GET  /api/v1/nodes          → list all registered nodes (public)
  GET  /api/v1/jobs           → list jobs with pagination (public)
  GET  /api/v1/jobs/{job_id}  → single job detail (public)
  GET  /api/v1/metrics/summary
  GET  /api/v1/metrics/nodes

  POST /api/v1/register_node  → GPU agent self-registration  [AUTH]
  POST /api/v1/heartbeat      → periodic liveness ping        [AUTH]
  POST /api/v1/submit_job     → submit a training job         [AUTH]
  POST /api/v1/job_complete   → agent reports outcome         [AUTH]
  DELETE /api/v1/nodes/{id}   → deregister a node            [AUTH]
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import (
    FastAPI, Depends, HTTPException, Query, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth.dependencies import require_api_key
from database.db import init_db, get_db
from database.models import TrainingJob, JobStatus, ReliabilityHistory
from orchestrator.node_manager import NodeManager
from orchestrator.job_manager import JobManager
from orchestrator.heartbeat_monitor import heartbeat_monitor_loop
from utils.logger import setup_logging
from utils.middleware import RequestLoggingMiddleware
from utils.settings import get_settings
from utils.metrics import MetricsCollector

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_settings = get_settings()
setup_logging(_settings.log_level, _settings.log_format)
logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_job_manager = JobManager()


# ── FastAPI lifespan (startup / shutdown) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    init_db()
    logger.info("Database initialised (url=%s)", _settings.database_url.split("@")[-1])
    monitor_task = asyncio.create_task(heartbeat_monitor_loop(_job_manager))
    logger.info(
        "Orchestrator ready  scheduler=%s  timeout=%ss  api_key_hint=%s...",
        _settings.scheduler_type,
        _settings.heartbeat_timeout_s,
        _settings.orchestrator_api_key[:4],
    )
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    logger.info("Orchestrator shut down cleanly.")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GPU Orchestrator",
    description=(
        "**Adaptive P2P GPU Orchestration** – production-ready distributed "
        "training scheduler with fault tolerance and reliability-aware scheduling.\n\n"
        "Write endpoints require `X-API-Key` header."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.get_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Static frontend ───────────────────────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/ui", include_in_schema=False)
def serve_ui():
    index = _FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class NodeRegistrationRequest(BaseModel):
    node_id:       str   = Field(..., examples=["node-1"])
    host:          str   = Field(..., examples=["192.168.1.55"])
    port:          int   = Field(..., ge=1024, le=65535, examples=[8001])
    gpu_memory_mb: float = Field(default=0.0, ge=0.0, examples=[8192.0])


class HeartbeatRequest(BaseModel):
    node_id:      str   = Field(..., examples=["node-1"])
    current_load: float = Field(default=0.0, ge=0.0, le=1.0, examples=[0.35])


class JobSubmissionRequest(BaseModel):
    job_name:   str = Field(..., examples=["mnist-run-1"])
    dataset:    str = Field(default="MNIST", examples=["MNIST"])
    epochs:     int = Field(default=3, ge=1, le=500)
    batch_size: int = Field(default=64, ge=1, le=4096)


class JobCompletionRequest(BaseModel):
    job_id:          int           = Field(...)
    node_id:         str           = Field(...)
    success:         bool          = Field(...)
    checkpoint_path: Optional[str] = None
    error:           Optional[str] = None


class NodeResponse(BaseModel):
    id:                str
    host:              str
    port:              int
    gpu_memory_mb:     float
    current_load:      float
    reliability_score: float
    total_jobs:        int
    successful_jobs:   int
    status:            str
    last_heartbeat:    str
    api_key_hint:      Optional[str] = None


class JobResponse(BaseModel):
    id:               int
    job_name:         str
    dataset:          str
    epochs:           int
    status:           str
    assigned_node:    Optional[str]
    submitted_at:     Optional[str]
    started_at:       Optional[str]
    completed_at:     Optional[str]
    recovery_time_s:  Optional[float]
    checkpoint_path:  Optional[str]
    error_message:    Optional[str]


class PaginatedJobResponse(BaseModel):
    total:    int
    page:     int
    page_size: int
    items:    List[JobResponse]


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoints (no auth required)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="Liveness + readiness check")
def health(db: Session = Depends(get_db)):
    """Returns service status and basic DB connectivity."""
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    payload = {
        "status": "ok" if db_ok else "degraded",
        "db": "ok" if db_ok else "error",
        "scheduler": _settings.scheduler_type,
        "version": "2.0.0",
    }
    code = status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=payload, status_code=code)


@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "service": "GPU Orchestrator", "version": "2.0.0",
            "docs": "/docs", "health": "/health"}


# ─────────────────────────────────────────────────────────────────────────────
# v1 API Router
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import APIRouter

v1 = APIRouter(prefix="/api/v1")


# ── Node management ─────────────────────────────────────────────────────────

@v1.post(
    "/register_node",
    summary="Register a GPU agent node",
    tags=["Nodes"],
    dependencies=[Depends(require_api_key)],
)
def register_node(
    req: NodeRegistrationRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    node = NodeManager.register_node(
        db, req.node_id, req.host, req.port, req.gpu_memory_mb,
        api_key_hint=api_key[:8],
    )
    return {
        "message":     f"Node '{node.id}' registered successfully.",
        "node_id":     node.id,
        "status":      node.status,
        "reliability": node.reliability_score,
    }


@v1.post(
    "/heartbeat",
    summary="Receive heartbeat from a GPU agent",
    tags=["Nodes"],
    dependencies=[Depends(require_api_key)],
)
def heartbeat(req: HeartbeatRequest, db: Session = Depends(get_db)):
    node = NodeManager.update_heartbeat(db, req.node_id, req.current_load)
    if not node:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{req.node_id}' not found. Register first.",
        )
    return {"message": "Heartbeat received.", "node_id": node.id}


@v1.get(
    "/nodes",
    summary="List all registered nodes",
    tags=["Nodes"],
    response_model=List[NodeResponse],
)
def list_nodes(db: Session = Depends(get_db)):
    nodes = NodeManager.get_all_nodes(db)
    return [
        NodeResponse(
            id=n.id,
            host=n.host,
            port=n.port,
            gpu_memory_mb=round(n.gpu_memory_mb, 1),
            current_load=round(n.current_load, 3),
            reliability_score=round(n.reliability_score, 4),
            total_jobs=n.total_jobs,
            successful_jobs=n.successful_jobs,
            status=n.status,
            last_heartbeat=n.last_heartbeat.isoformat(),
            api_key_hint=n.api_key_hint,
        )
        for n in nodes
    ]


@v1.delete(
    "/nodes/{node_id}",
    summary="Deregister a GPU agent node",
    tags=["Nodes"],
    dependencies=[Depends(require_api_key)],
)
def deregister_node(node_id: str, db: Session = Depends(get_db)):
    from database.models import NodeStatus
    node = NodeManager.get_node(db, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    node.status = NodeStatus.OFFLINE
    db.commit()
    logger.info("Node '%s' manually deregistered (OFFLINE).", node_id)
    return {"message": f"Node '{node_id}' marked OFFLINE.", "node_id": node_id}


# ── Job management ──────────────────────────────────────────────────────────

@v1.post(
    "/submit_job",
    summary="Submit a training job for scheduling",
    tags=["Jobs"],
    dependencies=[Depends(require_api_key)],
)
def submit_job(req: JobSubmissionRequest, db: Session = Depends(get_db)):
    job = _job_manager.submit_job(
        db,
        job_name=req.job_name,
        dataset=req.dataset,
        epochs=req.epochs,
        batch_size=req.batch_size,
    )
    return {
        "message":       "Job queued and scheduled.",
        "job_id":        job.id,
        "assigned_node": job.assigned_node,
        "status":        job.status,
    }


@v1.post(
    "/job_complete",
    summary="Agent reports job completion or failure",
    tags=["Jobs"],
    dependencies=[Depends(require_api_key)],
)
def job_complete(req: JobCompletionRequest, db: Session = Depends(get_db)):
    _job_manager.handle_completion(
        db,
        job_id=req.job_id,
        node_id=req.node_id,
        success=req.success,
        checkpoint_path=req.checkpoint_path,
        error=req.error,
    )
    return {"message": "Job outcome recorded."}


@v1.get(
    "/jobs",
    summary="List all training jobs (paginated)",
    tags=["Jobs"],
    response_model=PaginatedJobResponse,
)
def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    q = db.query(TrainingJob)
    if status_filter:
        q = q.filter(TrainingJob.status == status_filter)
    total = q.count()
    jobs = (
        q.order_by(TrainingJob.submitted_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    def _to_resp(j: TrainingJob) -> JobResponse:
        return JobResponse(
            id=j.id,
            job_name=j.job_name,
            dataset=j.dataset,
            epochs=j.epochs,
            status=j.status,
            assigned_node=j.assigned_node,
            submitted_at=j.submitted_at.isoformat() if j.submitted_at else None,
            started_at=j.started_at.isoformat() if j.started_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            recovery_time_s=j.recovery_time_s,
            checkpoint_path=j.checkpoint_path,
            error_message=j.error_message,
        )

    return PaginatedJobResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[_to_resp(j) for j in jobs],
    )


@v1.get(
    "/jobs/{job_id}",
    summary="Get single job detail",
    tags=["Jobs"],
    response_model=JobResponse,
)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobResponse(
        id=job.id,
        job_name=job.job_name,
        dataset=job.dataset,
        epochs=job.epochs,
        status=job.status,
        assigned_node=job.assigned_node,
        submitted_at=job.submitted_at.isoformat() if job.submitted_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        recovery_time_s=job.recovery_time_s,
        checkpoint_path=job.checkpoint_path,
        error_message=job.error_message,
    )


# ── Metrics ─────────────────────────────────────────────────────────────────

@v1.get("/metrics/summary", summary="Aggregated metrics snapshot", tags=["Metrics"])
def metrics_summary():
    return MetricsCollector.get().summary()


@v1.get("/metrics/nodes", summary="Per-node reliability overview", tags=["Metrics"])
def metrics_nodes(db: Session = Depends(get_db)):
    nodes = NodeManager.get_all_nodes(db)
    result = []
    for n in nodes:
        history = (
            db.query(ReliabilityHistory)
            .filter(ReliabilityHistory.node_id == n.id)
            .order_by(ReliabilityHistory.recorded_at.desc())
            .limit(10)
            .all()
        )
        result.append({
            "node_id":         n.id,
            "status":          n.status,
            "reliability":     round(n.reliability_score, 4),
            "total_jobs":      n.total_jobs,
            "successful_jobs": n.successful_jobs,
            "recent_history":  [
                {"score": h.reliability_score, "at": h.recorded_at.isoformat()}
                for h in history
            ],
        })
    return result


# ── Register the v1 router ──────────────────────────────────────────────────
app.include_router(v1)

# ── Legacy path aliases (backward compatibility) ───────────────────────────
# Keep the old paths working for existing scripts while we fully migrate
from fastapi import APIRouter as _AR
_legacy = _AR(tags=["Legacy (deprecated – use /api/v1/*)"])

@_legacy.get("/nodes", include_in_schema=False)
def _legacy_nodes(db: Session = Depends(get_db)):
    return list_nodes(db)

@_legacy.get("/jobs", include_in_schema=False)
def _legacy_jobs(
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    return list_jobs(status_filter=status, page=1, page_size=200, db=db)

@_legacy.post("/register_node", include_in_schema=False, dependencies=[Depends(require_api_key)])
def _legacy_register(req: NodeRegistrationRequest, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    return register_node(req, db, api_key)

@_legacy.post("/heartbeat", include_in_schema=False, dependencies=[Depends(require_api_key)])
def _legacy_heartbeat(req: HeartbeatRequest, db: Session = Depends(get_db)):
    return heartbeat(req, db)

@_legacy.post("/submit_job", include_in_schema=False, dependencies=[Depends(require_api_key)])
def _legacy_submit(req: JobSubmissionRequest, db: Session = Depends(get_db)):
    return submit_job(req, db)

@_legacy.post("/job_complete", include_in_schema=False, dependencies=[Depends(require_api_key)])
def _legacy_complete(req: JobCompletionRequest, db: Session = Depends(get_db)):
    return job_complete(req, db)

@_legacy.get("/metrics/summary", include_in_schema=False)
def _legacy_metrics():
    return metrics_summary()

@_legacy.get("/metrics/nodes", include_in_schema=False)
def _legacy_metrics_nodes(db: Session = Depends(get_db)):
    return metrics_nodes(db)

app.include_router(_legacy)
