"""
gpu_agent/agent.py
-------------------
GPU Agent Node – production-grade FastAPI micro-service.

Responsibilities:
  1. Register itself with the orchestrator on startup.
  2. Send heartbeat every N seconds (background thread).
  3. Expose POST /run_job  → spawns training subprocess.
  4. On SIGTERM: alert orchestrator (OFFLINE) before exiting.

Environment variables (all optional with defaults):
  NODE_ID              e.g. "node-1"
  AGENT_HOST           Advertised host; auto-detected from hostname if not set
  AGENT_PORT           Port this agent listens on (default: 8001)
  GPU_MEMORY_MB        Override reported GPU VRAM (default: auto-detected)
  ORCHESTRATOR_URL     (default: http://127.0.0.1:8000)
  ORCHESTRATOR_API_KEY API key matching the orchestrator's key
  HEARTBEAT_INTERVAL_S (default: 5)
"""

import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import psutil
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.logger import setup_logging
from utils.settings import get_settings

# ── Bootstrap settings ────────────────────────────────────────────────────────
_settings = get_settings()
setup_logging(_settings.log_level, _settings.log_format)
logger = logging.getLogger("gpu_agent")

# ── Config (env overrides settings) ──────────────────────────────────────────
NODE_ID           = os.getenv("NODE_ID",           _settings.node_id)
AGENT_PORT        = int(os.getenv("AGENT_PORT",    str(_settings.agent_port)))
ORCHESTRATOR_URL  = os.getenv("ORCHESTRATOR_URL",  _settings.orchestrator_url)
API_KEY           = os.getenv("ORCHESTRATOR_API_KEY", _settings.orchestrator_api_key)
HEARTBEAT_INTERVAL= int(os.getenv("HEARTBEAT_INTERVAL_S", str(_settings.heartbeat_interval_s)))

# Auto-detect AGENT_HOST: use env var → settings → hostname resolution
def _detect_agent_host() -> str:
    explicit = os.getenv("AGENT_HOST") or _settings.agent_host
    if explicit:
        return explicit
    try:
        # Connect to an external address to discover local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

AGENT_HOST = _detect_agent_host()

# ── GPU detection ─────────────────────────────────────────────────────────────
def _detect_gpu_memory() -> float:
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 2)
        except Exception:
            pass
    return float(os.getenv("GPU_MEMORY_MB", "0"))

GPU_MEMORY_MB = _detect_gpu_memory()

# ── Auth headers ──────────────────────────────────────────────────────────────
def _auth_headers() -> dict:
    return {"X-API-Key": API_KEY}

# ── Current job tracking ──────────────────────────────────────────────────────
_current_job_id: Optional[int] = None
_job_lock = threading.Lock()
_shutdown_event = threading.Event()


# ── Registration & heartbeat ──────────────────────────────────────────────────

def _register() -> bool:
    payload = {
        "node_id":       NODE_ID,
        "host":          AGENT_HOST,
        "port":          AGENT_PORT,
        "gpu_memory_mb": GPU_MEMORY_MB,
    }
    for attempt in range(1, 6):
        try:
            # Try /api/v1/ first, fallback to legacy
            for path in ["/api/v1/register_node", "/register_node"]:
                try:
                    r = httpx.post(
                        f"{ORCHESTRATOR_URL}{path}",
                        json=payload,
                        headers=_auth_headers(),
                        timeout=10,
                    )
                    r.raise_for_status()
                    logger.info(
                        "Registered with orchestrator as '%s' @ %s:%d (host_ip=%s gpu=%.0fMB)",
                        NODE_ID, AGENT_HOST, AGENT_PORT, AGENT_HOST, GPU_MEMORY_MB,
                    )
                    return True
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        continue  # try next path
                    raise
        except Exception as exc:
            logger.warning("Registration attempt %d failed: %s", attempt, exc)
            time.sleep(3)
    logger.error("Could not register with orchestrator after 5 attempts.")
    return False


def _deregister() -> None:
    """Notify the orchestrator that this node is going offline gracefully."""
    for path in [f"/api/v1/nodes/{NODE_ID}", None]:
        if path is None:
            break
        try:
            httpx.delete(
                f"{ORCHESTRATOR_URL}{path}",
                headers=_auth_headers(),
                timeout=5,
            )
            logger.info("Gracefully deregistered node '%s'.", NODE_ID)
            return
        except Exception as exc:
            logger.warning("Deregister failed: %s", exc)


def _get_current_load() -> float:
    try:
        cpu = psutil.cpu_percent(interval=0.1) / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        return round((cpu + mem) / 2.0, 3)
    except Exception:
        return 0.0


def _heartbeat_loop():
    """Daemon thread: sends heartbeat to orchestrator every N seconds."""
    while not _shutdown_event.is_set():
        _shutdown_event.wait(HEARTBEAT_INTERVAL)
        if _shutdown_event.is_set():
            break
        try:
            load = _get_current_load()
            for path in ["/api/v1/heartbeat", "/heartbeat"]:
                try:
                    httpx.post(
                        f"{ORCHESTRATOR_URL}{path}",
                        json={"node_id": NODE_ID, "current_load": load},
                        headers=_auth_headers(),
                        timeout=5,
                    )
                    logger.debug("Heartbeat sent (load=%.3f)", load)
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        continue
                    raise
        except Exception as exc:
            logger.warning("Heartbeat failed: %s", exc)


# ── Signal handler for graceful SIGTERM ──────────────────────────────────────

def _handle_sigterm(*_):
    logger.info("SIGTERM received – shutting down agent '%s'.", NODE_ID)
    _shutdown_event.set()
    _deregister()
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _register()
    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()
    logger.info("GPU Agent '%s' running on %s:%d", NODE_ID, AGENT_HOST, AGENT_PORT)
    yield
    _shutdown_event.set()
    _deregister()
    logger.info("GPU Agent '%s' shut down.", NODE_ID)


app = FastAPI(
    title=f"GPU Agent – {NODE_ID}",
    description="Peer node agent for GPU Orchestration (production build)",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RunJobRequest(BaseModel):
    job_id:          int           = Field(...)
    job_name:        str           = Field(...)
    dataset:         str           = Field(default="MNIST")
    epochs:          int           = Field(default=3)
    batch_size:      int           = Field(default=64)
    checkpoint_path: Optional[str] = None
    resume_epoch:    int           = Field(default=0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Agent health check")
def root():
    return {
        "status":        "ok",
        "node_id":       NODE_ID,
        "agent_host":    AGENT_HOST,
        "agent_port":    AGENT_PORT,
        "gpu_memory_mb": GPU_MEMORY_MB,
        "current_load":  _get_current_load(),
        "version":       "2.0.0",
    }


@app.get("/health", summary="Liveness probe")
def health():
    return {"status": "ok", "node_id": NODE_ID}


@app.post("/run_job", summary="Accept and execute a training job")
def run_job(req: RunJobRequest):
    global _current_job_id
    with _job_lock:
        if _current_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Node '{NODE_ID}' is already running job {_current_job_id}.",
            )
        _current_job_id = req.job_id

    logger.info(
        "Received job %d '%s' | epochs=%d | resume=%d",
        req.job_id, req.job_name, req.epochs, req.resume_epoch,
    )

    train_script = Path(__file__).parent.parent / "training" / "train_mnist.py"
    cmd = [
        sys.executable, str(train_script),
        "--job-id",           str(req.job_id),
        "--job-name",         req.job_name,
        "--epochs",           str(req.epochs),
        "--batch-size",       str(req.batch_size),
        "--resume-epoch",     str(req.resume_epoch),
        "--node-id",          NODE_ID,
        "--orchestrator-url", ORCHESTRATOR_URL,
        "--api-key",          API_KEY,
    ]
    if req.checkpoint_path:
        cmd += ["--checkpoint-path", req.checkpoint_path]

    def _run():
        global _current_job_id
        logger.info("Launching training subprocess for job %d", req.job_id)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            logger.info("[train] %s", line.decode(errors="replace").rstrip())
        proc.wait()
        logger.info("Training process exited with code %d", proc.returncode)
        with _job_lock:
            _current_job_id = None

    threading.Thread(target=_run, daemon=True).start()
    return {
        "message": f"Training job {req.job_id} started on node '{NODE_ID}'.",
        "job_id":  req.job_id,
        "node_id": NODE_ID,
    }


@app.get("/status", summary="Current agent status")
def agent_status():
    return {
        "node_id":       NODE_ID,
        "agent_host":    AGENT_HOST,
        "agent_port":    AGENT_PORT,
        "current_job":   _current_job_id,
        "current_load":  _get_current_load(),
        "gpu_memory_mb": GPU_MEMORY_MB,
    }
