# Adaptive P2P GPU Orchestration for Distributed AI Training
### Academic MVP – Readme & Setup Guide

---

## Overview

This project implements a **distributed GPU training orchestration system** designed as an academic MVP demonstrating:

| Feature | Description |
|---|---|
| Distributed job allocation | Training jobs submitted to a central orchestrator and dispatched to peer GPU agents |
| Adaptive scheduling | Three algorithms (Round-Robin, Least-Loaded, Reliability-Aware Adaptive) |
| Fault tolerance | Heartbeat monitoring, automatic failure detection, job recovery from checkpoints |
| Experimental measurability | Metrics collection (JSONL), per-node reliability history, scheduling logs |

---

## Project Structure

```
gpu_orchestration/
├── orchestrator/               # Central controller (FastAPI)
│   ├── main.py                 # REST API + lifespan
│   ├── node_manager.py         # Node registration, heartbeat, reliability
│   ├── job_manager.py          # Job queue, dispatch, fault recovery
│   └── heartbeat_monitor.py   # Background failure detector
├── gpu_agent/                  # Peer node service (FastAPI)
│   └── agent.py                # Registration, heartbeat sender, job runner
├── scheduler/                  # Custom scheduling algorithms
│   ├── base.py
│   ├── round_robin.py
│   ├── least_loaded.py
│   ├── adaptive.py             # Reliability-Aware (configurable α, β, γ)
│   └── scheduler_factory.py
├── training/
│   └── train_mnist.py          # PyTorch MNIST trainer with checkpoint support
├── database/
│   ├── models.py               # SQLAlchemy ORM models
│   └── db.py                   # SQLite engine + session management
├── utils/
│   ├── logger.py
│   ├── config_loader.py
│   └── metrics.py
├── config.yaml                 # All tunable parameters
├── requirements.txt
├── simulate_failure.py         # Node failure simulation script
└── logs/                       # Auto-created: orchestration.log, metrics.jsonl
```

---

## Prerequisites

- Python 3.10 or 3.11
- Windows / Linux / macOS
- No GPU required (CPU training works; GPU if available is auto-detected)

---

## Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **GPU Users**: Replace the `torch` / `torchvision` lines in `requirements.txt` with your CUDA wheel:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Running a 3-Node Local Setup

Open **4 separate terminal windows** (all from the `gpu_orchestration/` directory).

### Terminal 1 – Orchestrator

```bash
cd gpu_orchestration
uvicorn orchestrator.main:app --host 0.0.0.0 --port 8000 --reload
```

The orchestrator is now available at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs`

---

### Terminal 2 – GPU Agent Node-1

```bash
cd gpu_orchestration
set NODE_ID=node-1
set AGENT_PORT=8001
set ORCHESTRATOR_URL=http://127.0.0.1:8000
uvicorn gpu_agent.agent:app --host 0.0.0.0 --port 8001
```

Linux/macOS equivalent:
```bash
NODE_ID=node-1 AGENT_PORT=8001 ORCHESTRATOR_URL=http://127.0.0.1:8000 \
  uvicorn gpu_agent.agent:app --host 0.0.0.0 --port 8001
```

---

### Terminal 3 – GPU Agent Node-2

```bash
set NODE_ID=node-2
set AGENT_PORT=8002
set ORCHESTRATOR_URL=http://127.0.0.1:8000
uvicorn gpu_agent.agent:app --host 0.0.0.0 --port 8002
```

---

### Terminal 4 – GPU Agent Node-3 (optional)

```bash
set NODE_ID=node-3
set AGENT_PORT=8003
set ORCHESTRATOR_URL=http://127.0.0.1:8000
uvicorn gpu_agent.agent:app --host 0.0.0.0 --port 8003
```

---

## Submitting Training Jobs

### Using curl

```bash
# Submit job 1
curl -X POST http://localhost:8000/submit_job \
  -H "Content-Type: application/json" \
  -d '{"job_name": "mnist-exp-1", "dataset": "MNIST", "epochs": 3, "batch_size": 64}'

# Submit job 2
curl -X POST http://localhost:8000/submit_job \
  -H "Content-Type: application/json" \
  -d '{"job_name": "mnist-exp-2", "epochs": 5}'
```

### Using Python

```python
import requests
resp = requests.post("http://localhost:8000/submit_job", json={
    "job_name":   "mnist-run-1",
    "dataset":    "MNIST",
    "epochs":     3,
    "batch_size": 64,
})
print(resp.json())
```

---

## Monitoring the System

| Endpoint | Purpose |
|---|---|
| `GET /nodes` | All registered nodes with status, load, reliability |
| `GET /jobs` | All training jobs and their states |
| `GET /jobs?status=running` | Filter by status |
| `GET /metrics/summary` | Aggregated metrics (avg times, recovery counts) |
| `GET /metrics/nodes` | Per-node reliability timeline |
| `GET /docs` | Interactive Swagger UI |

```bash
# Check nodes
curl http://localhost:8000/nodes

# Check jobs
curl http://localhost:8000/jobs

# Check metrics
curl http://localhost:8000/metrics/summary
```

---

## Changing the Scheduler

Edit `config.yaml`:

```yaml
orchestrator:
  scheduler_type: "round_robin"   # or "least_loaded" or "adaptive"
```

Or for the adaptive algorithm, tune the weights:

```yaml
scheduler:
  adaptive:
    alpha: 0.5   # load penalty
    beta:  0.4   # reliability reward
    gamma: 0.1   # heartbeat latency penalty
```

Restart the orchestrator after any config change.

---

## Simulating Node Failure

See the dedicated section below.

---

## Simulating Node Failure (Instructions)

### Method 1: Automatic Script

With the full 3-node setup running and at least one job in progress:

```bash
cd gpu_orchestration
python simulate_failure.py --node-id node-1 --orchestrator-url http://127.0.0.1:8000
```

This script:
1. Submits two jobs to ensure one lands on `node-1`.
2. Forcibly stops the heartbeat from `node-1` by killing that process (via OS kill).
3. Waits 20 seconds (past the 15-second timeout).
4. Polls the orchestrator and reports whether job recovery succeeded.

### Method 2: Manual Kill

1. Submit a job via `/submit_job`.
2. Confirm it is running on `node-1` via `GET /jobs`.
3. **Kill the Terminal 2 process** (Ctrl+C or close the window).
4. Within 15 seconds (heartbeat timeout), the orchestrator marks `node-1` FAILED.
5. The running job is automatically restarted on the next-best node, resuming from the last saved checkpoint.
6. Confirm recovery: `curl http://localhost:8000/jobs`

### Method 3: Network Simulation (advanced)

For a more realistic test, use a firewall rule to drop packets from the agent port:

```powershell
# Windows – block node-1 outbound on port 8001
netsh advfirewall firewall add rule name="BlockNode1" dir=out protocol=tcp localport=8001 action=block

# Remove the block to "recover" the node
netsh advfirewall firewall delete rule name="BlockNode1"
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| SQLite (default) | Zero-config for academic demo; swap to PostgreSQL via `config.yaml` |
| Subprocess for training | Keeps agent stateless; clean process isolation |
| In-process heartbeat monitor | Avoids Celery/Redis dependency; sufficient for 2-3 nodes |
| Checkpoint every epoch | Minimises recovery data loss; configurable via training script |
| Reliability formula | `R = successful / total` – simple, interpretable, academically defensible |

---

## Adaptive Scheduling Formula

```
Score_i = α × Load_i  −  β × Reliability_i  +  γ × Latency_i
```

- `Load_i` — current CPU/GPU utilisation (0–1)
- `Reliability_i` — `successful_jobs / total_jobs` (0–1); subtracted so reliable nodes score lower
- `Latency_i` — seconds since last heartbeat / 60 (normalised)
- Node with **minimum score** is selected

Default: `α=0.5, β=0.4, γ=0.1` (configurable in `config.yaml`).

---

## Logs and Metrics

| File | Content |
|---|---|
| `logs/orchestration.log` | Rotating text log (all components) |
| `logs/metrics.jsonl` | Newline-delimited JSON; one record per job event |

Parse metrics for analysis:
```python
import json, pathlib
records = [json.loads(l) for l in pathlib.Path("logs/metrics.jsonl").read_text().splitlines()]
```

---

## Academic Evaluation Notes

- All three schedulers can be compared by changing `scheduler_type` and re-running experiments.
- Reliability history is stored per-node in the database for plotting.
- Recovery time is logged both in the DB (`recovery_time_s` column) and in `metrics.jsonl`.
- The adaptive scheduler logs per-candidate scores at DEBUG level for detailed analysis.
