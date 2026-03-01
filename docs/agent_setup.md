# How to Join the GPU Cluster as a Remote Node

This guide lets you share your GPU (or CPU) with the GPU Orchestration cluster in **5 minutes**.

---

## Prerequisites

- Python 3.10 or 3.11
- The project files (just the `gpu_agent/`, `utils/`, `training/` folders + `requirements.txt`)
- Network access to the orchestrator machine
- The **API key** from the orchestrator admin

---

## Step 1 – Get the files

```bash
# Option A: clone full repo
git clone <repo-url>
cd gpu_orchestration

# Option B: copy just what the agent needs
# gpu_agent/, utils/, training/, requirements.txt, config.yaml
```

## Step 2 – Install dependencies

```bash
pip install -r requirements.txt

# GPU users: replace torch with CUDA wheel first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Step 3 – Set environment variables

**Windows (PowerShell):**
```powershell
$env:NODE_ID             = "alice-rtx4090"      # unique name for your node
$env:AGENT_HOST          = "192.168.1.55"        # YOUR machine's IP (not 127.0.0.1)
$env:AGENT_PORT          = "8001"
$env:ORCHESTRATOR_URL    = "http://192.168.1.10:8000"   # orchestrator's IP
$env:ORCHESTRATOR_API_KEY= "the-api-key-from-admin"
```

**Linux / macOS:**
```bash
export NODE_ID=alice-rtx4090
export AGENT_HOST=$(hostname -I | awk '{print $1}')   # auto-detect your IP
export AGENT_PORT=8001
export ORCHESTRATOR_URL=http://192.168.1.10:8000
export ORCHESTRATOR_API_KEY=the-api-key-from-admin
```

> **Finding your IP**: run `ipconfig` (Windows) or `ip addr` / `hostname -I` (Linux).

## Step 4 – Start the agent

```bash
uvicorn gpu_agent.agent:app --host 0.0.0.0 --port 8001
```

You should see:
```
INFO | Registered with orchestrator as 'alice-rtx4090' @ 192.168.1.55:8001
INFO | GPU Agent 'alice-rtx4090' running on 192.168.1.55:8001
```

## Step 5 – Verify you're registered

Ask the orchestrator admin to check:
```
GET http://<orchestrator-ip>:8000/api/v1/nodes
```

Your node should appear with `"status": "online"`.

---

## Notes

- Your agent sends heartbeats every 5 seconds to stay registered.
- If your agent stops unexpectedly, it's marked `FAILED` after 15 seconds and any running job is recovered on another node.
- To shutdown gracefully (marks you `OFFLINE` instead of `FAILED`): press **Ctrl+C** or send SIGTERM.
- The agent only runs **training jobs** dispatched by the orchestrator. It does **not** give the orchestrator direct access to your machine.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401 Unauthorized` on registration | Check `ORCHESTRATOR_API_KEY` matches the orchestrator's key |
| Node shows `offline` after starting | The orchestrator can't reach `AGENT_HOST:AGENT_PORT` — check firewall / NAT |
| `Connection refused` to orchestrator | Check `ORCHESTRATOR_URL` and that port 8000 is open on the orchestrator machine |
