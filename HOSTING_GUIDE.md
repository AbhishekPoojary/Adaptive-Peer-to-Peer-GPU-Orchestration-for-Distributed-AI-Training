# Hosting the GPU Orchestration Platform

## What Was Fixed First

The frontend previously used `http://127.0.0.1:8000` as the default API URL.  
When hosted on a server, visitors' browsers would try to connect to **their own** localhost instead of your server.  
✅ **Fixed**: the frontend now defaults to `window.location.origin` — it automatically uses the server it was loaded from.

---

## How It Works When Hosted

The FastAPI orchestrator **serves everything**:

```
https://your-domain.com/        → root JSON
https://your-domain.com/ui      → the dashboard (index.html)
https://your-domain.com/docs    → Swagger API docs
https://your-domain.com/api/v1/ → API endpoints
```

No separate frontend server, CDN, or build step is needed.

---

## Option A – VPS / Cloud VM (recommended)

Any server with a public IP works: **Oracle Free Tier**, DigitalOcean Droplet ($6/mo), AWS EC2, etc.

### 1. Provision the server
- Ubuntu 22.04, 1 vCPU, 1 GB RAM minimum
- Open ports **80**, **443**, and **8000** in the firewall/security group

### 2. Install dependencies
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git nginx certbot python3-certbot-nginx
git clone <your-repo-url> gpu_orchestration
cd gpu_orchestration
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create a `.env` file
```bash
cp .env.example .env
nano .env
```
Set these at minimum:
```env
ORCHESTRATOR_API_KEY=choose-a-strong-key-here
ALLOWED_ORIGINS=https://your-domain.com
DATABASE_URL=sqlite:///./gpu_orchestration.db
LOG_FORMAT=json
```

### 4. Run as a systemd service (auto-restarts on crashes/reboots)
```bash
sudo nano /etc/systemd/system/gpu-orch.service
```
```ini
[Unit]
Description=GPU Orchestration Platform
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/gpu_orchestration
EnvironmentFile=/home/ubuntu/gpu_orchestration/.env
ExecStart=/home/ubuntu/gpu_orchestration/.venv/bin/uvicorn orchestrator.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-orch
sudo systemctl status gpu-orch   # should show "active (running)"
```

### 5. Set up Nginx as reverse proxy (HTTPS)
```bash
sudo nano /etc/nginx/sites-available/gpu-orch
```
```nginx
server {
    server_name your-domain.com;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```
```bash
sudo ln -s /etc/nginx/sites-available/gpu-orch /etc/nginx/sites-enabled/
sudo nginx -t && sudo nginx -s reload

# Free HTTPS via Let's Encrypt:
sudo certbot --nginx -d your-domain.com
```

Now visit **https://your-domain.com/ui** — the dashboard loads and calls the API automatically.

---

## Option B – Docker (simplest)

```bash
cp .env.example .env   # edit ORCHESTRATOR_API_KEY
docker-compose up -d --build
```

The orchestrator binds to port **8000**. Put Nginx in front for HTTPS the same way.

---

## Option C – No domain (IP only, for testing)

```bash
# On the server (not docker)
$env:ORCHESTRATOR_API_KEY="my-key"
uvicorn orchestrator.main:app --host 0.0.0.0 --port 8000
```

Visit: `http://<server-ip>:8000/ui`

> **Security warning**: running without HTTPS exposes the API key in plain text. Use this only for testing.

---

## Connecting Remote GPU Agents to a Hosted Orchestrator

Once deployed, anyone can contribute their GPU:

1. Get the API key from the admin
2. Set env vars on their machine:
   ```
   ORCHESTRATOR_URL=https://your-domain.com
   ORCHESTRATOR_API_KEY=the-key-from-admin
   NODE_ID=their-unique-name
   AGENT_HOST=their-public-ip
   ```
3. Start the agent: `uvicorn gpu_agent.agent:app --host 0.0.0.0 --port 8001`
4. Their node appears in the dashboard automatically

---

## Your Current Local IP (for LAN testing)

Your machine's IP on the local network is **`10.116.198.147`**.  
People on the same Wi-Fi network can access your dashboard at:

```
http://10.116.198.147:8000/ui
```

Set `AGENT_HOST=10.116.198.147` when registering agents from other machines on the same network.
