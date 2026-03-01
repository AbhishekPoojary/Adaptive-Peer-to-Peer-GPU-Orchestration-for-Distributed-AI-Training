"""
simulate_failure.py
--------------------
Academic demonstration script for simulating a node failure and verifying
automatic job recovery.

What it does:
  1. Submits two training jobs to the orchestrator.
  2. Waits a moment to ensure jobs are dispatched.
  3. Reports which node each job landed on.
  4. Instructs user to kill the relevant agent OR auto-stops the agent
     by sending a SIGTERM (Unix) / taskkill (Windows) if --auto-kill is set.
  5. Waits for the heartbeat timeout window.
  6. Polls orchestrator until recovery is detected or timeout.
  7. Prints a summary of the recovery event.

Usage:
    python simulate_failure.py \
        --node-id node-1 \
        --orchestrator-url http://127.0.0.1:8000 \
        [--auto-kill]       # WARNING: kills the uvicorn process for node-1

Requirements:
    pip install requests
"""

import argparse
import json
import sys
import time
import os

import requests

TIMEOUT_S = 60      # max seconds to wait for recovery


def submit_jobs(url: str, count: int = 2) -> list[dict]:
    jobs = []
    for i in range(1, count + 1):
        resp = requests.post(
            f"{url}/submit_job",
            json={
                "job_name":   f"sim-failure-job-{i}",
                "dataset":    "MNIST",
                "epochs":     3,
                "batch_size": 64,
            },
            timeout=10,
        )
        resp.raise_for_status()
        jobs.append(resp.json())
        print(f"  ✓ Submitted job {resp.json()['job_id']} → node {resp.json().get('assigned_node', '?')}")
    return jobs


def get_jobs(url: str) -> list[dict]:
    return requests.get(f"{url}/jobs", timeout=10).json()


def get_nodes(url: str) -> list[dict]:
    return requests.get(f"{url}/nodes", timeout=10).json()


def find_node_status(nodes: list[dict], node_id: str) -> str:
    for n in nodes:
        if n["id"] == node_id:
            return n["status"]
    return "unknown"


def wait_for_failure(url: str, node_id: str, timeout_s: int = TIMEOUT_S) -> bool:
    print(f"\n  Waiting up to {timeout_s}s for orchestrator to detect node '{node_id}' failure …")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        nodes = get_nodes(url)
        status = find_node_status(nodes, node_id)
        if status == "failed":
            print(f"  ✓ Node '{node_id}' marked FAILED (elapsed {time.time()-t0:.1f}s)")
            return True
        print(f"    … node status: {status}", end="\r")
        time.sleep(2)
    return False


def wait_for_recovery(url: str, job_ids: list[int], timeout_s: int = TIMEOUT_S) -> bool:
    print(f"\n  Waiting up to {timeout_s}s for job recovery …")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        jobs = get_jobs(url)
        relevant = [j for j in jobs if j["id"] in job_ids]
        statuses = {j["id"]: j["status"] for j in relevant}
        all_done = all(s in ("completed", "failed") for s in statuses.values())
        recovered = any(j.get("recovery_time_s") is not None for j in relevant)
        print(f"    … job statuses: {statuses}", end="\r")
        if all_done or recovered:
            print()
            return True
        time.sleep(3)
    return False


def print_summary(url: str, job_ids: list[int], node_id: str) -> None:
    print("\n" + "=" * 60)
    print("  SIMULATION SUMMARY")
    print("=" * 60)
    jobs = get_jobs(url)
    nodes = get_nodes(url)

    print("\n  Failed node:")
    for n in nodes:
        if n["id"] == node_id:
            print(f"    {n['id']}  status={n['status']}  reliability={n['reliability_score']:.3f}")

    print("\n  Affected jobs:")
    for j in jobs:
        if j["id"] in job_ids:
            print(
                f"    job_id={j['id']}  name={j['job_name']}"
                f"  status={j['status']}"
                f"  node={j['assigned_node']}"
                f"  recovery_time={j.get('recovery_time_s', 'n/a')}s"
                f"  checkpoint={j.get('checkpoint_path') or 'none'}"
            )

    metrics = requests.get(f"{url}/metrics/summary", timeout=10).json()
    print("\n  System metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="Simulate GPU node failure and recovery")
    p.add_argument("--node-id",          default="node-1")
    p.add_argument("--orchestrator-url", default="http://127.0.0.1:8000")
    p.add_argument("--auto-kill",        action="store_true",
                   help="Attempt to automatically kill the named node's agent process")
    args = p.parse_args()

    url     = args.orchestrator_url
    node_id = args.node_id

    print("\n" + "=" * 60)
    print("  Adaptive P2P GPU Orchestration – Failure Simulation")
    print("=" * 60)

    # 1. Verify orchestrator reachable
    try:
        requests.get(f"{url}/", timeout=5).raise_for_status()
        print(f"  ✓ Orchestrator reachable at {url}")
    except Exception as exc:
        print(f"  ✗ Cannot reach orchestrator at {url}: {exc}")
        sys.exit(1)

    # 2. Show current nodes
    nodes = get_nodes(url)
    print(f"\n  Registered nodes ({len(nodes)}):")
    for n in nodes:
        print(f"    {n['id']}  {n['host']}:{n['port']}  status={n['status']}  load={n['current_load']}")

    node_ids = [n["id"] for n in nodes]
    if node_id not in node_ids:
        print(f"\n  ✗ Node '{node_id}' is not registered. Registered: {node_ids}")
        sys.exit(1)

    # 3. Submit jobs
    print(f"\n  Submitting 2 training jobs …")
    submitted = submit_jobs(url, count=2)
    job_ids   = [j["job_id"] for j in submitted]
    time.sleep(2)    # allow dispatch

    # 4. If auto-kill is set, find and kill the agent process
    if args.auto_kill:
        print(f"\n  ⚠ Auto-kill requested for node '{node_id}'.")
        import subprocess
        # Find uvicorn processes listening on the agent's port
        node_port_map = {n["id"]: n["port"] for n in nodes}
        target_port   = node_port_map.get(node_id)
        if target_port:
            if os.name == "nt":
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True, text=True
                )
                for line in result.stdout.splitlines():
                    if f":{target_port}" in line and "LISTEN" in line:
                        parts = line.split()
                        pid   = parts[-1]
                        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                        print(f"  ✓ Killed PID {pid} (node '{node_id}' on port {target_port})")
                        break
            else:
                subprocess.run(["fuser", "-k", f"{target_port}/tcp"], capture_output=True)
                print(f"  ✓ Sent kill signal to port {target_port}")
    else:
        print(
            f"\n  ➜ ACTION REQUIRED: Kill the agent terminal for '{node_id}' now.\n"
            f"    Press Ctrl+C in the terminal running node '{node_id}'.\n"
            f"    The orchestrator will detect the failure within ~15 seconds.\n"
        )
        input("    Press ENTER here once you have killed the node agent …")

    # 5. Wait for failure detection
    detected = wait_for_failure(url, node_id, timeout_s=30)
    if not detected:
        print(f"  ✗ Orchestrator did not mark '{node_id}' as failed within 30s.")
        print("    Check heartbeat_timeout_s in config.yaml (default: 15s)")
        sys.exit(1)

    # 6. Wait for job recovery
    recovered = wait_for_recovery(url, job_ids, timeout_s=TIMEOUT_S)
    if not recovered:
        print("  ✗ Job recovery not detected within timeout.")

    # 7. Summary
    print_summary(url, job_ids, node_id)


if __name__ == "__main__":
    main()
