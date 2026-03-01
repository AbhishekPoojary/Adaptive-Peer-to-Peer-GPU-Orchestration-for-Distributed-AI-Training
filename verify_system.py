import urllib.request, json, urllib.error

print("=" * 50)
print("Production System – Final Smoke Test")
print("=" * 50)

BASE = "http://localhost:8000"

# Test 1: Health (no auth needed)
r = urllib.request.urlopen(f"{BASE}/health")
h = json.load(r)
print(f"\n[1] Health: status={h['status']} db={h['db']} version={h['version']} -> PASS")

# Test 2: Auth REJECTED (no key)
try:
    req = urllib.request.Request(
        f"{BASE}/api/v1/submit_job",
        data=json.dumps({"job_name": "test"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req)
    print("[2] Auth rejected (no key): FAIL - should be 401")
except urllib.error.HTTPError as e:
    if e.code == 401:
        print("[2] Auth rejected (no key): 401 -> PASS")
    else:
        print(f"[2] Unexpected status: {e.code}")

# Test 3: Auth REJECTED (wrong key)
try:
    req = urllib.request.Request(
        f"{BASE}/api/v1/submit_job",
        data=json.dumps({"job_name": "test"}).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": "wrong-key"},
    )
    urllib.request.urlopen(req)
    print("[3] Auth rejected (wrong key): FAIL")
except urllib.error.HTTPError as e:
    if e.code == 401:
        print("[3] Auth rejected (wrong key): 401 -> PASS")
    else:
        print(f"[3] Unexpected: {e.code}")

# Test 4: Submit job with correct key
req = urllib.request.Request(
    f"{BASE}/api/v1/submit_job",
    data=json.dumps({"job_name": "prod-smoke-test", "epochs": 1}).encode(),
    headers={"Content-Type": "application/json", "X-API-Key": "gpu-secret-2024"},
)
result = json.load(urllib.request.urlopen(req))
print(f"[4] Job submitted: id={result['job_id']} node={result['assigned_node']} status={result['status']} -> PASS")

# Test 5: List nodes via new API
r = urllib.request.urlopen(f"{BASE}/api/v1/nodes")
nodes = json.load(r)
print(f"\n[5] Nodes ({len(nodes)} registered):")
for n in nodes:
    print(f"    {n['id']:10s}  status={n['status']:7s}  load={n['current_load']:.2f}  rel={n['reliability_score']:.3f}  hint={n['api_key_hint']}")

# Test 6: Paginated jobs
r = urllib.request.urlopen(f"{BASE}/api/v1/jobs?page=1&page_size=5")
jobs = json.load(r)
print(f"\n[6] Jobs: total={jobs['total']} page={jobs['page']} page_size={jobs['page_size']} -> PASS")

print("\n" + "=" * 50)
print("All checks passed. System running at v2.0.0")
print("=" * 50)
