import sys
import os
import json
import time

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from api.main import app

routes_to_test = [
    {"name": "Root Index", "path": "/"},
    {"name": "Swagger UI Docs", "path": "/docs"},
    {"name": "ReDoc Documentation", "path": "/redoc"},
    {"name": "OpenAPI JSON Schema", "path": "/openapi.json"},
    {"name": "Forecast API (Lucknow)", "path": "/api/v1/forecast/Lucknow"},
    {"name": "Forecast API (Agra)", "path": "/api/v1/forecast/Agra"},
    {"name": "History API (Lucknow)", "path": "/api/v1/history/Lucknow?start_date=2026-08-01&end_date=2026-08-15"},
    {"name": "India Context API", "path": "/api/v1/context/india?date=2026-08-15"},
    {"name": "Satellite Model API (Lucknow)", "path": "/api/v1/sat_model/Lucknow"},
    {"name": "Satellite Model Report (Lucknow)", "path": "/api/v1/sat_model/Lucknow/report"},
    {"name": "Unified Weather Current (Lucknow)", "path": "/api/v1/weather/Lucknow/current"},
    {"name": "Unified Weather Forecast (Lucknow)", "path": "/api/v1/weather/Lucknow/forecast"},
    {"name": "Unified Weather Previous (Lucknow)", "path": "/api/v1/weather/Lucknow/previous?date=2026-08-15"}
]

print("==================================================")
print("       TESTING ALL HEATZONE REST API ROUTES       ")
print("==================================================")

client = TestClient(app)
results = []

for item in routes_to_test:
    name = item["name"]
    path = item["path"]
    start_time = time.time()
    try:
        response = client.get(path)
        status = response.status_code
        content_type = response.headers.get('content-type', '')
        elapsed = round((time.time() - start_time) * 1000, 2)
        
        sample = ""
        if "json" in content_type:
            try:
                data = response.json()
                sample = json.dumps(data, indent=2)[:250] + "..."
            except Exception:
                sample = response.text[:150] + "..."
        else:
            sample = f"<{content_type}> ({len(response.content)} bytes)"
            
        if status == 200:
            print(f"[PASS] {status} | {name:<35} | {elapsed:>7.2f} ms | {path}")
            results.append({"name": name, "path": path, "status": status, "ms": elapsed, "sample": sample})
        else:
            print(f"[FAIL] {status} | {name:<35} | {elapsed:>7.2f} ms | {path} -> {response.text[:100]}")
            results.append({"name": name, "path": path, "status": status, "ms": elapsed, "sample": response.text})
    except Exception as e:
        elapsed = round((time.time() - start_time) * 1000, 2)
        print(f"[FAIL] ERR | {name:<35} | {elapsed:>7.2f} ms | {path} -> Error: {e}")
        results.append({"name": name, "path": path, "status": "ERROR", "ms": elapsed, "sample": str(e)})

print("==================================================")
print(f"Total Routes Tested: {len(routes_to_test)}")
passed_count = sum(1 for r in results if r['status'] == 200)
print(f"Passed: {passed_count}/{len(routes_to_test)}")
print("==================================================")

if passed_count == len(routes_to_test):
    print("ALL ROUTES WORKING PERFECTLY!")
else:
    sys.exit(1)


