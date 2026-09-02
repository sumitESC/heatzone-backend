import sys
import os
import uvicorn

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.main import app

if __name__ == "__main__":
    print("==================================================")
    print("      HEATZONE REST API FORECASTING SERVER        ")
    print("==================================================")
    print("Server running at:  http://localhost:8000")
    print("Swagger Docs at:    http://localhost:8000/docs")
    print("ReDoc Docs at:      http://localhost:8000/redoc")
    print("OpenAPI Schema:     http://localhost:8000/openapi.json")
    print("==================================================")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
