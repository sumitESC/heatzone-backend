from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import config

from api.endpoints import forecast, history, context, sat_model, weather
import threading
import sys
import os

# Add root directory to sys path so we can import top-level scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import update_latest_weather
import generate_all_heatscores

app = FastAPI(
    title=config.PROJECT_NAME,
    description="AI Weather & Heatwave Forecasting REST API Server for Uttar Pradesh, India.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin API consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(forecast.router, prefix=f"{config.API_PREFIX}/forecast", tags=["Forecast"])
app.include_router(history.router, prefix=f"{config.API_PREFIX}/history", tags=["Historical Data"])
app.include_router(context.router, prefix=f"{config.API_PREFIX}/context", tags=["Context Data"])
app.include_router(sat_model.router, prefix=f"{config.API_PREFIX}/sat_model", tags=["Satellite Telemetry & Corridors"])
app.include_router(weather.router, prefix=f"{config.API_PREFIX}/weather", tags=["Unified Weather"])

@app.on_event("startup")
async def startup_event():
    print("=======================================")
    print("STARTING INITIALIZATION BACKGROUND JOB")
    print("=======================================")
    def run_init():
        try:
            print("1. Updating weather data...")
            update_latest_weather.main()
            print("2. Generating ensemble 16-day forecasts...")
            generate_all_heatscores.generate_all()
            print("Init jobs complete!")
        except Exception as e:
            print(f"Init job failed: {e}")
            
    thread = threading.Thread(target=run_init)
    thread.daemon = True
    thread.start()

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": config.PROJECT_NAME,
        "version": "1.0.0",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "endpoints": {
            "forecast": f"{config.API_PREFIX}/forecast/{{city}}",
            "history": f"{config.API_PREFIX}/history/{{city}}?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD",
            "context": f"{config.API_PREFIX}/context/{{city}}",
            "satellite_model": f"{config.API_PREFIX}/sat_model/{{city}}",
            "unified_weather": f"{config.API_PREFIX}/weather/{{city}}"
        }
    }
