import os
import json
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query

from api.services import model_runner
from api.services.heatscore_service import predict_heatscores

router = APIRouter()

def get_forecast_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "all_cities_heatscore_forecast.json")
    if not os.path.exists(data_path):
        return []
    try:
        with open(data_path, "r") as f:
            return json.load(f)
    except Exception:
        return []

def get_historical_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "temperature_data.csv")
    if not os.path.exists(data_path):
        return pd.DataFrame()
    return pd.read_csv(data_path)

def generate_city_unified_forecast(city: str):
    """Dynamically generate forecast & heatscores for a city if not cached."""
    try:
        result = model_runner.generate_forecast(city)
        predictions_with_scores = predict_heatscores(city, result["predictions"])
        return {
            "city": city,
            "base_date": result["base_date"],
            "forecast": predictions_with_scores
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast for {city}: {e}")

@router.get("/{city}/current")
async def get_current_weather(city: str):
    """Get the current day's unified weather & heatscore prediction."""
    forecasts = get_forecast_data()
    for f in forecasts:
        if f.get("city", "").lower() == city.lower():
            preds = f.get("forecast", [])
            if preds:
                return {
                    "city": f.get("city"),
                    "base_date": f.get("base_date"),
                    "current": preds[0]
                }
    
    # Fallback to dynamic model execution
    dynamic_f = generate_city_unified_forecast(city)
    preds = dynamic_f.get("forecast", [])
    if not preds:
        raise HTTPException(status_code=404, detail="No forecast data found for city")
    return {
        "city": dynamic_f.get("city"),
        "base_date": dynamic_f.get("base_date"),
        "current": preds[0]
    }

@router.get("/{city}/forecast")
async def get_forecast_weather(city: str):
    """Get the full 16-day unified weather & heatscore prediction."""
    forecasts = get_forecast_data()
    for f in forecasts:
        if f.get("city", "").lower() == city.lower():
            return f
            
    # Fallback to dynamic model execution
    return generate_city_unified_forecast(city)

@router.get("/{city}/previous")
async def get_previous_weather(city: str, date: str = Query(..., description="Date in YYYY-MM-DD format")):
    """Get historical weather data from the dataset."""
    df = get_historical_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="Historical dataset not available")
        
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        year, month = dt.year, dt.month
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
    city_df = df[df['city'].str.lower() == city.lower()]
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found in historical dataset")

    if 'Date' in city_df.columns:
        match = city_df[city_df['Date'] == date]
    else:
        # Match year and month if listed, or fallback to matching month for latest available year
        match = city_df[(city_df['year'] == year) & (city_df['month'] == month)]
        if match.empty:
            latest_year = city_df['year'].max()
            match = city_df[(city_df['year'] == latest_year) & (city_df['month'] == month)]
        
    if match.empty:
        raise HTTPException(status_code=404, detail="No historical data found for this city and date")
        
    return match.iloc[0].to_dict()

