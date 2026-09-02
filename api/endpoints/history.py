from fastapi import APIRouter, HTTPException
import pandas as pd
import config
from api.schemas import HistoryResponse

import os

router = APIRouter()

def _load_history_df():
    if os.path.exists(config.UP_WEATHER_CSV):
        df = pd.read_csv(config.UP_WEATHER_CSV)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    fallback_path = os.path.join(config.DATA_DIR, "temperature_data.csv")
    if os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
        df = df.rename(columns={"city": "City"})
        if "Date" not in df.columns:
            df["Date"] = pd.to_datetime(
                df["year"].astype(str) + "-" + 
                df["month"].astype(str).str.zfill(2) + "-01"
            )
        return df
    return pd.DataFrame()

@router.get("/{city}", response_model=HistoryResponse)
async def get_historical_weather(city: str, start_date: str, end_date: str):
    """
    Get historical weather data for a city.
    """
    try:
        df = _load_history_df()
        if df.empty:
            raise ValueError("Historical data not available.")
        
        city_df = df[df['City'].str.lower() == city.lower()]
        
        if city_df.empty:
            raise ValueError(f"City '{city}' not found.")
            
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (city_df['Date'] >= start_dt) & (city_df['Date'] <= end_dt)
        filtered_df = city_df.loc[mask].copy()
        
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')
        filtered_df = filtered_df.where(pd.notnull(filtered_df), None)
        
        data = filtered_df.to_dict(orient='records')
        
        return HistoryResponse(
            city=city,
            start_date=start_date,
            end_date=end_date,
            data=data
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
