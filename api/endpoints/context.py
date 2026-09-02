from fastapi import APIRouter, HTTPException
import pandas as pd
import config
from api.schemas import ContextResponse

import os

router = APIRouter()

def _load_context_df():
    if os.path.exists(config.INDIA_CONTEXT_WEATHER_CSV):
        df = pd.read_csv(config.INDIA_CONTEXT_WEATHER_CSV)
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

@router.get("/india", response_model=ContextResponse)
async def get_india_context(date: str):
    """
    Get the context signals from sentinel cities across India for a given date.
    """
    try:
        df = _load_context_df()
        if df.empty:
            raise ValueError("Context data not available.")
        
        target_dt = pd.to_datetime(date)
        
        date_df = df[df['Date'] == target_dt].copy()
        
        if date_df.empty:
            raise ValueError(f"No context data found for date '{date}'.")
            
        date_df['Date'] = date_df['Date'].dt.strftime('%Y-%m-%d')
        date_df = date_df.where(pd.notnull(date_df), None)
        
        context_signals = {}
        for _, row in date_df.iterrows():
            city = row['City']
            row_dict = row.to_dict()
            context_signals[city] = row_dict
            
        return ContextResponse(
            date=date,
            sentinel_cities_analyzed=len(context_signals),
            context_signals=context_signals
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
