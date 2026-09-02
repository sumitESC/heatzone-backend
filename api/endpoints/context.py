from fastapi import APIRouter, HTTPException
import pandas as pd
import config
from api.schemas import ContextResponse

router = APIRouter()

@router.get("/india", response_model=ContextResponse)
async def get_india_context(date: str):
    """
    Get the context signals from sentinel cities across India for a given date.
    """
    try:
        df = pd.read_csv(config.INDIA_CONTEXT_WEATHER_CSV)
        df['Date'] = pd.to_datetime(df['Date'])
        
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
