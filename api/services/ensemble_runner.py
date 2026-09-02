import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Import the individual prediction functions
from api.services.model_runner import generate_satellite_forecast

def generate_ensemble_forecast(city: str, date=None, mc_samples=10):
    """
    Combines the SAT model with the LSTM/XGBoost ensemble to create
    the ultimate unified weather forecast.
    
    v2: Propagates rain_probability and dynamic confidence scores.
    """
    print(f"\n============================================================")
    print(f"Generating UNIFIED ENSEMBLE forecast for {city}")
    print(f"============================================================")
    
    # 1. Get SAT forecast
    print("\n[Phase 1] Running Spatiotemporal SAT Model...")
    sat_forecast = generate_satellite_forecast(city, date=date, mc_samples=mc_samples)
    sat_preds = sat_forecast.get("predictions", [])
    
    # 2. Get LSTM+XGB forecast (returns a DataFrame)
    print("\n[Phase 2] Running LSTM-Transformer + XGBoost Baseline...")
    lstm_xgb_df = None
    try:
        from models.lstm_transformer.ensemble import ensemble_predict as lstm_xgb_ensemble_predict
        lstm_xgb_df = lstm_xgb_ensemble_predict(city, date=date)
    except Exception as e:
        print(f"  LSTM/XGBoost prediction failed: {e}")
    
    # 3. Combine them
    print("\n[Phase 3] Fusing Multi-Model Ensembles...")
    final_predictions = []
    
    # SAT model predictions is a list of dicts
    for i, sat_day in enumerate(sat_preds):
        combined_day = sat_day.copy()
        
        if lstm_xgb_df is not None and i < len(lstm_xgb_df):
            lx_day = lstm_xgb_df.iloc[i]
            
            # Combine metrics if present in both
            for metric in ["Temp_Max_C", "Temp_Min_C", "Precipitation_mm", "Humidity_Mean_pct", "Wind_Speed_Max_kmh", "Pressure_MSL_hPa", "Shortwave_Radiation_MJm2"]:
                if metric in sat_day and metric in lstm_xgb_df.columns:
                    sat_val = sat_day[metric]
                    lx_val = lx_day[metric]
                    if not pd.isna(lx_val):
                        if metric == "Precipitation_mm":
                            # Use LSTM's two-stage prediction as primary for rain
                            # (it has the classifier gate + dedicated regression head)
                            rain_prob = float(lx_day.get("rain_probability", 0.0)) if "rain_probability" in lstm_xgb_df.columns else 0.0
                            
                            if rain_prob < 0.2:
                                # High confidence no-rain from LSTM classifier
                                combined_day[metric] = 0.0
                            elif rain_prob >= 0.2 and float(lx_val) > 0:
                                # Rain predicted: average SAT and LSTM amounts
                                # but weight LSTM higher since it has the dedicated rain head
                                sat_rain = max(0, float(sat_val))
                                lstm_rain = float(lx_val)
                                combined_day[metric] = 0.4 * sat_rain + 0.6 * lstm_rain
                            else:
                                combined_day[metric] = float(lx_val)
                        else:
                            combined_day[metric] = (sat_val + float(lx_val)) / 2.0
            
            # Propagate rain_probability from LSTM
            if "rain_probability" in lstm_xgb_df.columns:
                combined_day["rain_probability"] = float(lx_day["rain_probability"])
            
            # Propagate dynamic confidence from LSTM
            if "confidence" in lstm_xgb_df.columns:
                combined_day["confidence"] = float(lx_day["confidence"])
            else:
                # Fallback: horizon-based decay
                combined_day["confidence"] = round(0.95 * np.exp(-0.045 * i), 3)
                        
        final_predictions.append(combined_day)
        
    return {
        "predictions": final_predictions,
        "base_date": sat_forecast.get("base_date", "")
    }
