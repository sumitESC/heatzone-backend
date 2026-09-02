import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import requests

def update_weather_file(csv_path, cities_json_path, daily_vars, is_context=False):
    print(f"\n--- Updating {csv_path} ---")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    if 'Date' not in df.columns:
        print("Date column missing.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    max_date = df['Date'].max()
    start_date = (max_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date > end_date:
        print(f"Data is already up to date (Latest: {max_date.strftime('%Y-%m-%d')}). No new data to fetch.")
        return

    print(f"Gap detected! Fetching missing data from {start_date} to {end_date}...")

    # Load city coordinates
    with open(cities_json_path, "r") as f:
        city_data = json.load(f)

    city_map = {}
    if is_context:
        for c in city_data.get("cities", []):
            city_map[c["name"]] = {"lat": c["latitude"], "lon": c["longitude"], "state": c.get("state", ""), "event": c.get("climate_event", "")}
    else:
        for c in city_data.get("cities", []) + city_data.get("all_75_districts", []):
            city_map[c["name"]] = {"lat": c["latitude"], "lon": c["longitude"]}

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    # Fallback to forecast API for very recent days (last 5 days)
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    
    # Decide which API to use based on how recent the start_date is
    days_ago = (datetime.now() - (max_date + timedelta(days=1))).days
    url_to_use = forecast_url if days_ago <= 5 else base_url

    new_rows = []
    unique_cities = df['City'].unique()

    for city in unique_cities:
        if city not in city_map:
            continue
            
        lat, lon = city_map[city]['lat'], city_map[city]['lon']
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(daily_vars),
            "timezone": "Asia/Kolkata",
        }
        
        while True:
            try:
                resp = requests.get(url_to_use, params=params, timeout=30)
                if resp.status_code == 429:
                    print("Rate limit hit, sleeping 5s...")
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"Error fetching {city}: {e}")
                time.sleep(2)
                break # break to avoid infinite loop on bad city
                
        if 'daily' not in data:
            continue
            
        daily = data.get("daily", {})
        times = daily.get("time", [])
        
        for i, date_str in enumerate(times):
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if is_context:
                row = [city, city_map[city]['state'], city_map[city]['event'], date_str, dt.year, dt.month, dt.day]
            else:
                row = [city, date_str, dt.year, dt.month, dt.day]
                
            for var in daily_vars:
                val = daily.get(var, [None] * len(times))
                row.append(val[i] if i < len(val) else None)
            new_rows.append(row)
            
        print(f"Fetched {city} ({len(times)} days)")
        time.sleep(0.5)

    if new_rows:
        cols = list(df.columns)
        new_df = pd.DataFrame(new_rows, columns=cols)
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        full_df = pd.concat([df, new_df])
        full_df = full_df.drop_duplicates(subset=["City", "Date"], keep="last")
        full_df.sort_values(by=["City", "Date"], inplace=True)
        
        full_df['Date'] = full_df['Date'].dt.strftime('%Y-%m-%d')
        full_df.to_csv(csv_path, index=False)
        print(f"Successfully appended {len(new_rows)} new records to {csv_path}!")
    else:
        print("No new records were fetched.")

def main():
    print("Starting Weather Data Updater...")
    
    daily_vars = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min",
        "precipitation_sum", "rain_sum", "wind_speed_10m_max",
        "wind_gusts_10m_max", "wind_direction_10m_dominant",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration",
        "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min",
        "dew_point_2m_mean", "pressure_msl_mean",
        "soil_temperature_0_to_7cm_mean", "soil_moisture_0_to_7cm_mean"
    ]

    # 1. Update the Main UP Weather Data (75 Cities)
    main_weather_csv = os.path.join("data", "processed", "india_regional_weather.csv")
    main_cities_json = os.path.join("data", "all_up_cities.json")
    update_weather_file(main_weather_csv, main_cities_json, daily_vars, is_context=False)

    # 2. Update the Context Weather Data (19 Sentinel Cities)
    context_weather_csv = os.path.join("data", "india_context_weather.csv")
    context_cities_json = os.path.join("data", "india_context_cities.json")
    update_weather_file(context_weather_csv, context_cities_json, daily_vars, is_context=True)
    
    print("\nData update complete! Your datasets are now current.")

if __name__ == "__main__":
    main()
