import json
import logging
import math
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import schedule
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Configuration
CITIES_FILE = r"d:\code\dataset_heatzone\heatzone-weather-api\data\all_up_cities.json"
OUTPUT_DIR = r"d:\code\dataset_heatzone\heatzone-weather-api\data\weather_data"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weather_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_city_grid(center_lat, center_lon, area_sqkm=150, spacing_km=5):
    """
    Generates a grid of coordinates around a center point covering the given area.
    """
    # Side length of the square bounding box
    side_length_km = math.sqrt(area_sqkm)
    half_side = side_length_km / 2.0
    
    # 1 degree latitude is approx 111 km
    lat_offset = half_side / 111.0
    
    # 1 degree longitude varies by latitude
    lon_offset = half_side / (111.0 * math.cos(math.radians(center_lat)))
    
    min_lat = center_lat - lat_offset
    max_lat = center_lat + lat_offset
    min_lon = center_lon - lon_offset
    max_lon = center_lon + lon_offset
    
    # Step sizes in degrees
    lat_step = spacing_km / 111.0
    lon_step = spacing_km / (111.0 * math.cos(math.radians(center_lat)))
    
    lats = []
    lons = []
    
    current_lat = min_lat
    while current_lat <= max_lat:
        current_lon = min_lon
        while current_lon <= max_lon:
            lats.append(round(current_lat, 4))
            lons.append(round(current_lon, 4))
            current_lon += lon_step
        current_lat += lat_step
        
    return lats, lons

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))
)
def fetch_openmeteo(lats, lons, start_date, end_date, is_historical=True):
    """
    Fetches weather data from Open-Meteo. Uses archive API for historical, forecast API for recent.
    """
    if is_historical:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    # Open-Meteo accepts multiple coordinates as comma-separated lists
    lat_str = ",".join(map(str, lats))
    lon_str = ",".join(map(str, lons))

    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=30)
    
    if response.status_code == 429:
        logger.warning("Rate limit exceeded. Retrying...")
        response.raise_for_status()

    if response.status_code != 200:
        logger.error(f"API Error {response.status_code}: {response.text}")
        response.raise_for_status()

    data = response.json()
    if "error" in data:
        raise ValueError(f"Open-Meteo returned an error: {data.get('reason')}")

    return data

def process_api_response(city_name, data, lats, lons):
    """
    Converts Open-Meteo JSON response into a pandas DataFrame.
    """
    records = []
    
    # If only 1 coordinate was requested, the response is a dict, not a list of dicts.
    # Open-Meteo returns a list of results if multiple coords are passed.
    if isinstance(data, dict) and "hourly" in data:
        results = [data]
    elif isinstance(data, list):
        results = data
    else:
        logger.error(f"Unexpected data format for {city_name}")
        return pd.DataFrame()
        
    for i, res in enumerate(results):
        lat = lats[i] if i < len(lats) else None
        lon = lons[i] if i < len(lons) else None
        
        hourly = res.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        wind = hourly.get("wind_speed_10m", [])
        precip = hourly.get("precipitation", [])
        
        for t, temp, hum, wnd, prc in zip(times, temps, humidity, wind, precip):
            records.append({
                "city": city_name,
                "latitude": lat,
                "longitude": lon,
                "timestamp": t,
                "temperature_2m": temp,
                "relative_humidity_2m": hum,
                "wind_speed_10m": wnd,
                "precipitation": prc,
                "fetched_at": datetime.now().isoformat()
            })
            
    df = pd.DataFrame(records)
    if not df.empty:
        # Convert timestamp to datetime and extract year for partitioning
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year
    return df

def get_last_fetched_date(city_name):
    """
    Determines the last date we successfully fetched data for a city.
    Returns Jan 1, 2000 if no data exists.
    """
    city_dir = os.path.join(OUTPUT_DIR, f"city={city_name}")
    if not os.path.exists(city_dir):
        return datetime(2000, 1, 1).date()
        
    latest_date = datetime(2000, 1, 1).date()
    
    # Simple logic: Read all parquet files for this city to find the max date
    # In a production system, a metadata DB (SQLite) would be faster
    try:
        df = pd.read_parquet(city_dir)
        if not df.empty and "timestamp" in df.columns:
            max_ts = pd.to_datetime(df["timestamp"]).max()
            latest_date = max_ts.date()
    except Exception as e:
        logger.warning(f"Could not read parquet for {city_name} to determine last date: {e}")
        
    return latest_date

def fetch_data_for_city(city, start_date, end_date):
    """
    Fetches data for a single city between start_date and end_date.
    Chunks the requests by year to avoid API response size limits.
    """
    city_name = city["name"]
    center_lat = city["latitude"]
    center_lon = city["longitude"]
    area = city.get("total_area_sqkm", 150)
    
    lats, lons = generate_city_grid(center_lat, center_lon, area)
    logger.info(f"Generated {len(lats)} grid points for {city_name}.")
    
    current_start = start_date
    
    while current_start <= end_date:
        # Chunk by year
        current_end = datetime(current_start.year, 12, 31).date()
        if current_end > end_date:
            current_end = end_date
            
        logger.info(f"Fetching {city_name} from {current_start} to {current_end}...")
        
        # Open-Meteo archive API has data up to ~5 days ago. 
        # For dates within the last 5 days, we use the forecast API (is_historical=False)
        today = datetime.now().date()
        days_diff = (today - current_start).days
        
        is_historical = True
        if days_diff <= 5:
            is_historical = False
            
        try:
            data = fetch_openmeteo(lats, lons, current_start, current_end, is_historical=is_historical)
            df = process_api_response(city_name, data, lats, lons)
            
            if not df.empty:
                # Save to partitioned parquet
                df.to_parquet(
                    OUTPUT_DIR,
                    partition_cols=["city", "year"],
                    engine="pyarrow",
                    existing_data_behavior="delete_matching" 
                )
                logger.info(f"Saved {len(df)} records for {city_name} chunk {current_start.year}.")
                
        except Exception as e:
            logger.error(f"Failed to fetch data for {city_name} ({current_start} - {current_end}): {e}")
            
        # Move to next year
        current_start = datetime(current_start.year + 1, 1, 1).date()
        
        # Be nice to the API
        time.sleep(1)

def run_backfill():
    """
    Iterates through all cities and fetches missing historical data.
    """
    logger.info("Starting data collection/backfill process...")
    
    if not os.path.exists(CITIES_FILE):
        logger.error(f"Cities file not found: {CITIES_FILE}")
        return

    with open(CITIES_FILE, 'r') as f:
        data = json.load(f)
        cities = data.get("cities", [])
        
    if not cities:
        logger.error("No cities found in the JSON file.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now().date()
    
    for city in cities:
        city_name = city["name"]
        last_date = get_last_fetched_date(city_name)
        
        # If we already have data up to yesterday, no need to backfill fully
        if last_date >= today - timedelta(days=1):
            logger.info(f"{city_name} is up to date (last fetched: {last_date}).")
            continue
            
        logger.info(f"Backfilling {city_name} starting from {last_date}")
        fetch_data_for_city(city, last_date, today)

def run_hourly_update():
    """
    Job that runs every hour to fetch the latest data for all cities.
    """
    logger.info("Running scheduled hourly update...")
    today = datetime.now().date()
    
    with open(CITIES_FILE, 'r') as f:
        data = json.load(f)
        cities = data.get("cities", [])
        
    for city in cities:
        # We fetch the last 2 days just to be safe and overwrite recent data to ensure completeness
        start_date = today - timedelta(days=2)
        fetch_data_for_city(city, start_date, today)
        
    logger.info("Hourly update complete.")

if __name__ == "__main__":
    # 1. Backfill all historical data up to today
    run_backfill()
    
    # 2. Schedule the hourly update
    logger.info("Scheduling continuous hourly updates...")
    schedule.every().hour.at(":00").do(run_hourly_update)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
