import pandas as pd
import numpy as np
import os
import time
import requests
import logging
from pathlib import Path

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('soilgrids_failed_only')

# Config local paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'data' / '02-processed'
OUTPUT_DIR = BASE_DIR / 'data' / '03-external'

LAT_COL     = 'Latitude'
LON_COL     = 'Longitude'
STATION_COL = 'station_id'

SOIL_PROPERTIES = ['phh2o', 'clay', 'sand', 'silt', 'ocd', 'cec']

def fetch_soilgrids_batch(lat, lon, station_id, station_num, total_stations, retries=5):
    """
    Fetches all 6 soil properties in a single batch query (reduces calls by 6x).
    Uses custom headers and exponential backoff to handle rate limits and timeouts.
    """
    url = 'https://rest.isric.org/soilgrids/v2.0/properties/query'
    params = [
        ('lat', lat),
        ('lon', lon),
        ('depth', '0-5cm'),
        ('value', 'mean')
    ]
    for prop in SOIL_PROPERTIES:
        params.append(('property', prop))
        
    headers = {
        'User-Agent': 'curl/8.7.1',
        'Accept': '*/*'
    }
    
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=45)
            r.raise_for_status()
            data = r.json()
            layers = data.get('properties', {}).get('layers', [])
            
            results = {}
            for layer in layers:
                name = layer.get('name')
                depths = layer.get('depths', [])
                if depths:
                    val = depths[0].get('values', {}).get('mean')
                    results[f'soil_{name}'] = float(val) if val is not None else np.nan
                else:
                    results[f'soil_{name}'] = np.nan
                    
            # Check if we got all 6 properties
            if len(results) == len(SOIL_PROPERTIES):
                log.info(f"[{station_num:3d}/{total_stations}] {station_id[:20]:20s} -> Success (pH: {results.get('soil_phh2o', np.nan)/10:.1f})")
                return results
            else:
                raise ValueError(f"Expected {len(SOIL_PROPERTIES)} properties, got {len(results)}")
                
        except Exception as e:
            sleep_time = 6 * (attempt + 1)
            if attempt < retries - 1:
                log.warning(f"[{station_num}/{total_stations}] Query failed ({type(e).__name__}). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                log.error(f"[{station_num}/{total_stations}] {station_id[:20]:20s} -> FAILED ALL RETRIES (Error: {type(e).__name__})")
                
    return {f'soil_{p}': np.nan for p in SOIL_PROPERTIES}

def main():
    out_path = OUTPUT_DIR / 'soilgrids.parquet'
    if not out_path.exists():
        log.error(f"File not found: {out_path}")
        print("Please run the main soilgrids extraction first or make sure soilgrids.parquet is in data/03-external/")
        return
        
    df = pd.read_parquet(out_path)
    
    # Identify failed rows (rows where any soil property is null)
    soil_cols = [f'soil_{p}' for p in SOIL_PROPERTIES]
    for col in soil_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    failed_mask = df[soil_cols].isnull().any(axis=1)
    failed_indices = df[failed_mask].index
    total_failed = len(failed_indices)
    
    if total_failed == 0:
        log.info("ALL STATIONS ARE SUCCESSFUL! No failed/null soil coordinates found to pull.")
        return
        
    log.info(f"Found {total_failed} failed/null stations out of {len(df)} total stations.")
    log.info("Extracting data ONLY for these failed coordinates using optimized batch requests...")
    
    results_updated = 0
    start_time = time.time()
    
    for idx, df_idx in enumerate(failed_indices):
        row = df.loc[df_idx]
        lat, lon = row[LAT_COL], row[LON_COL]
        station = row[STATION_COL]
        
        soil_data = fetch_soilgrids_batch(lat, lon, station, idx + 1, total_failed)
        
        # Update row in place if successfully fetched
        if not pd.isna(list(soil_data.values())[0]):
            for k, v in soil_data.items():
                df.loc[df_idx, k] = v
            results_updated += 1
            
        # Throttling to respect SoilGrids API fair use limits (approx 10 requests per minute)
        time.sleep(6.0)
        
    # Save updated file
    df.to_parquet(out_path, index=False)
    
    elapsed = time.time() - start_time
    size_kb = os.path.getsize(out_path) / 1024
    log.info(f"DONE in {elapsed/60:.1f} min. Updated {results_updated}/{total_failed} stations. Saved to: {out_path} ({size_kb:.1f} KB)")
    
    # Print final null counts
    print("\nFinal Null Counts per column:")
    for col in soil_cols:
        print(f" - {col}: {df[col].isnull().sum()} nulls")

if __name__ == '__main__':
    main()
