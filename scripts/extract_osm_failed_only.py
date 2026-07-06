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
log = logging.getLogger('osm_failed_only')

# Config local paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'data' / '02-processed'
OUTPUT_DIR = BASE_DIR / 'data' / '03-external'

LAT_COL     = 'Latitude'
LON_COL     = 'Longitude'
STATION_COL = 'station_id'

OVERPASS_ENDPOINTS = [
    'http://overpass-api.de/api/interpreter',          # Main (DE)
    'https://lz4.overpass-api.de/api/interpreter',     # Mirror lz4 (DE)
    'https://z.overpass-api.de/api/interpreter',       # Mirror z (DE)
    'https://overpass.kumi.systems/api/interpreter'    # Mirror Kumi (DE)
]

def fetch_osm_station(lat, lon, station_id, station_num, total_stations, retries=4):
    """
    Fetches all 12 OSM counts (4 categories x 3 radii) in a single batch query.
    Uses expanded queries for mines (mine/quarry) and wastewater (wastewater_plant/sewerage_works)
    to ensure complete geospacial feature extraction in South Africa.
    """
    query = f"""[out:json][timeout:90];
    (node(around:1000,{lat},{lon})["man_made"="mine"];way(around:1000,{lat},{lon})["man_made"="mine"];node(around:1000,{lat},{lon})["landuse"="quarry"];way(around:1000,{lat},{lon})["landuse"="quarry"];node(around:1000,{lat},{lon})["industrial"="mine"];way(around:1000,{lat},{lon})["industrial"="mine"];)->.m1;.m1 out count;
    (node(around:1000,{lat},{lon})["man_made"="wastewater_plant"];way(around:1000,{lat},{lon})["man_made"="wastewater_plant"];node(around:1000,{lat},{lon})["man_made"="sewerage_works"];way(around:1000,{lat},{lon})["man_made"="sewerage_works"];)->.w1;.w1 out count;
    way(around:1000,{lat},{lon})["landuse"="farmland"]->.f1;.f1 out count;
    way(around:1000,{lat},{lon})["highway"]->.h1;.h1 out count;

    (node(around:5000,{lat},{lon})["man_made"="mine"];way(around:5000,{lat},{lon})["man_made"="mine"];node(around:5000,{lat},{lon})["landuse"="quarry"];way(around:5000,{lat},{lon})["landuse"="quarry"];node(around:5000,{lat},{lon})["industrial"="mine"];way(around:5000,{lat},{lon})["industrial"="mine"];)->.m5;.m5 out count;
    (node(around:5000,{lat},{lon})["man_made"="wastewater_plant"];way(around:5000,{lat},{lon})["man_made"="wastewater_plant"];node(around:5000,{lat},{lon})["man_made"="sewerage_works"];way(around:5000,{lat},{lon})["man_made"="sewerage_works"];)->.w5;.w5 out count;
    way(around:5000,{lat},{lon})["landuse"="farmland"]->.f5;.f5 out count;
    way(around:5000,{lat},{lon})["highway"]->.h5;.h5 out count;

    (node(around:10000,{lat},{lon})["man_made"="mine"];way(around:10000,{lat},{lon})["man_made"="mine"];node(around:10000,{lat},{lon})["landuse"="quarry"];way(around:10000,{lat},{lon})["landuse"="quarry"];node(around:10000,{lat},{lon})["industrial"="mine"];way(around:10000,{lat},{lon})["industrial"="mine"];)->.m10;.m10 out count;
    (node(around:10000,{lat},{lon})["man_made"="wastewater_plant"];way(around:10000,{lat},{lon})["man_made"="wastewater_plant"];node(around:10000,{lat},{lon})["man_made"="sewerage_works"];way(around:10000,{lat},{lon})["man_made"="sewerage_works"];)->.w10;.w10 out count;
    way(around:10000,{lat},{lon})["landuse"="farmland"]->.f10;.f10 out count;
    way(around:10000,{lat},{lon})["highway"]->.h10;.h10 out count;
    """
    
    keys = [
        'osm_mines_1000m', 'osm_wastewater_1000m', 'osm_farmland_1000m', 'osm_roads_1000m',
        'osm_mines_5000m', 'osm_wastewater_5000m', 'osm_farmland_5000m', 'osm_roads_5000m',
        'osm_mines_10000m', 'osm_wastewater_10000m', 'osm_farmland_10000m', 'osm_roads_10000m'
    ]
    
    headers = {
        'User-Agent': 'curl/8.7.1',
        'Accept': '*/*'
    }
    
    for attempt in range(retries):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.get(endpoint, params={'data': query}, headers=headers, timeout=95)
            r.raise_for_status()
            elements = r.json().get('elements', [])
            
            if len(elements) == 12:
                results = {}
                for i, key in enumerate(keys):
                    count = int(elements[i].get('tags', {}).get('total', 0))
                    results[key] = count
                
                results['osm_total_1000m'] = sum(results[k] for k in keys[0:4])
                results['osm_total_5000m'] = sum(results[k] for k in keys[4:8])
                results['osm_total_10000m'] = sum(results[k] for k in keys[8:12])
                results['osm_status'] = 'Success'
                
                log.info(f"[{station_num:3d}/{total_stations}] {station_id[:20]:20s} -> Success via {endpoint.split('/')[2]}")
                print(f"    Mines  (1k/5k/10k): [{results['osm_mines_1000m']}/{results['osm_mines_5000m']}/{results['osm_mines_10000m']}] | "
                      f"Waste: [{results['osm_wastewater_1000m']}/{results['osm_wastewater_5000m']}/{results['osm_wastewater_10000m']}]")
                print(f"    Farm   (1k/5k/10k): [{results['osm_farmland_1000m']}/{results['osm_farmland_5000m']}/{results['osm_farmland_10000m']}] | "
                      f"Roads: [{results['osm_roads_1000m']}/{results['osm_roads_5000m']}/{results['osm_roads_10000m']}]")
                return results
            else:
                raise ValueError(f"Expected 12 elements, got {len(elements)}")
                
        except Exception as e:
            sleep_time = 4 * (attempt + 1)
            if attempt < retries - 1:
                log.warning(f"[{station_num}/{total_stations}] Endpoint '{endpoint.split('/')[2]}' failed ({type(e).__name__}). Rotating in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                log.error(f"[{station_num}/{total_stations}] {station_id[:20]:20s} -> FAILED ALL RETRIES (Error: {type(e).__name__})")
                
    fallback = {k: 0 for k in keys}
    fallback['osm_total_1000m'] = 0
    fallback['osm_total_5000m'] = 0
    fallback['osm_total_10000m'] = 0
    fallback['osm_status'] = 'Failed'
    return fallback

def main():
    out_path = OUTPUT_DIR / 'osm.parquet'
    if not out_path.exists():
        log.error(f"File not found: {out_path}")
        print("Please run the main OSM extraction script first to generate the base osm.parquet file.")
        return
        
    df = pd.read_parquet(out_path)
    
    # Infer osm_status if missing
    if 'osm_status' not in df.columns:
        df['osm_status'] = df.apply(
            lambda r: 'Failed' if r.get('osm_total_5000m', 0) == 0 and r.get('osm_roads_5000m', 0) == 0 else 'Success', 
            axis=1
        )
        
    failed_mask = df['osm_status'] == 'Failed'
    failed_indices = df[failed_mask].index
    total_failed = len(failed_indices)
    
    if total_failed == 0:
        log.info("ALL STATIONS ARE SUCCESSFUL! No failed coordinates found to pull.")
        return
        
    log.info(f"Found {total_failed} failed coordinates out of {len(df)} total stations.")
    log.info("Extracting data ONLY for these failed coordinates...")
    
    results_updated = 0
    
    for idx, df_idx in enumerate(failed_indices):
        row = df.loc[df_idx]
        lat, lon = row[LAT_COL], row[LON_COL]
        station = row[STATION_COL]
        
        osm_data = fetch_osm_station(lat, lon, station, idx + 1, total_failed)
        
        # Update row in place
        for k, v in osm_data.items():
            df.loc[df_idx, k] = v
            
        if osm_data['osm_status'] == 'Success':
            results_updated += 1
            
        time.sleep(2.0) # Polite delay
        
    # Save updated file
    df.to_parquet(out_path, index=False)
    
    size_kb = os.path.getsize(out_path) / 1024
    log.info(f"DONE. Updated {results_updated}/{total_failed} failed stations. Saved to: {out_path} ({size_kb:.1f} KB)")
    print(f"\nFinal status counts:")
    print(df['osm_status'].value_counts())

if __name__ == '__main__':
    main()
