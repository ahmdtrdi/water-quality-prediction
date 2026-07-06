import pandas as pd
import numpy as np
from pathlib import Path
import os

def infer_status(df, name):
    if 'osm_status' not in df.columns:
        print(f"Warning: {name} is missing 'osm_status' column. Inferring from data...")
        return df.apply(
            lambda r: 'Failed' if r.get('osm_total_5000m', 0) == 0 and r.get('osm_roads_5000m', 0) == 0 else 'Success', 
            axis=1
        )
    return df['osm_status']

def merge_three_osm_sources(latest_local_path, old_local_path, kaggle_path, output_path):
    """
    Prioritized merge of 3 OSM sources:
    1. Prefer Latest Local (osm.parquet) if 'Success' (has the new expanded mine & waste query).
    2. Fallback to Old Local (osm-01.parquet) if 'Success'.
    3. Fallback to Kaggle (osm-kaggle.parquet) if 'Success'.
    4. Otherwise, fall back to Latest Local (Failed).
    """
    if not os.path.exists(latest_local_path):
        print(f"Error: Latest local file not found at {latest_local_path}")
        return
    if not os.path.exists(old_local_path):
        print(f"Error: Old local file not found at {old_local_path}")
        return
    if not os.path.exists(kaggle_path):
        print(f"Error: Kaggle file not found at {kaggle_path}")
        return

    latest_df = pd.read_parquet(latest_local_path)
    old_df = pd.read_parquet(old_local_path)
    kaggle_df = pd.read_parquet(kaggle_path)
    
    # Infer statuses
    latest_df['osm_status'] = infer_status(latest_df, 'latest_local')
    old_df['osm_status'] = infer_status(old_df, 'old_local')
    kaggle_df['osm_status'] = infer_status(kaggle_df, 'kaggle')
    
    # Set index on station_id
    latest_df = latest_df.set_index('station_id')
    old_df = old_df.set_index('station_id')
    kaggle_df = kaggle_df.set_index('station_id')
    
    all_stations = latest_df.index.union(old_df.index).union(kaggle_df.index)
    merged_rows = []
    
    # Counters for diagnostics
    source_counts = {
        'Latest Local (Success)': 0,
        'Old Local (Success)': 0,
        'Kaggle (Success)': 0,
        'Failed (Both/All)': 0
    }
    
    failed_both = []
    
    for sid in all_stations:
        in_latest = sid in latest_df.index
        in_old = sid in old_df.index
        in_kaggle = sid in kaggle_df.index
        
        status_latest = latest_df.loc[sid, 'osm_status'] if in_latest else 'Failed'
        status_old = old_df.loc[sid, 'osm_status'] if in_old else 'Failed'
        status_kaggle = kaggle_df.loc[sid, 'osm_status'] if in_kaggle else 'Failed'
        
        # Priority 1: Latest Local Success
        if in_latest and status_latest == 'Success':
            row = latest_df.loc[sid].to_dict()
            source_counts['Latest Local (Success)'] += 1
            
        # Priority 2: Old Local Success
        elif in_old and status_old == 'Success':
            row = old_df.loc[sid].to_dict()
            source_counts['Old Local (Success)'] += 1
            
        # Priority 3: Kaggle Success
        elif in_kaggle and status_kaggle == 'Success':
            row = kaggle_df.loc[sid].to_dict()
            source_counts['Kaggle (Success)'] += 1
            
        # Priority 4: Fallback to Latest Local (Failed)
        else:
            row = latest_df.loc[sid].to_dict() if in_latest else (old_df.loc[sid].to_dict() if in_old else kaggle_df.loc[sid].to_dict())
            source_counts['Failed (Both/All)'] += 1
            lat = row.get('Latitude', np.nan)
            lon = row.get('Longitude', np.nan)
            failed_both.append(f"Station: {sid} (Lat: {lat}, Lon: {lon})")
            
        row['station_id'] = sid
        merged_rows.append(row)
        
    merged_df = pd.DataFrame(merged_rows)
    
    # Reorder columns
    cols = ['station_id', 'Latitude', 'Longitude', 'osm_status']
    other_cols = [c for c in merged_df.columns if c not in cols]
    merged_df = merged_df[cols + other_cols]
    
    # Save output
    merged_df.to_parquet(output_path, index=False)
    
    print("\n" + "="*50)
    print(" 3-WAY OSM MERGE SUMMARY")
    print("="*50)
    print(f"Merged output saved to : {output_path}")
    print(f"Total Unique Stations  : {len(merged_df)}")
    print(f"Success count          : {sum(merged_df['osm_status'] == 'Success')}")
    print(f"Failed count           : {sum(merged_df['osm_status'] == 'Failed')}")
    print("\nBreakdown of chosen sources:")
    for k, v in source_counts.items():
        print(f" - {k:24s}: {v} stations ({v/len(merged_df)*100:.1f}%)")
        
    if failed_both:
        print("\nStations that failed in ALL environments (all 0s):")
        for f in failed_both:
            print(f" - {f}")

if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent.parent
    latest_local = BASE_DIR / 'data' / '03-external' / 'osm.parquet'
    old_local = BASE_DIR / 'data' / '03-external' / 'osm-01.parquet'
    kaggle = BASE_DIR / 'data' / '03-external' / 'osm-kaggle.parquet'
    output = BASE_DIR / 'data' / '03-external' / 'osm.parquet' # Overwrite with merged version
    
    print(f"Merging OSM data sources...\nLatest Local: {latest_local}\nOld Local   : {old_local}\nKaggle      : {kaggle}")
    merge_three_osm_sources(latest_local, old_local, kaggle, output)
