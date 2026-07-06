import pandas as pd
import numpy as np
from pathlib import Path
import os

def infer_status(df, name):
    if 'osm_status' not in df.columns:
        return df.apply(
            lambda r: 'Failed' if r.get('osm_total_5000m', 0) == 0 and r.get('osm_roads_5000m', 0) == 0 else 'Success', 
            axis=1
        )
    return df['osm_status']

def analyze_osm_sources(latest_local_path, old_local_path, kaggle_path):
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
    
    latest_df['osm_status'] = infer_status(latest_df, 'latest_local')
    old_df['osm_status'] = infer_status(old_df, 'old_local')
    kaggle_df['osm_status'] = infer_status(kaggle_df, 'kaggle')
    
    latest_df = latest_df.set_index('station_id')
    old_df = old_df.set_index('station_id')
    kaggle_df = kaggle_df.set_index('station_id')
    
    all_stations = latest_df.index.union(old_df.index).union(kaggle_df.index)
    
    stats = {
        'latest_success': set(latest_df[latest_df['osm_status'] == 'Success'].index),
        'latest_failed': set(latest_df[latest_df['osm_status'] == 'Failed'].index),
        'old_success': set(old_df[old_df['osm_status'] == 'Success'].index),
        'old_failed': set(old_df[old_df['osm_status'] == 'Failed'].index),
        'kaggle_success': set(kaggle_df[kaggle_df['osm_status'] == 'Success'].index),
        'kaggle_failed': set(kaggle_df[kaggle_df['osm_status'] == 'Failed'].index),
    }
    
    # Overlap analysis
    latest_fail_but_old_success = stats['latest_failed'].intersection(stats['old_success'])
    latest_fail_but_kaggle_success = stats['latest_failed'].intersection(stats['kaggle_success'])
    latest_fail_but_both_success = stats['latest_failed'].intersection(stats['old_success']).intersection(stats['kaggle_success'])
    latest_fail_but_any_success = stats['latest_failed'].intersection(stats['old_success'].union(stats['kaggle_success']))
    
    failed_all = stats['latest_failed'].intersection(stats['old_failed']).intersection(stats['kaggle_failed'])
    
    print("\n" + "="*50)
    print(" OSM DATASETS OVERLAP DIAGNOSTIC")
    print("="*50)
    print(f"Total Unique Stations across all files: {len(all_stations)}")
    print(f"Latest Local Success: {len(stats['latest_success'])} | Failed: {len(stats['latest_failed'])}")
    print(f"Old Local Success   : {len(stats['old_success'])} | Failed: {len(stats['old_failed'])}")
    print(f"Kaggle Success      : {len(stats['kaggle_success'])} | Failed: {len(stats['kaggle_failed'])}")
    print("-"*50)
    print(f"Stations Failed in Latest Local but Succeeded in:")
    print(f" - Old Local Only     : {len(latest_fail_but_old_success - latest_fail_but_kaggle_success)} stations")
    print(f" - Kaggle Only        : {len(latest_fail_but_kaggle_success - latest_fail_but_old_success)} stations")
    print(f" - Both Old & Kaggle  : {len(latest_fail_but_both_success)} stations")
    print(f" - ANY of them        : {len(latest_fail_but_any_success)} stations (These will be successfully merged!)")
    print("-"*50)
    print(f"Stations Failed in ALL three files (will remain 0): {len(failed_all)}")
    
    if latest_fail_but_any_success:
        print("\nDetail of stations to be recovered from other files:")
        for sid in latest_fail_but_any_success:
            loc = "Old Local" if sid in stats['old_success'] else ""
            kag = "Kaggle" if sid in stats['kaggle_success'] else ""
            sources = " & ".join(filter(None, [loc, kag]))
            lat = latest_df.loc[sid, 'Latitude'] if sid in latest_df.index else old_df.loc[sid, 'Latitude']
            lon = latest_df.loc[sid, 'Longitude'] if sid in latest_df.index else old_df.loc[sid, 'Longitude']
            print(f" - {sid} (Lat: {lat:.6f}, Lon: {lon:.6f}) -> Recoverable from: {sources}")
            
    if failed_all:
        print("\nDetail of stations that failed in ALL files (all 0s):")
        for sid in failed_all:
            row = latest_df.loc[sid] if sid in latest_df.index else old_df.loc[sid]
            print(f" - {sid} (Lat: {row['Latitude']:.6f}, Lon: {row['Longitude']:.6f})")

if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent.parent
    latest_local = BASE_DIR / 'data' / '03-external' / 'osm.parquet'
    old_local = BASE_DIR / 'data' / '03-external' / 'osm-01.parquet'
    kaggle = BASE_DIR / 'data' / '03-external' / 'osm-kaggle.parquet'
    analyze_osm_sources(latest_local, old_local, kaggle)
