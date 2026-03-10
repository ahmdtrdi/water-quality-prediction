import time
import os
import requests
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import Point
from typing import Dict, Any, Optional
from datetime import timedelta
from functools import lru_cache

def _execute_api_get_request(url: str, params: Dict[str, Any], max_retries: int = 5, backoff_factor: float = 1.0, timeout: int = 60) -> Optional[Dict]:
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait_time = backoff_factor * (2 ** attempt)
                print(f"Rate limit hit (429), retrying in {wait_time:.1f}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                print(f"HTTP Error: {e}")
                return None
        except requests.exceptions.RequestException as e:
            wait_time = backoff_factor * (2 ** attempt)
            print(f"Request error (e.g., timeout): {e}, retrying in {wait_time:.1f}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)
            continue
    print(f"Max retries exceeded for {url}")
    return None

@lru_cache(maxsize=2048)
def fetch_elevation(lat: float, lon: float) -> float:
    data = _execute_api_get_request("https://api.opentopodata.org/v1/aster30m", {"locations": f"{lat},{lon}"})
    return float(data["results"][0]["elevation"]) if data else 0.0

@lru_cache(maxsize=2048)
def fetch_historical_weather(latitude: float, longitude: float, target_date_str: str, time_lag_days: int = 7) -> Dict[str, float]:
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    target_date = pd.to_datetime(target_date_str, format='mixed', dayfirst=False)
    
    start_date = (target_date - timedelta(days=time_lag_days)).strftime('%Y-%m-%d')
    end_date = target_date.strftime('%Y-%m-%d')
    
    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["precipitation_sum", "windspeed_10m_max"],
        "timezone": "Africa/Johannesburg"
    }
    
    data = _execute_api_get_request(base_url, parameters)
    weather_metrics = {"total_precipitation": 0.0, "average_wind_speed": 0.0}
    
    if data and "daily" in data:
        precip_list = data["daily"].get("precipitation_sum", [])
        wind_list = data["daily"].get("windspeed_10m_max", [])
        valid_precip = [p for p in precip_list if p is not None]
        valid_wind = [w for w in wind_list if w is not None]
        
        if valid_precip: weather_metrics["total_precipitation"] = sum(valid_precip)
        if valid_wind: weather_metrics["average_wind_speed"] = sum(valid_wind) / len(valid_wind)
            
    return weather_metrics

@lru_cache(maxsize=2048)
def fetch_soilgrids_properties(latitude: float, longitude: float) -> Dict[str, float]:
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    properties = ["phh2o", "clay", "sand", "silt", "organic_carbon", "cec"]
    results = {}
    
    for prop in properties:
        params = {"lat": latitude, "lon": longitude, "property": prop, "depth": "0-5cm", "value": "mean"}
        data = _execute_api_get_request(base_url, params)
        if data and "properties" in data and "layers" in data["properties"]:
            try:
                mean_val = data["properties"]["layers"][0]["depths"][0]["values"]["mean"]
                results[f"soil_{prop}_mean_0_5cm"] = float(mean_val) if mean_val is not None else 0.0
            except (KeyError, IndexError):
                results[f"soil_{prop}_mean_0_5cm"] = 0.0
        else:
            results[f"soil_{prop}_mean_0_5cm"] = 0.0
    return results

@lru_cache(maxsize=2048)
def fetch_osm_pollution_counts(latitude: float, longitude: float, radius_m: int = 1000) -> Dict[str, int]:
    base_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:30];
    (
      node(around:{radius_m},{latitude},{longitude})["man_made"="mine"];
      way(around:{radius_m},{latitude},{longitude})["man_made"="mine"];
      node(around:{radius_m},{latitude},{longitude})["man_made"="wastewater_plant"];
      way(around:{radius_m},{latitude},{longitude})["man_made"="wastewater_plant"];
      node(around:{radius_m},{latitude},{longitude})["landuse"="farmland"];
      way(around:{radius_m},{latitude},{longitude})["landuse"="farmland"];
      way(around:{radius_m},{latitude},{longitude})["highway"];
    );
    out count;
    """
    data = _execute_api_get_request(base_url, {"data": query})
    counts = {"mine_count": 0, "wastewater_count": 0, "farmland_count": 0, "road_count": 0}
    
    if data and "elements" in data:
        for el in data["elements"]:
            if "tags" in el and el["tags"].get("man_made") == "mine": counts["mine_count"] += 1
            elif "tags" in el and el["tags"].get("man_made") == "wastewater_plant": counts["wastewater_count"] += 1
            elif "tags" in el and el["tags"].get("landuse") == "farmland": counts["farmland_count"] += 1
            elif "tags" in el and "highway" in el["tags"]: counts["road_count"] += 1
    return counts
    
# def fetch_sentinel2_features(latitude: float, longitude: float, target_date_str: str, window_days: int = 15) -> Dict[str, float]:
#     """
#     Fetches median Sentinel-2 surface reflectance values (B02, B03, B04, B08) 
#     using Planetary Computer STAC API to represent water surface properties.
#     """
#     default_metrics = {"s2_blue": 0.0, "s2_green": 0.0, "s2_red": 0.0, "s2_nir": 0.0}
    
#     try:
#         catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        
#         target_date = pd.to_datetime(target_date_str)
#         start_date = (target_date - timedelta(days=window_days)).strftime('%Y-%m-%d')
#         end_date = (target_date + timedelta(days=window_days)).strftime('%Y-%m-%d')
#         time_window = f"{start_date}/{end_date}"
        
#         buffer_deg = 0.005
#         bbox = [longitude - buffer_deg, latitude - buffer_deg, longitude + buffer_deg, latitude + buffer_deg]
        
#         search = catalog.search(
#             collections=["sentinel-2-l2a"],
#             bbox=bbox,
#             datetime=time_window,
#             query={"eo:cloud_cover": {"lt": 20}}
#         )
#         items = list(search.get_items())
        
#         if not items:
#             return default_metrics
            
#         resolution_scale = 10 / 111320.0
#         data_cube = stac_load(
#             items,
#             bands=["B02", "B03", "B04", "B08"],
#             crs="EPSG:4326",
#             resolution=resolution_scale,
#             bbox=bbox,
#             chunks={"x": 256, "y": 256}
#         )
        
#         median_composite = data_cube.median(dim="time").compute()
        
#         default_metrics["s2_blue"] = float(median_composite["B02"].mean().item()) / 10000.0
#         default_metrics["s2_green"] = float(median_composite["B03"].mean().item()) / 10000.0
#         default_metrics["s2_red"] = float(median_composite["B04"].mean().item()) / 10000.0
#         default_metrics["s2_nir"] = float(median_composite["B08"].mean().item()) / 10000.0
        
#         return default_metrics
#     except Exception as e:
#         print(f"Sentinel-2 STAC Error for ({latitude}, {longitude}): {e}")
#         return default_metrics

@lru_cache(maxsize=128)
def _get_loaded_static_layer(layer: str):
    base = "/tmp/"
    try:
        if layer == "worldpop":    return rasterio.open(base + "zaf_pop_2025_CN_100m_R2025A_v1.tif")
        if layer == "sanlc2022":   return rasterio.open(base + "SA_NLC_2022_ALBERS.tif")
        if layer == "sanlc2020":   return rasterio.open(base + "SA_NLC_2020_ALBERS.tif")
        if layer == "hydroatlas":  return gpd.read_parquet(base + "BasinATLAS_v10_lev12.parquet")
        if layer == "riveratlas":  return gpd.read_parquet(base + "RiverATLAS_Data_v10.parquet")
    except Exception as e:
        print(f"Load error {layer}: {e}")
        return None

def fetch_static_raster_value(latitude: float, longitude: float, layer_identifier: str, buffer_radius_meters: int = 1000) -> float:
    point_geometry = Point(longitude, latitude)
    degree_buffer = buffer_radius_meters / 111320.0
    buffered_area = point_geometry.buffer(degree_buffer)
    
    raster_source = _get_loaded_static_layer(layer_identifier)
    if raster_source is None: return 0.0
    
    try:
        masked_image, _ = mask(raster_source, [buffered_area], crop=True, nodata=0)
        valid_pixels = masked_image[0][masked_image[0] > 0]
        return float(valid_pixels.mean()) if valid_pixels.size > 0 else 0.0
    except ValueError:
        return 0.0

@lru_cache(maxsize=2)
def _load_raster_attribute_table(layer_identifier: str) -> Dict[float, str]:
    """
    Loads the Value Attribute Table (VAT) DBF file and returns a mapping dictionary.
    Implements caching to prevent redundant I/O operations.
    """
    base_directory = "/tmp/"
    file_mapping = {
        "sanlc2022": "SA_NLC_2022_ALBERS.tif.vat.dbf",
        "sanlc2020": "SA_NLC_2020_ALBERS.tif.vat.dbf"
    }
    
    target_file = file_mapping.get(layer_identifier)
    if not target_file:
        return {}
        
    try:
        dbf_dataframe = gpd.read_file(f"{base_directory}{target_file}")
        
        # Kolom standar pada SANLC biasanya 'Value' untuk angka dan 'LC_Class' (atau sejenisnya) untuk nama
        value_column = 'Value' if 'Value' in dbf_dataframe.columns else dbf_dataframe.columns[0]
        class_column = 'LC_Class' if 'LC_Class' in dbf_dataframe.columns else dbf_dataframe.columns[1]
        
        attribute_mapping = dict(zip(dbf_dataframe[value_column], dbf_dataframe[class_column]))
        return attribute_mapping
    except Exception as initialization_error:
        print(f"Failed to load attribute table for {layer_identifier}: {initialization_error}")
        return {}

def fetch_mapped_sanlc_class(latitude: float, longitude: float, layer_identifier: str, buffer_radius_meters: int = 1000) -> str:
    """
    Fetches the numerical raster value and maps it to its descriptive class name.
    """
    numeric_raster_value = fetch_static_raster_value(latitude, longitude, layer_identifier, buffer_radius_meters)
    
    if numeric_raster_value == 0.0:
        return "Unclassified/No Data"
        
    attribute_mapping = _load_raster_attribute_table(layer_identifier)
    
    # Menggunakan integer karena nilai raster kelas lahan diskrit tidak memiliki koma
    rounded_value = int(round(numeric_raster_value))
    
    mapped_class_name = attribute_mapping.get(rounded_value, f"Class_{rounded_value}")
    return mapped_class_name
    
def fetch_hydroatlas_attributes(latitude: float, longitude: float) -> Dict[str, float]:
    hydroatlas_gdf = _get_loaded_static_layer("hydroatlas")
    default_attributes = {"basin_upstream_area_km2": 0.0, "basin_population": 0.0, "basin_agriculture_pct": 0.0, "basin_slope_deg": 0.0}
    
    if hydroatlas_gdf is None or hydroatlas_gdf.empty: return default_attributes
        
    point_gdf = gpd.GeoDataFrame([{'geometry': Point(longitude, latitude)}], crs="EPSG:4326")
    if point_gdf.crs != hydroatlas_gdf.crs: point_gdf = point_gdf.to_crs(hydroatlas_gdf.crs)
        
    joined_data = gpd.sjoin(point_gdf, hydroatlas_gdf, how="left", predicate="intersects")
    
    if not joined_data.empty:
        default_attributes["basin_upstream_area_km2"] = float(joined_data.iloc[0].get("UP_AREA", 0.0))
        default_attributes["basin_population"] = float(joined_data.iloc[0].get("POP", 0.0))
        default_attributes["basin_agriculture_pct"] = float(joined_data.iloc[0].get("AG", 0.0))
        default_attributes["basin_slope_deg"] = float(joined_data.iloc[0].get("SLOPE", 0.0))
    return default_attributes

def fetch_riveratlas_attributes(latitude: float, longitude: float) -> Dict[str, float]:
    river_gdf = _get_loaded_static_layer("riveratlas")
    default_attributes = {"river_avg_discharge_cms": 0.0, "river_order": 0, "river_width_m": 0.0, "river_upstream_area_km2": 0.0}
    
    if river_gdf is None or river_gdf.empty: return default_attributes 
    point_gdf = gpd.GeoDataFrame([{'geometry': Point(longitude, latitude)}], crs="EPSG:4326")
    
    if point_gdf.crs != river_gdf.crs: point_gdf = point_gdf.to_crs(river_gdf.crs)
    joined_data = gpd.sjoin_nearest(point_gdf, river_gdf, how="left", distance_col="distance_to_river")
    
    if not joined_data.empty:
        default_attributes["river_avg_discharge_cms"] = float(joined_data.iloc[0].get("DIS_AV_CMS", 0.0))
        default_attributes["river_order"] = int(joined_data.iloc[0].get("RIV_ORD", 0))
        default_attributes["river_width_m"] = float(joined_data.iloc[0].get("RIV_WIDTH", 0.0))
        default_attributes["river_upstream_area_km2"] = float(joined_data.iloc[0].get("UP_AREA", 0.0))
    return default_attributes

def enrich_dataset_with_external_api(dataframe: pd.DataFrame, latitude_col: str, longitude_col: str, date_col: str) -> pd.DataFrame:
    """
    DEPRECATED: This monolithic function is replaced by the per-API approach.
    Use process_in_chunks() together with extract_elevation_features(),
    extract_weather_features(), extract_soilgrids_features(), and
    extract_osm_features() instead. See notebook 02-external_data_extraction.ipynb.
    """
    import warnings
    warnings.warn(
        "enrich_dataset_with_external_api() is deprecated. "
        "Use per-API extraction functions with process_in_chunks() instead. "
        "See notebook 02-external_data_extraction.ipynb.",
        DeprecationWarning,
        stacklevel=2,
    )
    return dataframe


def process_in_chunks(df: pd.DataFrame, process_func, chunk_size: int = 50, desc: str = "Processing") -> pd.DataFrame:
    """Process dataframe in chunks with progress tracking to avoid memory issues."""
    results = []
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunk_result = process_func(chunk)
        results.append(chunk_result)
        print(f"  [{desc}] Chunk {i // chunk_size + 1}/{total_chunks} done ({min(i + chunk_size, len(df))}/{len(df)} rows)")
    return pd.concat(results, ignore_index=True)


def extract_elevation_features(df: pd.DataFrame, lat_col: str = 'Latitude', lon_col: str = 'Longitude') -> pd.DataFrame:
    """Extract elevation for each unique coordinate. Returns df with keys + elevation_meters."""
    coords = df[[lat_col, lon_col]].drop_duplicates().copy()
    coords['elevation_meters'] = coords.apply(
        lambda r: fetch_elevation(r[lat_col], r[lon_col]), axis=1
    )
    return df[[lat_col, lon_col]].merge(coords, on=[lat_col, lon_col], how='left')


def extract_weather_features(df: pd.DataFrame, lat_col: str = 'Latitude', lon_col: str = 'Longitude', date_col: str = 'Sample_Date') -> pd.DataFrame:
    """Extract weather features for each unique coordinate+date. Returns df with keys + total_precipitation, average_wind_speed."""
    result = df[[lat_col, lon_col, date_col]].copy()
    weather_data = result.apply(
        lambda r: fetch_historical_weather(r[lat_col], r[lon_col], str(r[date_col])), axis=1
    )
    return pd.concat([result, pd.DataFrame(weather_data.tolist(), index=result.index)], axis=1)


def extract_soilgrids_features(df: pd.DataFrame, lat_col: str = 'Latitude', lon_col: str = 'Longitude') -> pd.DataFrame:
    """Extract soil properties for each unique coordinate. Returns df with keys + soil_phh2o_mean_0_5cm, soil_clay_mean_0_5cm, etc."""
    coords = df[[lat_col, lon_col]].drop_duplicates().copy()
    soil_data = coords.apply(
        lambda r: fetch_soilgrids_properties(r[lat_col], r[lon_col]), axis=1
    )
    coords = pd.concat([coords, pd.DataFrame(soil_data.tolist(), index=coords.index)], axis=1)
    return df[[lat_col, lon_col]].merge(coords, on=[lat_col, lon_col], how='left')


def extract_osm_features(df: pd.DataFrame, lat_col: str = 'Latitude', lon_col: str = 'Longitude') -> pd.DataFrame:
    """Extract OSM pollution counts for each unique coordinate. Returns df with keys + mine_count, wastewater_count, farmland_count, road_count."""
    coords = df[[lat_col, lon_col]].drop_duplicates().copy()
    osm_data = coords.apply(
        lambda r: fetch_osm_pollution_counts(r[lat_col], r[lon_col], radius_m=1000), axis=1
    )
    coords = pd.concat([coords, pd.DataFrame(osm_data.tolist(), index=coords.index)], axis=1)
    return df[[lat_col, lon_col]].merge(coords, on=[lat_col, lon_col], how='left')


def save_and_upload_to_stage(df: pd.DataFrame, file_name: str, session, stage_path: str = "@EXTERNAL_DATA_STAGE") -> None:
    """Save dataframe as parquet to /tmp/ and upload to Snowflake Stage."""
    local_path = f"/tmp/{file_name}"
    df.to_parquet(local_path, index=False, engine='pyarrow', compression='snappy')
    session.file.put(f"file://{local_path}", stage_path, auto_compress=False, overwrite=True)
    print(f"✓ {file_name} uploaded to {stage_path} ({len(df)} rows, {os.path.getsize(local_path) / 1024:.1f} KB)")