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

def _execute_api_get_request(url: str, query_parameters: Dict[str, Any], timeout_seconds: int = 10) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, params=query_parameters, timeout=timeout_seconds)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        print(f"API Request failed for URL {url}: {error}")
        return None

@lru_cache(maxsize=2048)
def fetch_elevation(latitude: float, longitude: float) -> float:
    """
    Fetches elevation data (DEM) for a specific coordinate.
    """
    base_url = "https://api.opentopodata.org/v1/aster30m"
    parameters = {
        "locations": f"{latitude},{longitude}"
    }
    
    data = _execute_api_get_request(base_url, parameters)
    
    if data and "results" in data and len(data["results"]) > 0:
        elevation = data["results"][0].get("elevation")
        return float(elevation) if elevation is not None else 0.0
    
    return 0.0

@lru_cache(maxsize=2048)
def fetch_historical_weather(latitude: float, longitude: float, target_date_str: str, time_lag_days: int = 7) -> Dict[str, float]:
    """
    Fetches historical weather data (precipitation, wind speed) for a specific coordinate and time window.
    target_date_str must be in 'YYYY-MM-DD' format.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    target_date = pd.to_datetime(target_date_str)
    
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
    
    weather_metrics = {
        "total_precipitation": 0.0,
        "average_wind_speed": 0.0
    }
    
    if data and "daily" in data:
        precip_list = data["daily"].get("precipitation_sum", [])
        wind_list = data["daily"].get("windspeed_10m_max", [])
        
        valid_precip = [p for p in precip_list if p is not None]
        valid_wind = [w for w in wind_list if w is not None]
        
        if valid_precip:
            weather_metrics["total_precipitation"] = sum(valid_precip)
        if valid_wind:
            weather_metrics["average_wind_speed"] = sum(valid_wind) / len(valid_wind)
            
    return weather_metrics


@lru_cache(maxsize=2048)
def fetch_soilgrids_properties(latitude: float, longitude: float) -> Dict[str, float]:
    """Getting pH, clay, sand, silt, organic carbon, CEC di depth 0-5cm (mean)"""
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    properties = ["phh2o", "clay", "sand", "silt", "organic_carbon", "cec"]
    results = {}
    
    for prop in properties:
        params = {
            "lat": latitude,
            "lon": longitude,
            "property": prop,
            "depth": "0-5cm",
            "value": "mean"
        }
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
    """Hitung jumlah titik polusi di radius tertentu (mirip buffer 1km seperti pemenang 2025)"""
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
            if "tags" in el and el["tags"].get("man_made") == "mine":
                counts["mine_count"] += 1
            elif "tags" in el and el["tags"].get("man_made") == "wastewater_plant":
                counts["wastewater_count"] += 1
            elif "tags" in el and el["tags"].get("landuse") == "farmland":
                counts["farmland_count"] += 1
            elif "tags" in el and "highway" in el["tags"]:
                counts["road_count"] += 1
    return counts

@lru_cache(maxsize=128)
def _get_loaded_static_layer(layer_identifier: str):
    """
    Ensures that heavy geospatial files are loaded into memory only once.
    Requires files to be present in the /tmp/ directory.
    """
    base_directory = "/tmp/"
    try:
        if layer_identifier == "worldpop":
            return rasterio.open(f"{base_directory}zaf_pop_2025_CN_100m_R2025A_v1.tif")
        elif layer_identifier == "sanlc":
            return rasterio.open(f"{base_directory}SA_NLC_2022_ALBERS.tif")
        elif layer_identifier == "hydroatlas":
            return gpd.read_file(f"{base_directory}BasinATLAS_v1.gdb")
    except Exception as initialization_error:
        print(f"Failed to load layer {layer_identifier}: {initialization_error}")
        return None

def fetch_static_raster_value(latitude: float, longitude: float, layer_identifier: str, buffer_radius_meters: int = 1000) -> float:
    point_geometry = Point(longitude, latitude)
    degree_buffer = buffer_radius_meters / 111320.0
    buffered_area = point_geometry.buffer(degree_buffer)
    
    raster_source = _get_loaded_static_layer(layer_identifier)
    if raster_source is None:
        return 0.0
    
    try:
        masked_image, _ = mask(raster_source, [buffered_area], crop=True, nodata=0)
        valid_pixels = masked_image[0][masked_image[0] > 0]
        return float(valid_pixels.mean()) if valid_pixels.size > 0 else 0.0
    except ValueError:
        return 0.0
    
def fetch_hydroatlas_attributes(latitude: float, longitude: float) -> Dict[str, float]:
    hydroatlas_gdf = _get_loaded_static_layer("hydroatlas")
    default_attributes = {
        "basin_upstream_area_km2": 0.0,
        "basin_population": 0.0,
        "basin_agriculture_pct": 0.0,
        "basin_slope_deg": 0.0
    }
    
    if hydroatlas_gdf is None or hydroatlas_gdf.empty:
        return default_attributes
        
    point_gdf = gpd.GeoDataFrame([{'geometry': Point(longitude, latitude)}], crs="EPSG:4326")
    
    if point_gdf.crs != hydroatlas_gdf.crs:
        point_gdf = point_gdf.to_crs(hydroatlas_gdf.crs)
        
    joined_data = gpd.sjoin(point_gdf, hydroatlas_gdf, how="left", predicate="intersects")
    
    if not joined_data.empty:
        default_attributes["basin_upstream_area_km2"] = float(joined_data.iloc[0].get("UP_AREA", 0.0))
        default_attributes["basin_population"] = float(joined_data.iloc[0].get("POP", 0.0))
        default_attributes["basin_agriculture_pct"] = float(joined_data.iloc[0].get("AG", 0.0))
        default_attributes["basin_slope_deg"] = float(joined_data.iloc[0].get("SLOPE", 0.0))
        
    return default_attributes

def enrich_dataset_with_external_api(dataframe: pd.DataFrame, latitude_col: str, longitude_col: str, date_col: str) -> pd.DataFrame:
    """
    Main function to orchestrate the enrichment of the base dataframe with external API data.
    """
    enriched_df = dataframe.copy()

    enriched_df['elevation_meters'] = enriched_df.apply(
        lambda row: fetch_elevation(row[latitude_col], row[longitude_col]), axis=1)
    weather_data = enriched_df.apply(
        lambda row: fetch_historical_weather(row[latitude_col], row[longitude_col], str(row[date_col].date())), axis=1)
    weather_df = pd.DataFrame(weather_data.tolist(), index=enriched_df.index)
    enriched_df = pd.concat([enriched_df, weather_df], axis=1)
    
    soil_data = enriched_df.apply(
        lambda row: fetch_soilgrids_properties(row[latitude_col], row[longitude_col]), axis=1)
    soil_df = pd.DataFrame(soil_data.tolist(), index=enriched_df.index)
    enriched_df = pd.concat([enriched_df, soil_df], axis=1)
    
    osm_data = enriched_df.apply(
        lambda row: fetch_osm_pollution_counts(row[latitude_col], row[longitude_col], radius_m=1000), axis=1)
    osm_df = pd.DataFrame(osm_data.tolist(), index=enriched_df.index)
    enriched_df = pd.concat([enriched_df, osm_df], axis=1)
    
    enriched_df['worldpop_density_1km'] = enriched_df.apply(
        lambda row: fetch_static_raster_value(row[latitude_col], row[longitude_col], "worldpop", 1000), axis=1)
    
    enriched_df['sanlc_landcover_mode_1km'] = enriched_df.apply(  # mode = kelas terbanyak
        lambda row: fetch_static_raster_value(row[latitude_col], row[longitude_col], "sanlc", 1000), axis=1)
    
    hydro_data = enriched_df.apply(
        lambda row: fetch_hydroatlas_attributes(row[latitude_col], row[longitude_col]), axis=1)
    hydro_df = pd.DataFrame(hydro_data.tolist(), index=enriched_df.index)
    enriched_df = pd.concat([enriched_df, hydro_df], axis=1)
    
    return enriched_df