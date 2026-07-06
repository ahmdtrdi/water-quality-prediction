# Data Science Experiment Log
# Project: EY Water Quality Prediction 2026
# Methodology: Experiment-batched, spatial CV, per-target optimization

---

## Experiment Registry

| Exp | Name | Datasets | Status | Mean R² (CV) | Notes |
|-----|------|----------|--------|--------------|-------|
| 0 | Baseline | Water Quality only | Not started | - | Naïve & linear baselines |
| 1 | EY Core | + Landsat + TerraClimate | Not started | - | Provided satellite features |
| 2 | External APIs | + SoilGrids + Weather + Elevation + OSM | Not started | - | Domain-driven external |
| 3 | Spatial Context | + HydroATLAS + RiverATLAS + SANLC + WorldPop | Not started | - | Catchment/land-use |
| 4 | Multi-Target | Same as Exp 3 | Not started | - | RegressorChain vs Separate |
| 5 | Satellite++ | + Sentinel-2 | Not started | - | Higher-res indices |
| 6 | Ensemble | Top from Exp 1-5 | Not started | - | OOF stacking |

---

## Log Entries

### 2026-07-02: Modularization of Kaggle Pipeline for Background Execution
- **Task:** Split raw data extraction and heavy downloads into individual notebooks.
- **Added Notebooks:**
  - [01f_extract_hydroatlas.ipynb](file:///Users/tri/Documents/code/water-quality-prediction/notebook/kaggle/01f_extract_hydroatlas.ipynb): downloads BasinATLAS, intersects stations, maps UP_AREA, POP, AG, SLOPE.
  - [01g_extract_riveratlas.ipynb](file:///Users/tri/Documents/code/water-quality-prediction/notebook/kaggle/01g_extract_riveratlas.ipynb): downloads RiverATLAS, performs metric `sjoin_nearest` for discharge, riv order, river width, distance.
  - [01h_extract_sanlc.ipynb](file:///Users/tri/Documents/code/water-quality-prediction/notebook/kaggle/01h_extract_sanlc.ipynb): processes manual SANLC 2020 & 2022 land cover rasters.
  - [01i_extract_worldpop.ipynb](file:///Users/tri/Documents/code/water-quality-prediction/notebook/kaggle/01i_extract_worldpop.ipynb): downloads and masks global 100m population density raster.
  - [01j_extract_sentinel.ipynb](file:///Users/tri/Documents/code/water-quality-prediction/notebook/kaggle/01j_extract_sentinel.ipynb): queries Planetary Computer STAC for 2020 climatology to extract high-res (10m) bands and calculate NDWI, EVI, and Albedo proxies per station.
- **Updated Notebooks:**
  - [01e_merge_external.ipynb](file:///Users/tri/Documents/code/water-quality-prediction/notebook/kaggle/01e_merge_external.ipynb): integrated all 9 datasets (including Sentinel-2) and added `find_file` auto-path logic for Kaggle inputs. The merge flow is structured batch-by-batch, outputting intermediate datasets (`train_enriched_exp2.parquet` / `val_enriched_exp2.parquet` for Batch 2) and final datasets (`train_enriched.parquet` / `val_enriched.parquet` for Batch 3 & 5) to align with the experiment plan.
- **Reasoning:** Modular notebooks allow running heavy GIS extractions in Kaggle background sessions in parallel, preventing session timeout and avoiding exceeding the 20GB disk limit (by deleting heavy files after station-level query). Gradual batch merging ensures data reproducibility matching the exact experiment configurations. Sentinel-2 is processed using a climatological approach (median over year 2020) to solve historical data limitations (since Sentinel-2 was launched after the water quality sampling dates).

### 2026-07-03: OSM API Optimization and Path Tuning
- **Task:** Fix rate-limiting (null/zero counts) in OSM queries and ensure flexible weather data path ingestion.
- **Updated Notebooks:**
  - [01d_extract_osm.ipynb](file:///Users/tri/Documents/code/water-quality-prediction/notebook/kaggle/01d_extract_osm.ipynb): Bundled all 12 queries (4 categories x 3 radii) into a single batch Overpass query per station. This reduced total API calls from 2,040 to 170. Added **Overpass mirror endpoint rotation failover** (rotates between main DE, lz4, z, Kumi, Swiss mirrors on retries) to prevent HTTP errors and rate-limiting from a single server.
  - [01e_merge_external.ipynb](file:///Users/tri/Documents/code/water-quality-prediction/notebook/kaggle/01e_merge_external.ipynb): Confirmed dynamic `find_file` logic searches all `/kaggle/input/` directories. This allows the user to upload the `weather.parquet` file manually as a dataset to bypass Kaggle's interactive session output limits without modifying the merge notebook.
- **Reasoning:** Batching OSM requests resolves the public API rate-limit constraint and guarantees correct count features. Endpoint rotation ensures resilience against temporary server outages or specific mirror blocks. Using try-except blocks with `find_file` in the merge notebook allows running the merge pipeline gracefully even when some future datasets (like Sentinel-2) are not yet generated or are uploaded manually.

