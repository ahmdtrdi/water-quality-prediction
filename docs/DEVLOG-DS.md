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

_No experiments logged yet. Entries will be appended as experiments are run._
