# EY Water Quality Prediction — AI Data Challenge 2026

Predicting **3 water quality parameters** (Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus) for **unseen river locations** across South Africa.

> **Evaluation metric:** Mean R² across all 3 targets.

## 🏗️ Project Structure

```
water-quality-prediction/
├── config/
│   ├── base.yaml                  # Pipeline configuration
│   └── feature_sets.yaml          # Per-target selected features (from notebook 03)
├── data/
│   ├── 01-raw/                    # Immutable source CSVs (from EY)
│   ├── 02-processed/              # Cleaned parquet files
│   └── 03-external/               # External features (generated on Kaggle)
├── docs/
│   ├── DEVLOG-DS.md               # Experiment log (R² per experiment)
│   ├── AGENTS-DataScientist.md    # Data science methodology guidelines
│   ├── 2026_EY_AI_...             # Challenge guidance
│   └── ey_winners_approach.md     # Past winners' reference
├── notebook/
│   ├── kaggle/                    # 8 Kaggle-compatible notebooks (main pipeline)
│   ├── ey_provide/                # EY-provided benchmark notebooks (reference)
│   └── custom_snowflake_archive/  # Archived Snowflake-based notebooks
├── src/
│   ├── data_ingestion.py          # API calls, checkpoint logic
│   ├── feature.py                 # Feature engineering utilities
│   ├── spatial_cv.py              # Spatial cross-validation (LeaveStationGroupOut)
│   ├── ensemble.py                # OOF stacking ensemble framework
│   └── utils.py
└── requirements.txt
```

## 🧪 Experiment Batching Strategy

Every dataset must **earn its place** through measured R² improvement:

| Exp | Name | Datasets Added | Purpose |
|-----|------|----------------|---------|
| 0 | Baseline | Water Quality only | Floor: naïve baselines |
| 1 | EY Core | + Landsat + TerraClimate | Provided satellite features |
| 2 | External APIs | + SoilGrids + Weather + Elevation + OSM | Domain-driven features |
| 3 | Spatial Context | + HydroATLAS + RiverATLAS + SANLC + WorldPop | Catchment & land-use |
| 4 | Multi-Target | Same as Exp 3 | RegressorChain vs Separate |
| 5 | Satellite++ | + Sentinel-2 | Higher-res indices |
| 6 | Ensemble | Top from Exp 1-5 | OOF stacking |

## 🏃 How to Run

### Setup (Kaggle)
1. Upload raw CSVs (`data/01-raw/*.csv`) as a **private Kaggle Dataset** named `ey-water-quality-data`
2. Create a new Kaggle Notebook, add the dataset, and **enable Internet**
3. Run notebooks sequentially: `00` → `01` → `02` → ... → `07`

### Notebook Pipeline

| # | Notebook | Purpose | Outputs |
|---|----------|---------|---------|
| 00 | `data_preparation` | Merge raw CSVs, clean, verify | `train_base.parquet`, `val_base.parquet` |
| 01 | `external_data_extraction` | API calls + wget heavy files (100% on Kaggle) | `train_enriched.parquet`, `val_enriched.parquet` |
| 02 | `eda_and_profiling` | Target correlations, MI scores, covariate shift | EDA insights |
| 03 | `feature_engineering` | Temporal/interaction features, RFECV selection | `train_featured.parquet`, `feature_sets.yaml` |
| 04 | `modeling_baseline` | Spatial CV setup, Exp 0 + Exp 1 | Baseline R² scores |
| 05 | `modeling_advanced` | Optuna tuning, Exp 2-5, multi-target comparison | Best models per target |
| 06 | `ensemble_and_stacking` | OOF stacking, KernelRidge meta-learner | `submission.csv` |
| 07 | `explainability` | SHAP analysis, spatial maps, business insights | Visualizations |

## 🔬 Methodology

- **Cross-validation:** Leave-Station-Group-Out (10-fold spatial CV) — simulates the real task of predicting unseen locations
- **Modeling:** Separate models per target (default), with multi-target exploration in Exp 4
- **Ensemble:** Diversified base models (XGBoost, LightGBM, ExtraTrees, RF) → StandardScale OOF → KernelRidge meta-learner
- **Feature selection:** RFECV with GroupKFold per target — each target gets its own optimized feature set

## 📊 Results

See [docs/DEVLOG-DS.md](docs/DEVLOG-DS.md) for per-experiment results.

## 📚 References

- [Challenge Guidance](docs/2026_EY_AI_Data_Challenge_Participant_Guidance-context.md)
- [Winners' Approach (2025)](docs/ey_winners_approach.md)
- [Data Science Guidelines](docs/AGENTS-DataScientist.md)
