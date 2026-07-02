import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

def impute_missing_values(dataframe: pd.DataFrame, numeric_strategy: str = 'median') -> pd.DataFrame:
    """
    Imputes missing values in the dataframe.
    Applies median imputation for numeric features to avoid outlier influence.
    """
    processed_df = dataframe.copy()
    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        if processed_df[column].isnull().any():
            if numeric_strategy == 'median':
                fill_value = processed_df[column].median()
            elif numeric_strategy == 'mean':
                fill_value = processed_df[column].mean()
            else:
                fill_value = 0.0
            processed_df[column] = processed_df[column].fillna(fill_value)
            
    return processed_df

def transform_skewed_features(dataframe: pd.DataFrame, columns_to_transform: List[str]) -> pd.DataFrame:
    """
    Applies logarithmic transformation to specified highly skewed columns.
    Adds 1 to avoid log(0) mathematically.
    """
    transformed_df = dataframe.copy()
    for column in columns_to_transform:
        if column in transformed_df.columns:
            transformed_df[f"{column}_log"] = np.log1p(transformed_df[column])
            transformed_df = transformed_df.drop(columns=[column])
            
    return transformed_df

def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_columns: List[str], exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales features using StandardScaler. 
    Fits only on training data to prevent data leakage, then transforms both train and test.
    """
    if exclude_columns is None:
        exclude_columns = []
        
    columns_to_exclude = target_columns + exclude_columns
    feature_columns = [col for col in train_df.columns if col not in columns_to_exclude]
    
    scaler = StandardScaler()
    
    scaled_train_df = train_df.copy()
    scaled_test_df = test_df.copy()
    
    scaled_train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    scaled_test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    
    return scaled_train_df, scaled_test_df, scaler

def prepare_model_inputs(dataframe: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Removes restricted columns (e.g., Latitude, Longitude, Date) before modeling
    to ensure generalization and rule compliance.
    """
    clean_df = dataframe.copy()
    existing_drop_cols = [col for col in columns_to_drop if col in clean_df.columns]
    
    return clean_df.drop(columns=existing_drop_cols)


# ===========================================================================
# Temporal Feature Engineering
# ===========================================================================

def add_temporal_features(df: pd.DataFrame, date_col: str = 'Sample Date') -> pd.DataFrame:
    """
    Add cyclical temporal features from a date column.
    
    Uses sine/cosine encoding to capture the cyclical nature of months/seasons
    (e.g., December and January are adjacent, not 11 months apart).
    South Africa seasons are Southern Hemisphere (Dec=summer).
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['day_of_year'] = dt.dt.dayofyear
    df['year'] = dt.dt.year
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Southern Hemisphere seasons
    season_map = {12: 'summer', 1: 'summer', 2: 'summer',
                  3: 'autumn', 4: 'autumn', 5: 'autumn',
                  6: 'winter', 7: 'winter', 8: 'winter',
                  9: 'spring', 10: 'spring', 11: 'spring'}
    df['season'] = df['month'].map(season_map)
    
    return df


# ===========================================================================
# Domain-Driven Interaction Features
# ===========================================================================

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physically-motivated interaction features for water quality prediction.
    
    These features capture domain knowledge:
    - Runoff: precipitation × slope drives pollutant transport
    - Agricultural load: farmland % × precipitation drives DRP (phosphorus)
    - Dilution: discharge × area indicates how diluted pollutants are
    - Weathering: soil pH × elevation drives alkalinity
    """
    df = df.copy()
    
    # Runoff proxy (precipitation × slope → pollutant wash-off)
    if 'precip_sum_7d' in df.columns and 'basin_slope_deg' in df.columns:
        df['runoff_proxy'] = df['precip_sum_7d'] * df['basin_slope_deg']
    
    # Agricultural phosphorus load (farmland % × precipitation → DRP driver)
    if 'basin_agriculture_pct' in df.columns and 'precip_sum_7d' in df.columns:
        df['agri_phosphorus_load'] = df['basin_agriculture_pct'] * df['precip_sum_7d']
    
    # Dilution factor (discharge × upstream area → concentration reducer)
    if 'river_discharge_cms' in df.columns and 'basin_upstream_area_km2' in df.columns:
        df['dilution_factor'] = df['river_discharge_cms'] * df['basin_upstream_area_km2']
    
    # Soil weathering index (pH × elevation → alkalinity driver)
    if 'soil_phh2o' in df.columns and 'elevation_m' in df.columns:
        df['weathering_index'] = df['soil_phh2o'] * df['elevation_m']
    
    # Clay-organic interaction (affects nutrient binding capacity)
    if 'soil_clay' in df.columns and 'soil_ocd' in df.columns:
        df['clay_organic_interaction'] = df['soil_clay'] * df['soil_ocd']
    
    # Urbanization stress (population density × impervious surface)
    if 'population_1km' in df.columns:
        df['urban_stress'] = df['population_1km']
        if 'osm_total_1000m' in df.columns:
            df['urban_stress'] = df['urban_stress'] * (1 + df['osm_total_1000m'])
    
    return df


# ===========================================================================
# Auto Log-Transform for Skewed Features
# ===========================================================================

def auto_log_transform(df: pd.DataFrame, skew_threshold: float = 2.0, 
                       exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Automatically apply log1p transform to highly skewed non-negative features.
    
    Returns the transformed dataframe and a list of transformed column names.
    """
    df = df.copy()
    exclude_cols = exclude_cols or []
    transformed = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in exclude_cols:
            continue
        skew = df[col].skew()
        if abs(skew) > skew_threshold and df[col].min() >= 0:
            df[f'{col}_log'] = np.log1p(df[col])
            transformed.append(col)
    
    return df, transformed