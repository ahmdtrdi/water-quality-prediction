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