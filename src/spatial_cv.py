"""
Spatial Cross-Validation utilities for water quality prediction.

The validation set uses completely different stations/rivers from training,
making this a spatial extrapolation problem. Standard random K-Fold will be
over-optimistic. These utilities implement honest spatial CV strategies.

References:
- Roberts et al. (2017) "Cross-validation strategies for data with temporal, 
  spatial, hierarchical, or phylogenetic structure" Ecography 40:913-929
- Ploton et al. (2020) "Spatial validation reveals poor predictive performance
  of large-scale ecological mapping models" Nature Communications 11:4540
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from typing import List, Tuple, Optional


class LeaveStationGroupOut(BaseCrossValidator):
    """
    Leave-Station-Group-Out cross-validator.
    
    Groups samples by station ID and holds out entire station groups per fold.
    This simulates the real prediction task where we must predict for unseen 
    locations (stations not in training data).
    
    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Stations are distributed across folds as evenly as possible.
    station_col : str, default='station_id'
        Column name containing station identifiers.
    random_state : int, default=42
        Random seed for reproducible fold assignment.
    """
    
    def __init__(self, n_splits: int = 10, station_col: str = 'station_id', 
                 random_state: int = 42):
        self.n_splits = n_splits
        self.station_col = station_col
        self.random_state = random_state
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    
    def split(self, X: pd.DataFrame, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : pd.DataFrame
            Must contain the station_col column.
        y : ignored
        groups : ignored (station_col is used directly)
        
        Yields
        ------
        train_idx, test_idx : arrays of indices
        """
        if self.station_col not in X.columns:
            raise ValueError(
                f"Column '{self.station_col}' not found in DataFrame. "
                f"Available columns: {list(X.columns)}"
            )
        
        unique_stations = X[self.station_col].unique()
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_stations)
        
        # Distribute stations across folds as evenly as possible
        station_folds = {}
        for i, station in enumerate(unique_stations):
            station_folds[station] = i % self.n_splits
        
        for fold_idx in range(self.n_splits):
            test_stations = [s for s, f in station_folds.items() if f == fold_idx]
            test_mask = X[self.station_col].isin(test_stations)
            
            train_idx = X.index[~test_mask].values
            test_idx = X.index[test_mask].values
            
            yield train_idx, test_idx
    
    def get_fold_station_mapping(self, X: pd.DataFrame) -> dict:
        """Return a dict mapping fold_idx -> list of station IDs for inspection."""
        mapping = {}
        for fold_idx, (_, test_idx) in enumerate(self.split(X)):
            stations = X.loc[test_idx, self.station_col].unique().tolist()
            mapping[fold_idx] = stations
        return mapping


class SpatialBlockCV(BaseCrossValidator):
    """
    Spatial Block Cross-Validator.
    
    Divides the study area into lat/lon grid blocks and holds out entire 
    blocks per fold. Useful as a sanity check alongside LeaveStationGroupOut.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    lat_col : str, default='Latitude'
        Column name for latitude.
    lon_col : str, default='Longitude'
        Column name for longitude.
    random_state : int, default=42
        Random seed for reproducible block assignment.
    """
    
    def __init__(self, n_splits: int = 5, lat_col: str = 'Latitude', 
                 lon_col: str = 'Longitude', random_state: int = 42):
        self.n_splits = n_splits
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.random_state = random_state
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    
    def split(self, X: pd.DataFrame, y=None, groups=None):
        """
        Generate indices by spatial blocks.
        
        Creates a grid over the lat/lon extent and assigns each grid cell
        to a fold. Samples within the same cell go to the same fold.
        """
        lats = X[self.lat_col].values
        lons = X[self.lon_col].values
        
        # Create grid — use sqrt(n_splits) blocks per dimension
        n_blocks_per_dim = max(2, int(np.ceil(np.sqrt(self.n_splits * 2))))
        
        lat_bins = np.linspace(lats.min(), lats.max() + 1e-6, n_blocks_per_dim + 1)
        lon_bins = np.linspace(lons.min(), lons.max() + 1e-6, n_blocks_per_dim + 1)
        
        lat_idx = np.digitize(lats, lat_bins) - 1
        lon_idx = np.digitize(lons, lon_bins) - 1
        
        # Assign each block to a fold
        block_ids = lat_idx * n_blocks_per_dim + lon_idx
        unique_blocks = np.unique(block_ids)
        
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_blocks)
        
        block_to_fold = {block: i % self.n_splits for i, block in enumerate(unique_blocks)}
        fold_assignments = np.array([block_to_fold[b] for b in block_ids])
        
        for fold_idx in range(self.n_splits):
            test_mask = fold_assignments == fold_idx
            train_idx = X.index[~test_mask].values
            test_idx = X.index[test_mask].values
            
            if len(test_idx) == 0:
                continue
            
            yield train_idx, test_idx


def evaluate_spatial_cv(
    model, 
    X: pd.DataFrame, 
    y: pd.Series,
    cv: BaseCrossValidator,
    metric_func=None,
    return_predictions: bool = False
) -> dict:
    """
    Evaluate a model using spatial cross-validation.
    
    Parameters
    ----------
    model : sklearn-compatible estimator
        Must have fit() and predict() methods.
    X : pd.DataFrame
        Feature matrix (may contain station_col for CV, will be excluded from fitting).
    y : pd.Series
        Target variable.
    cv : BaseCrossValidator
        Spatial CV splitter (LeaveStationGroupOut or SpatialBlockCV).
    metric_func : callable, optional
        Scoring function(y_true, y_pred) -> float. Defaults to R².
    return_predictions : bool, default=False
        If True, also return OOF predictions.
    
    Returns
    -------
    dict with keys: 'scores', 'mean_score', 'std_score', and optionally 'oof_predictions'
    """
    from sklearn.metrics import r2_score
    
    if metric_func is None:
        metric_func = r2_score
    
    scores = []
    oof_preds = np.full(len(X), np.nan) if return_predictions else None
    
    # Identify columns to exclude from features (metadata columns)
    meta_cols = ['station_id', 'Latitude', 'Longitude', 'Sample Date']
    feature_cols = [c for c in X.columns if c not in meta_cols]
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train = X.loc[train_idx, feature_cols]
        X_test = X.loc[test_idx, feature_cols]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
        
        model_clone = _clone_model(model)
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_test)
        
        score = metric_func(y_test, preds)
        scores.append(score)
        
        if return_predictions:
            oof_preds[test_idx] = preds
        
        print(f"  Fold {fold_idx + 1}: R² = {score:.4f} "
              f"(train={len(train_idx)}, test={len(test_idx)})")
    
    result = {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
    }
    
    if return_predictions:
        result['oof_predictions'] = oof_preds
    
    print(f"\n  Mean R² = {result['mean_score']:.4f} ± {result['std_score']:.4f}")
    return result


def _clone_model(model):
    """Clone a sklearn estimator with the same parameters."""
    from sklearn.base import clone
    return clone(model)
