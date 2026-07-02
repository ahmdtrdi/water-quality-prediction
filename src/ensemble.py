"""
Out-of-Fold (OOF) stacking ensemble framework for water quality prediction.

Implements the ensemble pattern from the winners' approach:
1. Train diverse base models with K-fold OOF predictions
2. Stack OOF predictions using a Ridge/KernelRidge meta-learner
3. Support for per-target ensembles (separate stacks for Alk, EC, DRP)

References:
- EY 2025 Winners: 27 ETR models + KernelRidge(poly, degree=2) meta-learner
- Wolpert (1992) "Stacked generalization" Neural Networks 5:241-259
"""

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from typing import List, Optional, Dict, Tuple


class OOFStackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Out-of-Fold Stacking Ensemble.
    
    Generates OOF predictions from base models, then trains a meta-learner
    on the stacked OOF outputs to produce final predictions.
    
    Parameters
    ----------
    base_models : list of (name, estimator) tuples
        Base-level models to generate OOF predictions.
    meta_model : estimator, optional
        Meta-learner trained on stacked OOF predictions. 
        Defaults to Ridge(alpha=1.0).
    cv : cross-validator
        Must be a spatial CV splitter from spatial_cv module.
    scale_oof : bool, default=True
        Whether to StandardScale OOF predictions before meta-learning.
        (Winners used this — it stabilizes Ridge/KernelRidge.)
    """
    
    def __init__(self, base_models: List[tuple], meta_model=None, 
                 cv=None, scale_oof: bool = True):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.cv = cv
        self.scale_oof = scale_oof
        
        # Fitted state
        self._fitted_base_models = {}  # {name: [model_fold0, model_fold1, ...]}
        self._oof_scaler = None
        self._fitted_meta = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'OOFStackingEnsemble':
        """
        Fit the stacking ensemble.
        
        1. For each base model, train on K-1 folds and predict on the held-out fold
        2. Collect all OOF predictions into a matrix
        3. Train meta-learner on OOF matrix → y
        """
        if self.cv is None:
            raise ValueError("cv must be provided (use LeaveStationGroupOut)")
        
        n_samples = len(X)
        n_models = len(self.base_models)
        oof_matrix = np.zeros((n_samples, n_models))
        
        # Identify feature columns (exclude metadata)
        meta_cols = ['station_id', 'Latitude', 'Longitude', 'Sample Date', 
                     'Sample Date']
        feature_cols = [c for c in X.columns if c not in meta_cols]
        
        for model_idx, (name, model) in enumerate(self.base_models):
            print(f"  Training base model: {name}")
            self._fitted_base_models[name] = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(X)):
                X_train = X.loc[train_idx, feature_cols]
                X_test = X.loc[test_idx, feature_cols]
                y_train = y.loc[train_idx]
                
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                oof_matrix[test_idx, model_idx] = model_clone.predict(X_test)
                self._fitted_base_models[name].append(model_clone)
        
        # Scale OOF predictions
        if self.scale_oof:
            self._oof_scaler = StandardScaler()
            oof_matrix = self._oof_scaler.fit_transform(oof_matrix)
        
        # Train meta-learner
        self._fitted_meta = clone(self.meta_model)
        self._fitted_meta.fit(oof_matrix, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions by averaging base model predictions across folds,
        then applying the meta-learner.
        """
        meta_cols = ['station_id', 'Latitude', 'Longitude', 'Sample Date', 
                     'Sample Date']
        feature_cols = [c for c in X.columns if c not in meta_cols]
        
        n_models = len(self.base_models)
        test_oof = np.zeros((len(X), n_models))
        
        for model_idx, (name, _) in enumerate(self.base_models):
            fold_preds = []
            for fold_model in self._fitted_base_models[name]:
                fold_preds.append(fold_model.predict(X[feature_cols]))
            # Average predictions across folds
            test_oof[:, model_idx] = np.mean(fold_preds, axis=0)
        
        if self.scale_oof and self._oof_scaler is not None:
            test_oof = self._oof_scaler.transform(test_oof)
        
        return self._fitted_meta.predict(test_oof)
    
    def get_oof_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get per-base-model OOF R² scores for diagnostics."""
        from sklearn.metrics import r2_score
        
        meta_cols = ['station_id', 'Latitude', 'Longitude', 'Sample Date', 
                     'Sample Date']
        feature_cols = [c for c in X.columns if c not in meta_cols]
        
        scores = {}
        for model_idx, (name, _) in enumerate(self.base_models):
            oof_preds = np.zeros(len(X))
            for fold_idx, (_, test_idx) in enumerate(self.cv.split(X)):
                fold_model = self._fitted_base_models[name][fold_idx]
                oof_preds[test_idx] = fold_model.predict(X.loc[test_idx, feature_cols])
            scores[name] = r2_score(y, oof_preds)
        
        return scores


def generate_oof_predictions(
    models: List[tuple],
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    feature_cols: List[str]
) -> Tuple[np.ndarray, List[List]]:
    """
    Generate OOF prediction matrix for a list of models.
    
    Returns
    -------
    oof_matrix : np.ndarray of shape (n_samples, n_models)
    fitted_models : list of lists, fitted_models[model_idx][fold_idx]
    """
    n_samples = len(X)
    n_models = len(models)
    oof_matrix = np.zeros((n_samples, n_models))
    fitted_models = [[] for _ in range(n_models)]
    
    for model_idx, (name, model) in enumerate(models):
        print(f"  [{model_idx+1}/{n_models}] Generating OOF for: {name}")
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            m = clone(model)
            m.fit(X.loc[train_idx, feature_cols], y.iloc[train_idx])
            oof_matrix[test_idx, model_idx] = m.predict(X.loc[test_idx, feature_cols])
            fitted_models[model_idx].append(m)
    
    return oof_matrix, fitted_models


def generate_test_predictions(
    fitted_models: List[List],
    X_test: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """
    Generate test predictions by averaging across folds for each base model.
    
    Returns
    -------
    test_matrix : np.ndarray of shape (n_test_samples, n_models)
    """
    n_models = len(fitted_models)
    n_samples = len(X_test)
    test_matrix = np.zeros((n_samples, n_models))
    
    for model_idx in range(n_models):
        fold_preds = []
        for fold_model in fitted_models[model_idx]:
            fold_preds.append(fold_model.predict(X_test[feature_cols]))
        test_matrix[:, model_idx] = np.mean(fold_preds, axis=0)
    
    return test_matrix
