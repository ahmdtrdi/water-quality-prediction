import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from src.utils import get_logger

log = get_logger("PREPROCESSING")

class Preprocessing:
    def __init__(self, config):
        self.config = config
        self.pp_conf = config['pipeline']['preprocessing']
        self.math_pipeline = None
        
    def custom_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Logic Bisnis (Row-wise operation)"""
        if not self.config['pipeline']['feature_engineering']['enable']:
            return df
            
        df = df.copy()
        log.info("Running Custom Feature Engineering...")

        # --- [VANILLA TEMPLATE] ---
        # EXP: df['new_col'] = df['col_a'] + df['col_b']
        pass 
        # ---------------------------------------------
        
        drop_cols = self.config['schema'].get('drop_cols', [])
        df = df.drop(columns=drop_cols, errors='ignore')
        
        return df

    def _build_math_pipeline(self):
        num_cols = self.config['schema']['features_numeric']
        cat_cols = self.config['schema']['features_categorical']
        
        transformers = []
        
        # Numeric Pipeline
        if num_cols:
            num_steps = []
            if self.pp_conf['imputation']['numeric']:
                num_steps.append(('imputer', SimpleImputer(strategy=self.pp_conf['imputation']['numeric'])))
            if self.pp_conf['scaling']['enable']:
                scaler = StandardScaler() if self.pp_conf['scaling']['method'] == 'standard' else MinMaxScaler()
                num_steps.append(('scaler', scaler))
            transformers.append(('num', Pipeline(num_steps), num_cols))
        
        # Categorical Pipeline
        if cat_cols:
            cat_steps = []
            if self.pp_conf['imputation']['categorical']:
                cat_steps.append(('imputer', SimpleImputer(strategy=self.pp_conf['imputation']['categorical'])))
            # Default OneHot
            cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))) # sparse_output untuk sklearn versi baru
            transformers.append(('cat', Pipeline(cat_steps), cat_cols))

        self.math_pipeline = ColumnTransformer(transformers=transformers, remainder='drop') 

    def fit_transform(self, df: pd.DataFrame):
        df_rich = self.custom_feature_engineering(df)

        if self.pp_conf['enable']:
            log.info("Fitting & Transforming Math Pipeline...")
            self._build_math_pipeline()
            X_processed = self.math_pipeline.fit_transform(df_rich)
            return X_processed 
        else:
            return df_rich 

    def transform(self, df: pd.DataFrame):
        df_rich = self.custom_feature_engineering(df)
        
        if self.pp_conf['enable']:
            if self.math_pipeline is None:
                raise Exception("Pipeline belum di-fit! Training dulu.")
            X_processed = self.math_pipeline.transform(df_rich)
            return X_processed
        else:
            return df_rich

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.math_pipeline, path)
        log.info(f"Preprocessor saved to {path}")