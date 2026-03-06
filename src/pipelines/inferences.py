import joblib
import pandas as pd
import os
from src.utils import get_logger

log = get_logger("INFERENCE")

class Predictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.preprocessor_pipeline = None
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.config['paths']['artifacts'], "model.pkl")
        self.model = joblib.load(model_path)
        
        prep_path = os.path.join(self.config['paths']['artifacts'], "preprocessor.pkl")
        if os.path.exists(prep_path):
            self.preprocessor_pipeline = joblib.load(prep_path)
        else:
            log.warning("Preprocessor not found! Inference might fail if data requires scaling.")

    def predict(self, data: pd.DataFrame):
        log.info("Running inference...")
        
        if self.preprocessor_pipeline:
            data_processed = self.preprocessor_pipeline.transform(data)
        else:
            data_processed = data

        predictions = self.model.predict(data_processed)
        return predictions