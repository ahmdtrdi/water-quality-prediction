import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import get_logger

log = get_logger("TRAINING")

class Trainer:
    def __init__(self, config):
        self.config = config

    def train(self, X_train, y_train, X_test, y_test):
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        with mlflow.start_run():
            params = self.config['train']['params']
            
            # --- [VANILLA] ---
            model = RandomForestClassifier(**params)
            # -------------------------------------
            
            log.info("Training Model...")
            model.fit(X_train, y_train)

            log.info("Evaluating Model...")
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            log.info(f"Accuracy: {acc}")
            print(classification_report(y_test, preds))

            # Logging
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            # Save Model Local
            model_dir = self.config['paths']['artifacts']
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model, save_path)
            log.info(f"Model saved to {save_path}")