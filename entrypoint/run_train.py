import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_config, get_logger
from src.pipelines.preprocessing import Preprocessing
from src.pipelines.training import Trainer

log = get_logger("RUN_TRAIN")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml", help="Path config")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    log.info(f"Loading data from {cfg['paths']['raw_data']}")
    try:
        df = pd.read_csv(cfg['paths']['raw_data'])
    except FileNotFoundError:
        log.error("File data tidak ditemukan.")
        return
    
    target_col = cfg['schema']['target']
    X = df.drop(columns=[target_col])
    y = df[target_col]

    log.info("Splitting Data (Train/Test)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, 
        test_size=cfg['train']['test_size'], 
        random_state=cfg['train']['random_state']
    )

    pipeline = Preprocessing(cfg)
    
    log.info("Preprocessing Train Data...")
    X_train_processed = pipeline.fit_transform(X_train_raw)
    
    log.info("Preprocessing Test Data...")
    X_test_processed = pipeline.transform(X_test_raw)

    pipeline.save(f"{cfg['paths']['artifacts']}/preprocessor.pkl")

    trainer = Trainer(cfg)
    trainer.train(X_train_processed, y_train, X_test_processed, y_test)

if __name__ == "__main__":
    main()