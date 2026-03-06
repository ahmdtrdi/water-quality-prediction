# AI/ML Pipeline Template 

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![Code Style](https://img.shields.io/badge/code%20style-modular-black.svg)

**A Production-Ready, Modular, and Config-Driven Machine Learning Project Structure.**

This template is designed for Data Scientists and ML Engineers who want to kickstart projects with high **Software Engineering standards**. The structure strictly separates logic (`src`), configuration (`config`), and execution (`entrypoint`), making it fully ready for **CI/CD** pipelines and **Docker** containerization.

---

## Key Features

* **Modular Architecture**: Logic is strictly separated (Preprocessing, Training, Inference) within the `src/` directory, installed as a local package.
* **Leakage-Proof Pipeline**: Implements a **"Split-First"** strategy, ensuring test data never leaks into training statistics (mean, std, outlier caps).
* **Config-Driven**: Modify hyperparameters, data paths, or preprocessing logic via `yaml` files without touching the Python code.
* **Reproducible**: Environment is locked and reproducible via `requirements.txt` and `Dockerfile`.
* **MLflow Integrated**: Automatic experiment tracking for parameters, metrics, and artifacts.
* **Deployment Ready**: Includes ready-to-use inference scripts that automatically load the trained model and the exact preprocessor used during training.

---

## Project Structure

```text
├── config/                          
│   ├── base.yaml                      # Main config (Target col, params, schema)
│   ├── dev.yaml                       # Override for local/laptop development
│   └── prod.yaml                      # Override for production/server
├── data/                             
│   ├── 01-raw/                        # Raw Data (Immutable)
│   ├── 02-preprocessed/               # Clean data ready for training
│   ├── 03-features/                   # Data with engineered features.
│   └── 04-predictions/                # Model output results
├── entrypoint/                        
│   ├── run_train.py                   # Training (CLI)
│   └── ...
├── models/                            # ARTIFACTS (Saved .pkl Models & Preprocessors)
├── notebooks/
│   ├── 01_EDA_Template.ipynb          # Empty template for data analysis 
│   ├── 02_Modeling_Playground.ipynb   # Experiment Model
├── src/                               
│   ├── pipelines/
│   │   ├── preprocessing.py            # Feature Eng & Math Transforms Logic
│   │   ├── training.py                 # Training & Logging Logic
│   │   └── inference.py                # Serving/Prediction Logic
│   └── utils.py
├── tests/                              # UNIT TESTS/QA
├── Dockerfile                          # Containerization Setup
├── Makefile                            # Command Shortcuts
└── requirements.txt                    # Python Dependencies
```
## Quick Start

### Installation
Ensure **Python 3.9+** is installed.

```bash
# Clone this repository (or use as a GitHub Template)
git clone [https://github.com/ahmdtrdi/AI-ML-Pipeline-Template.git](https://github.com/ahmdtrdi/AI-ML-Pipeline-Template.git)
cd ai-ml-pipeline-template

# Install dependencies & setup the 'src' package
make install

# Or manually:
# pip install -r requirements.txt && pip install -e .
```

### Configuration
Before running the pipeline, you must configure your project to match your dataset. Edit *`config/base.yaml`*:

`Target Variable`: Change target to your label column name.

`Schema`: Fill in features_numeric and features_categorical lists.

`Paths`: Adjust raw_data to point to your CSV file location.

### Training
Run the end-to-end training pipeline (Load -> Split -> Preprocess -> Train -> Save).

```bash
# Using Make (Recommended)
make train

# Or using Python directly
python entrypoint/run_train.py --config config/base.yaml
```
Artifacts (Trained Model & Preprocessor) will be saved in the *`models/`* directory.

### Tracking (MLflow)
Monitor your experiment runs, metrics, and parameters:

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

### Docker (Optional)
Run the training job inside an isolated container to ensure reproducibility.

```bash
# Build the image
docker build -t my-ml-project .

# Run the container
docker run my-ml-project
```
### Customization Guide
Adding New Features:
Edit *`src/pipelines/preprocessing.py.`* Look for the `custom_feature_engineering` function to add Pandas-based logic `(e.g., df['new_col'] = ...)`.

Changing the Model:
Edit *`src/pipelines/training.py.`* You can replace `RandomForestClassifier` with `XGBoost`, `LightGBM`, or any other Scikit-Learn compatible model.

Adding Tests:
Create new test files in the `tests/` directory and run:

```bash
#test pipelines
pytest
```

### License
Distributed under the MIT License. See `LICENSE` for more information.

### Author
Copyright © 2026 0xians
