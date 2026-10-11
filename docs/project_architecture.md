# MediPredict AI: System Architecture Document

## 1. Executive Architecture Summary
MediPredict AI is designed as a modular, decoupled, production-grade Machine Learning and Healthcare Analytics System. The system uses a React frontend and FastAPI backend to bridge machine learning with a browser-based demo. It includes a dropdown for selecting among five saved classification pipelines. Training/evaluation artifacts are kept separate from inference.

```
+-----------------------------------------------------------------------------------+
|                                 USER TOUCHPOINTS                                  |
|                                                                                   |
|    +-----------------------------+           +------------------------------+     |
|    | React Web Application       |           | Third-Party Clinical Clients |     |
|    | (frontend - Port 5173)      |           | (CURL, Hospital EHR / EMR)   |     |
|    +--------------+--------------+           +--------------+---------------+     |
+-------------------|-----------------------------------------|---------------------+
                    |                                         |
                    | (Direct Python call OR REST HTTP)       | (HTTP POST /predict)
                    v                                         v
+-----------------------------------------------------------------------------------+
|                                INFERENCE LAYER                                    |
|                                                                                   |
|    +------------------------------------+    +-------------------------------+    |
|    | FastAPI Microservice               |    | Pydantic v2 Schema Validator  |    |
|    | (api.py - Port 8000)               |--->| (src/inference/schemas.py)    |    |
|    +-----------------+------------------+    +---------------+---------------+    |
|                      |                                       |                    |
|                      +-------------------+-------------------+                    |
|                                          v                                        |
|                          +-------------------------------+                        |
|                          | Prediction Engine             |                        |
|                          | (src/inference/predict.py)    |                        |
|                          +---------------+---------------+                        |
+------------------------------------------|----------------------------------------+
                                           v
+-----------------------------------------------------------------------------------+
|                            SERIALIZED MODEL ARTIFACT                              |
|                                                                                   |
|     models/medipredict_best_model.joblib (Scikit-Learn Pipeline Object)           |
|     +-----------------------------------------------------------------------+     |
|     | Step 1: PhysiologicalZeroToNaN (replaces zero in Glucose, Insulin etc)|     |
|     | Step 2: SimpleImputer (median strategy learned on training fold)      |     |
|     | Step 3: StandardScaler (zero-mean unit-variance normalization)        |     |
|     | Step 4: Classifier (Calibrated / Hyperparameter-Tuned Estimator)      |     |
|     +-----------------------------------------------------------------------+     |
+-----------------------------------------------------------------------------------+
                                           ^
                                           | Saved by
+------------------------------------------|----------------------------------------+
|                          OFFLINE TRAINING PIPELINE                                |
|                                                                                   |
|    +-----------------------------+           +------------------------------+     |
|    | Data Ingestion & Hash Audit |           | Data Validation & Quality    |     |
|    | (src/data/load_data.py)     |---------->| (src/data/validate_data.py)  |     |
|    +-----------------------------+           +--------------+---------------+     |
|                                                             v                     |
|    +-----------------------------+           +------------------------------+     |
|    | Model Tuning & Training     |<----------| Stratified Train/Test Split  |     |
|    | (src/training/tune.py)      |           | (src/data/preprocess_data.py)|     |
|    +--------------+--------------+           +------------------------------+     |
|                   v                                                               |
|    +-----------------------------+           +------------------------------+     |
|    | Comprehensive Evaluation    |---------->| Multi-Criteria Model Selector|     |
|    | (src/training/evaluate.py)  |           | (Clinical Score Opt)         |     |
|    +-----------------------------+           +------------------------------+     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Responsibilities

### 2.1 Single Source of Truth (`src/features/build_features.py`)
- Defines canonical feature order: `['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']`.
- Establishes clinical validation bounds, measurement units, and physiological constraints.
- Prevents silent schema misalignment between training, FastAPI, and Streamlit.

### 2.2 Data Ingestion & Quality Audit (`src/data/`)
- `load_data.py`: Downloads or loads raw data, generates SHA-256 fingerprint for provenance.
- `validate_data.py`: Identifies physiologically impossible zeros (e.g. Glucose = 0 mg/dL), checks duplicates, inspects target distribution.
- `preprocess_data.py`: Builds Scikit-Learn `Pipeline` with `PhysiologicalZeroToNaN`, `SimpleImputer(strategy='median')`, and `StandardScaler()`. Stratified 80/20 train/test splitting prevents data leakage.

### 2.3 Training & Evaluation Engine (`src/training/`)
- `tune.py`: Uses `GridSearchCV` with 5-Fold Stratified Cross-Validation on training data for Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and XGBoost.
- `evaluate.py`: Generates Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, Brier Score, and Permutation Importance on held-out test data.
- `calibration.py`: Fits Platt scaling / sigmoid calibration and assesses Brier reliability.
- `train.py`: Master orchestrator. Saves winning pipeline as a single serialized joblib artifact and writes versioned metadata.

### 2.4 Serving & Inference (`src/inference/` & `api.py`)
- `predict.py`: Caches serialized pipeline in memory. Computes probability, assigns risk tier (Low, Moderate, High), and provides local biomarker deviation explanations.
- `api.py`: FastAPI server exposing `/health`, `/model-info`, `/predict`, `/predict-batch`, and `/drift-report`.

### 2.5 Monitoring & Telemetry (`src/monitoring/`)
- `logging_utils.py`: Records prediction volume, latency (ms), and risk tier counts to `artifacts/inference_logs.jsonl` without storing patient PII.
- `drift.py`: Statistical covariate drift monitoring using Kolmogorov-Smirnov test and Population Stability Index (PSI).

---

## 3. Data Flow & Leakage Prevention Strategy
1. **Split-Before-Transform**: Data is split into `X_train` and `X_test` *before* computing imputation medians or standard deviation scaling parameters.
2. **Encapsulated Pipeline**: The fitted imputer and scaler are packaged inside the Scikit-learn `Pipeline`.
3. **Identical Transform in Serving**: During inference, the pipeline's `.predict()` method executes the exact learned transformations on new inputs, eliminating training-serving skew.
