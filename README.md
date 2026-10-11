# MediPredict AI: An Explainable Medical Disease Risk Prediction and Machine Learning Evaluation System

> **Primary UI:** React + Vite frontend with a FastAPI inference backend. The legacy Streamlit dashboard is still available as an alternative. The main React navigation intentionally excludes an “About Project” page.

## Quick start (Windows)

1. Install **Python 3.11+** and **Node.js 20+**.
2. Extract the project ZIP and open the `MediPredictAI` folder in VS Code.
3. Double-click `run_all.bat` (or run `run_web.bat` to launch without running tests first). The script creates the Python virtual environment, installs dependencies, runs tests, and starts FastAPI and React.
4. Open the React interface at `http://127.0.0.1:5173`. FastAPI documentation is at `http://127.0.0.1:8000/docs`.

The React interface has three sections: **Dashboard**, **Risk prediction**, and **Model performance**. The ML pipeline remains in the Python backend; the browser sends validated input to `POST /predict`.

To run manually, use two terminals from the project root:

```powershell
# Terminal 1: backend
.venv\Scripts\Activate.ps1
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: frontend
cd frontend
npm install
npm run dev
```

---

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.ai/)
[![Tests](https://img.shields.io/badge/pytest-23%20passed-brightgreen.svg)]()

> **⚠️ MEDICAL SAFETY & REGULATORY DISCLAIMER:**
> **MediPredict AI is an academic educational prototype and research system. It is NOT a certified medical diagnostic device.** Predictions and risk probabilities are statistical estimations and must **NEVER** be used for clinical decision-making, direct patient diagnosis, prescribing medication, or patient triage. Always consult a licensed healthcare practitioner for medical advice.

---

## Table of Contents
1. [Project Title](#1-project-title)
2. [Abstract](#2-abstract)
3. [Problem Statement](#3-problem-statement)
4. [Motivation](#4-motivation)
5. [Objectives](#5-objectives)
6. [Scope](#6-scope)
7. [Dataset Source](#7-dataset-source)
8. [Dataset Description](#8-dataset-description)
9. [Data Cleaning Methodology](#9-data-cleaning-methodology)
10. [Feature Engineering](#10-feature-engineering)
11. [Algorithms Implemented](#11-algorithms-implemented)
12. [Mathematical Intuition Behind Logistic Regression](#12-mathematical-intuition-behind-logistic-regression)
13. [SVM and Ensemble Model Explanation](#13-svm-and-ensemble-model-explanation)
14. [Gradient Boosting Explanation](#14-gradient-boosting-explanation)
15. [Training and Validation Methodology](#15-training-and-validation-methodology)
16. [Evaluation Metrics](#16-evaluation-metrics)
17. [Model Selection Strategy](#17-model-selection-strategy)
18. [Explainability Methodology](#18-explainability-methodology)
19. [System Architecture](#19-system-architecture)
20. [API Documentation](#20-api-documentation)
21. [Installation Instructions](#21-installation-instructions)
22. [Dataset Setup Instructions](#22-dataset-setup-instructions)
23. [Training Commands](#23-training-commands)
24. [Application Startup Commands](#24-application-startup-commands)
25. [Testing Commands](#25-testing-commands)
26. [Screenshots & UI Walkthrough](#26-screenshots--ui-walkthrough)
27. [Results Table (Actual Generated Metrics)](#27-results-table-actual-generated-metrics)
28. [Limitations](#28-limitations)
29. [Ethical Considerations](#29-ethical-considerations)
30. [Future Improvements](#30-future-improvements)
31. [References](#31-references)

---

## 1. Project Title
**MediPredict AI: An Explainable Medical Disease Risk Prediction and Machine Learning Evaluation System**  
*Academic Capstone Project for Bachelor of Technology (BTech) in Artificial Intelligence and Data Science.*

---

## 2. Abstract
Type 2 diabetes represents an escalating global healthcare crisis characterized by insidious onset and severe microvascular/macrovascular sequelae. While machine learning offers immense potential for automated disease risk stratification, real-world deployment is frequently hampered by poor data quality handling, data leakage, black-box obscurity, and training-serving skew.

MediPredict AI implements a reproducible machine learning lifecycle evaluating five classification algorithms—Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and XGBoost—on the Pima Indians Diabetes Database. It includes a React frontend, FastAPI inference service, a model dropdown for single-patient demonstrations, evaluation metrics, reference-based biomarker explanations, and saved model comparison reports. The current default is Logistic Regression, while users can select any of the five trained pipelines on the Risk Prediction page. This is an educational prototype, not a diagnostic system.

---

## 3. Problem Statement
Design, engineer, evaluate, and deploy a clinically defensible machine learning system that accurately stratifies diabetes risk from routine physiological indicators while strictly mitigating data leakage, overcoming class imbalance, explaining prediction drivers, and serving predictions without training-serving skew.

---

## 4. Motivation
In clinical screening, conventional diagnostic assays such as oral glucose tolerance tests (OGTT) and glycated hemoglobin (HbA1c) are time-consuming and resource-intensive. Statistical and machine learning models can identify high-risk individuals from non-invasive or routine parameters. However, ordinary classifiers frequently fail in healthcare contexts because they optimize naive accuracy rather than sensitivity (recall), resulting in dangerous false negatives. MediPredict AI is motivated by the necessity of building clinically aligned AI models that balance sensitivity, discrimination, and interpretability.

---

## 5. Objectives
1. Implement a robust ingestion module with cryptographic SHA-256 fingerprinting.
2. Formulate a clinical data quality pipeline handling physiological zeroes without record deletion.
3. Construct encapsulated Scikit-learn Pipelines preventing train-test data contamination.
4. Tune linear and tree-based ensemble classifiers via Stratified 5-Fold Cross-Validation.
5. Formulate a multi-criteria clinical model selection objective penalizing false negatives.
6. Deliver model-agnostic explainability via Permutation Feature Importance and odds ratios.
7. Engineer dual serving layers: an asynchronous FastAPI service and an interactive Streamlit dashboard.
8. Implement operational telemetry and statistical data drift detection (KS-Test and PSI).

---

## 6. Scope
- **Target Disease:** Type 2 Diabetes Mellitus risk stratification.
- **Population Cohort:** Adult clinical cohort represented in the Pima Indians Diabetes Database.
- **Operational Horizon:** Screening and secondary risk estimation (not definitive diagnosis).
- **Architecture Scope:** Local offline training, SQLite experiment tracking, REST microservice, and web analytics dashboard.

---

## 7. Dataset Source
- **Origin:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
- **Public Repositories:** UCI Machine Learning Repository / Plotly Datasets.
- **Direct Source URL:** `https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`
- **SHA-256 Cryptographic Checksum:** `698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72`

---

## 8. Dataset Description
The dataset contains **768 patient instances** with **8 numerical continuous/discrete clinical biomarkers** and 1 binary outcome:
- `Pregnancies`: Number of times pregnant ($0 - 17$)
- `Glucose`: 2-hour plasma glucose concentration in oral glucose tolerance test ($44 - 199$ mg/dL)
- `BloodPressure`: Diastolic blood pressure ($24 - 122$ mm Hg)
- `SkinThickness`: Triceps skinfold thickness ($7 - 99$ mm)
- `Insulin`: 2-Hour post-load serum insulin ($14 - 846$ uU/mL)
- `BMI`: Body Mass Index ($18.2 - 67.1$ kg/m²)
- `DiabetesPedigreeFunction`: Genetic family history scoring function ($0.078 - 2.42$)
- `Age`: Age in years ($21 - 81$)
- `Outcome` (Target): 0 = Non-diabetic ($500$ instances, $65.1\%$), 1 = Diabetic ($268$ instances, $34.9\%$).

---

## 9. Data Cleaning Methodology
In living physiology, measurements such as Glucose, Blood Pressure, Insulin, Skin Thickness, and BMI cannot physically be zero:
- Glucose zeroes: 5 (0.65%)
- Blood Pressure zeroes: 35 (4.56%)
- Skin Thickness zeroes: 227 (29.56%)
- Insulin zeroes: 374 (48.70%)
- BMI zeroes: 11 (1.43%)

**Methodological Decision:** Rather than deleting nearly 50% of patient records, `PhysiologicalZeroToNaN` converts suspicious zeroes in these 5 columns to `np.nan`. A Scikit-learn `SimpleImputer(strategy='median')` imputes missing values based solely on medians computed from training folds, eliminating data leakage. Zero pregnancies are preserved as physiologically valid.

---

## 10. Feature Engineering
- **Schema Single Source of Truth:** `src/features/build_features.py` defines clinical bounds, units, and validation thresholds.
- **Exploratory Risk Flags:**
  - `Is_Obese`: $\text{BMI} \ge 30.0$
  - `High_Glucose`: $\text{Glucose} \ge 140.0$ mg/dL
  - `Senior_Risk_Age`: $\text{Age} \ge 35$
  - `Glucose_Insulin_Ratio`: Surrogate metabolic resistance index.

---

## 11. Algorithms Implemented
1. **Logistic Regression (L1 & L2 Regularized)**
2. **Random Forest Classifier (Bagged Ensemble)**
3. **Gradient Boosting Classifier (Sequential Ensemble)**
4. **Support Vector Machine (SVM)**
5. **XGBoost Classifier (Regularized Gradient Boosting)**

---

## 12. Mathematical Intuition Behind Logistic Regression
Logistic Regression models the posterior class probability via the sigmoid transformation:
$$\hat{p} = P(Y=1 \mid X) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
Trained by minimizing the binary cross-entropy log-loss:
$$J(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right] + \lambda ||w||_2^2$$
Coefficients quantify standardized log-odds multipliers; odds ratios $e^{w_j}$ measure proportional risk escalation.

---

## 13. SVM and Ensemble Model Explanation
- **SVM:** Finds a separating decision boundary with a maximum margin. The implementation uses scaled features and probability estimates so the selected SVM pipeline can return a positive-class probability.
- **Random Forest:** Ensemble of $B$ decorrelated decision trees using bootstrap aggregation (bagging) and random feature subspace sampling ($m = \sqrt{p}$) at each split to reduce variance without increasing bias.

---

## 14. Gradient Boosting Explanation
Gradient Boosting sequentially builds decision trees where tree $m$ fits the pseudo-residuals of the prior ensemble:
$$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}}$$
Predictions are updated with a shrinkage learning rate $\eta$:
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

---

## 15. Training and Validation Methodology
- **Train/Test Split:** Stratified 80% Train ($N=614$), 20% Held-Out Test ($N=154$), fixed random seed `42`.
- **Cross-Validation:** 5-Fold Stratified Cross-Validation on training data used for hyperparameter tuning.
- **Hyperparameter Optimization:** `GridSearchCV` scoring ROC-AUC.
- **Leakage Prevention:** Transformers are fitted strictly on training folds and evaluated on held-out test sets.

---

## 16. Evaluation Metrics
Evaluated across nine clinical metrics:
- **Accuracy:** Overall proportion of correct predictions.
- **Balanced Accuracy:** Average of recall across classes.
- **Sensitivity / Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (critical clinical screening metric).
- **Specificity:** $\frac{\text{TN}}{\text{TN} + \text{FP}}$ (true negative rate).
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (positive predictive value).
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Area under receiver operating characteristic curve.
- **PR-AUC:** Average precision under precision-recall curve.
- **Brier Score:** Mean squared probability calibration loss.

---

## 17. Model Selection Strategy
In diabetes screening, failing to flag a diabetic patient (**False Negative**) carries severe pathology, while a **False Positive** triggers a harmless laboratory test. The champion model was selected via:
$$\text{Clinical Score} = (0.35 \times \text{ROC-AUC}) + (0.30 \times \text{Recall}) + (0.20 \times \text{F1}) + (0.15 \times (1 - \text{Brier Score}))$$

---

## 18. Explainability Methodology
- **Global Permutation Importance:** Evaluated on held-out test data by shuffling feature values and measuring ROC-AUC degradation.
- **Logistic Regression Odds Ratios:** Quantifies exponential biomarker impact on odds of disease.
- **Local Biomarker Explanations:** Compares individual patient inputs against population reference medians and flagged clinical danger thresholds.

---

## 19. System Architecture
```
MediPredict AI System Flow:
[Client / UI] ---> [FastAPI REST / Streamlit] ---> [Pydantic Validation]
                                                           |
                                                           v
                                            [Serialized Pipeline (.joblib)]
                                            - Zero-To-NaN Imputation
                                            - Median Imputer
                                            - StandardScaler
                                            - Calibrated Classifier
                                                           |
                                                           v
                                            [Prediction, Risk Tier & XAI]
```

---

## 20. API Documentation
- `GET /`: Service metadata and disclaimer.
- `GET /health`: Health check, uptime, and model loading status.
- `GET /model-info`: Training timestamp, dataset SHA-256, and evaluation metrics.
- `POST /predict`: Single patient risk inference with local biomarker explanation.
- `POST /predict-batch`: Batch triage scoring up to 500 patient records.
- `GET /drift-report`: Kolmogorov-Smirnov and PSI covariate drift statistics.

---

## 21. Installation Instructions
```bash
# 1. Clone repository and navigate to root
cd d:\MediPredictAI

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 22. Dataset Setup Instructions
The dataset is downloaded automatically upon first run. To manually trigger ingestion:
```bash
python -m src.data.load_data
```

---

## 23. Training Commands
To execute the complete end-to-end training and evaluation pipeline:
```bash
python -m src.training.train
```

---

## 24. Application Startup Commands (React + FastAPI)
On Windows, the easiest option is to double-click `run_web.bat`. It creates `.venv` if needed, installs Python dependencies, installs frontend dependencies if missing, and opens the API and React UI in separate terminals.

To start manually, open two terminals from the project root:
```powershell
# Terminal 1 — FastAPI backend
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000

# Terminal 2 — React frontend
cd frontend
npm install
npm run dev
```
Open `http://127.0.0.1:5173` for the React app and `http://127.0.0.1:8000/docs` for API docs. The Risk Prediction page lets users choose Logistic Regression, Random Forest, Gradient Boosting, SVM, or XGBoost. `streamlit run app.py` remains available for the older standalone Streamlit dashboard, but it is not the primary React interface.

---

## 25. Testing Commands
To execute the automated Pytest test suite:
```bash
pytest -v
```

---

## 26. Screenshots & UI Walkthrough
The primary React dashboard (`frontend/`) provides Dashboard, Risk prediction, and Model performance views. The legacy Streamlit dashboard (`app.py`) remains available as an optional alternative with these views:
1. **Home Dashboard:** Key metric cards, dataset SHA-256 fingerprint, and pipeline DAG.
2. **Disease Risk Prediction:** Interactive sliders, probability calculation, and local biomarker deviations.
3. **Batch Predictions:** CSV file upload, automated validation, and downloadable predictions.
4. **Dataset Explorer:** Summary statistics, suspicious zero audit, and correlation heatmaps.
5. **Model Comparison:** Interactive Plotly ROC curves, PR curves, and confusion matrices.
6. **Explainable AI:** Permutation importance and logistic regression odds ratios.
7. **Unsupervised Learning:** K-Means elbow plot, silhouette scores, and 2D PCA projection.
8. **Model Monitoring:** Inference telemetry and Kolmogorov-Smirnov / PSI drift reports. The Risk Prediction page includes a dropdown for selecting one of the five trained models.

---

## 27. Results Table (Actual Generated Metrics)
*Evaluated on the held-out test set ($N=154$, 20% split) with random state 42:*

| Model Architecture | Accuracy | Balanced Acc | Precision | Recall (Sensitivity) | Specificity | F1-Score | Held-Out ROC-AUC | PR-AUC | Brier Score | CV ROC-AUC (Mean ± Std) | Clinical Selection Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Default) | **73.38%** | **72.69%** | **60.32%** | **70.37%** | **75.00%** | **0.6496** | **0.8126** | **0.6728** | **0.1814** | **0.8450 ± 0.0152** | **0.7482** |
| Random Forest | 72.73% | 67.50% | 64.29% | 50.00% | 85.00% | 0.5625 | 0.8043 | 0.6683 | 0.1713 | 0.8408 ± 0.0239 | 0.6683 |
| Gradient Boosting | 73.38% | 68.85% | 64.44% | 53.70% | 84.00% | 0.5859 | 0.8167 | 0.6861 | 0.1691 | 0.8387 ± 0.0180 | 0.6888 |
| SVM | 70.13% | 65.07% | 59.09% | 48.15% | 82.00% | 0.5306 | 0.8137 | 0.6750 | 0.1736 | 0.8447 ± 0.0162 | 0.6593 |
| XGBoost | 74.68% | 70.70% | 65.96% | 57.41% | 84.00% | 0.6139 | 0.8222 | 0.6772 | 0.1636 | 0.8400 ± 0.0187 | 0.7082 |

---

## 28. Limitations
- **Demographic Constraint:** Trained solely on female patients of Pima Indian heritage aged $\ge 21$.
- **Missing Marker Rate:** 48.7% of cohort lacked 2-hour serum insulin measurements.
- **Cross-Sectional Nature:** Lacks longitudinal follow-up data.

---

## 29. Ethical Considerations
- Zero clinical decision-making; educational prototype only.
- Strict data privacy: zero personally identifiable information (PII) stored.
- Transparent explainability ensures non-black-box operation.

---

## 30. Future Improvements
1. Incorporate large-scale multi-ethnic electronic health records (e.g., MIMIC-IV).
2. Implement Cox Proportional Hazards regression for time-to-onset survival modeling.
3. Integrate federated learning for privacy-preserving hospital multi-center training.

---

## 31. References
1. Smith, J. W., et al. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus*. IEEE CBMS.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
4. Breiman, L. (2001). *Random Forests*. Machine Learning.
5. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
