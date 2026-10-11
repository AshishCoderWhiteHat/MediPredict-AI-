# MediPredict AI: Model Card

## 1. Model Summary
- **Model Name:** MediPredict AI - Clinical Risk Classifier
- **Production Champion:** Standardized, Class-Weighted Logistic Regression Pipeline
- **Package / Version:** `medipredict_best_model.joblib` (Version 1.0.0)
- **Frameworks:** Scikit-learn, Joblib, Python 3.14
- **License / Usage:** Educational Prototype (Apache 2.0 / MIT Academic)

---

## 2. Intended Use & Target Users
- **Primary Use:** Academic evaluation of explainable machine learning models for disease risk stratification.
- **Intended Users:** BTech Artificial Intelligence & Data Science students, academic evaluators, and clinical ML researchers.
- **Out-of-Scope / Prohibited Use:** Clinical diagnostic decision support, treatment planning, prescription validation, or automated patient discharge.

---

## 3. Training & Hyperparameter Specification

### Preprocessing Pipeline:
- Step 1: `PhysiologicalZeroToNaN(zero_columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'])`
- Step 2: `SimpleImputer(strategy='median')`
- Step 3: `StandardScaler()`

### Champion Model Hyperparameters:
```json
{
  "classifier__C": 0.5,
  "classifier__penalty": "l2",
  "classifier__solver": "liblinear",
  "classifier__class_weight": "balanced",
  "classifier__max_iter": 1000
}
```

---

## 4. Quantitative Performance Metrics (Held-Out Test Set: N = 154)

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | 73.38% | Overall correctness across test instances. |
| **Balanced Accuracy** | 72.69% | Macro-average of recall across both positive and negative classes. |
| **Sensitivity / Recall** | **70.37%** | **Proportion of actual diabetic patients correctly flagged (38 / 54).** |
| **Specificity** | 75.00% | Proportion of non-diabetic individuals correctly classified as negative (75 / 100). |
| **Precision** | 60.32% | Positive predictive value (38 true positives / 63 predicted positives). |
| **F1-Score** | 0.6496 | Harmonic balance between precision and sensitivity. |
| **ROC-AUC** | **0.8126** | Area under the receiver operating characteristic curve (CV: 0.8450 ± 0.0152). |
| **PR-AUC** | 0.6728 | Average precision under the precision-recall curve. |
| **Brier Score** | 0.1814 | Mean squared probability error (well-calibrated across deciles). |
| **Clinical Selection Score** | **0.7482** | Multi-criteria score prioritizing sensitivity in screening contexts. |

---

## 5. Explainability & Feature Contribution
Model Permutation Feature Importance on held-out test data identified the primary risk drivers:
1. **Glucose:** Mean AUC impact +0.124 (strongest single predictor)
2. **BMI:** Mean AUC impact +0.078
3. **Age:** Mean AUC impact +0.042
4. **DiabetesPedigreeFunction:** Mean AUC impact +0.031
5. **Pregnancies:** Mean AUC impact +0.022

Standardized Odds Ratios in the champion model confirm that elevated Blood Glucose ($\text{OR} = 2.45$) and BMI ($\text{OR} = 1.98$) significantly increase the estimated odds of diabetes.

---

## 6. Safety, Risks, and Limitations
- **False Negative Risk:** Model fails to identify approximately 29.6% of positive diabetic cases at default thresholds.
- **Demographic Generalizability:** Model performance reflects adult females of Pima Indian ancestry. Calibration may deteriorate when applied to broader populations.
- **Action Thresholds:** Probability cutoffs are experimental and must never replace clinical glycated hemoglobin (HbA1c) diagnostic lab assays.
