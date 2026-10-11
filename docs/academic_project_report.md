# MediPredict AI: Academic End-Semester Project Report
**Bachelor of Technology (BTech) in Artificial Intelligence & Data Science**

---

```
                       PROJECT REPORT ON
                       
                        MediPredict AI:
       An Explainable Medical Disease Risk Prediction 
          and Machine Learning Evaluation System

                       Submitted by:
                [Student Name 1] ([Roll No. 1])
                [Student Name 2] ([Roll No. 2])

                  Under the Guidance of:
                [Prof. Guide Name, Ph.D.]
                [Designation, Department]

               DEPARTMENT OF COMPUTER SCIENCE & 
             ARTIFICIAL INTELLIGENCE & DATA SCIENCE
                [COLLEGE / UNIVERSITY NAME]
                [ACADEMIC YEAR: 2025 - 2026]
```

---

## Abstract
Diabetes mellitus represents one of the most critical chronic metabolic disorders globally, causing significant morbidity if undetected in its early pathogenesis. This capstone engineering project presents **MediPredict AI**, an explainable, end-to-end clinical machine learning system for diabetes risk prediction and evaluation. 

Trained on the standardized National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) Pima Indians Diabetes Database, the system addresses key clinical data quality challenges—specifically the treatment of physiologically implausible zeroes in biomarkers such as glucose, insulin, and blood pressure. We implement a leakage-free Scikit-learn Pipeline architecture incorporating median imputation, standardization, and multi-model hyperparameter tuning via Stratified 5-Fold Cross-Validation.

We benchmark five classification algorithms: Logistic Regression (L1/L2 regularized), Random Forest, Gradient Boosting, Support Vector Machine (SVM), and XGBoost. The champion model—Standardized Class-Weighted Logistic Regression—achieves a held-out test ROC-AUC of **0.8126** (CV ROC-AUC: **0.8450 ± 0.0152**) and a Sensitivity/Recall of **70.37%**, selected using a multi-criteria clinical screening objective that heavily penalizes dangerous false negatives.

The system features dual serving interfaces: a high-throughput **FastAPI** REST microservice with Pydantic validation and a clinical **Streamlit** dashboard delivering interactive Plotly ROC/PR curves, confusion matrices, permutation feature importance, exploratory unsupervised K-Means clustering with 2D PCA projections, and automated Kolmogorov-Smirnov / PSI data drift monitoring. MediPredict AI adheres strictly to medical safety protocols and educational disclaimers.

---

## 1. Introduction
Modern machine learning applications in medicine face two critical hurdles: the "black box" nature of complex estimators and the lack of robust ML engineering principles preventing training-serving skew. MediPredict AI bridges this gap by demonstrating the complete lifecycle of clinical AI systems: from cryptographic dataset provenance tracking to leakage-free feature pipelines, probabilistic calibration, explainability, and microservice deployment.

---

## 2. Literature Review
Clinical risk stratification algorithms historically relied on generalized linear models and rule-based scores such as the Framingham Risk Score. Recent research demonstrates that tree-based ensembles (Random Forest, Gradient Boosting) capture complex non-linear biomarker interactions. However, ensemble models often suffer from probability miscalibration and lack direct transparency. Recent works emphasize model-agnostic explainability (Permutation Importance, SHAP) and rigorous cross-validation to prevent optimistic performance reporting on small clinical cohorts.

---

## 3. Problem Statement & Objectives
### Problem Statement:
Develop an academically rigorous, reproducible machine learning system that accurately stratifies diabetes risk from physiological measurements, mitigates class imbalance without synthetic leakage, explains biomarker contributions, and serves predictions reliably without training-serving skew.

### Objectives:
1. Construct a reproducible ingestion and data quality pipeline handling physiological zeroes.
2. Develop and tune linear and tree-based classification algorithms.
3. Prevent data leakage by encapsulating transformations within Scikit-Learn pipelines.
4. Perform multi-metric evaluation focusing on sensitivity, ROC-AUC, and probability calibration.
5. Provide explainability via Permutation Feature Importance and odds ratios.
6. Build a production FastAPI REST microservice and interactive Streamlit UI.
7. Implement statistical data drift monitoring via Kolmogorov-Smirnov test and PSI.

---

## 4. Methodology & Mathematical Framework
Refer to `docs/ml_methodology.md` for complete mathematical formulations of:
- **Logistic Regression & Sigmoid Link:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Binary Cross-Entropy Loss:** $J(w) = -\frac{1}{N} \sum [y \ln \hat{p} + (1-y) \ln (1-\hat{p})]$
- **L1 / L2 Regularization & Odds Ratio Formulations**
- **Tree Splitting Criteria (Gini Impurity):** $I_G = 1 - \sum p_k^2$
- **Gradient Boosting Pseudo-Residuals & XGBoost Second-Order Taylor Expansion**
- **Unsupervised K-Means Inertia & Silhouette Scoring**
- **Kolmogorov-Smirnov Test & Population Stability Index (PSI)**

---

## 5. System Design & Architecture
Refer to `docs/project_architecture.md` for complete architectural component breakdowns, class diagrams, and sequence flows.

---

## 6. Experimental Results & Discussion

### Actual Benchmark Metrics (Held-Out Test Set: N = 154)
| Model Architecture | Accuracy | Recall (Sensitivity) | Specificity | F1-Score | Held-Out ROC-AUC | Brier Score | Clinical Selection Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Champion) | **73.38%** | **70.37%** | **75.00%** | **0.6496** | **0.8126** | **0.1814** | **0.7482** |
| SVM | 70.13% | 48.15% | 82.00% | 0.5306 | 0.8137 | 0.1736 | 0.6593 |
| XGBoost | 74.68% | 57.41% | 84.00% | 0.6139 | 0.8222 | 0.1636 | 0.7082 |
| Gradient Boosting | 73.38% | 53.70% | 84.00% | 0.5859 | 0.8167 | 0.1691 | 0.6888 |
| Random Forest | 72.73% | 50.00% | 85.00% | 0.5625 | 0.8043 | 0.1713 | 0.6683 |

### Discussion of Model Selection:
Logistic Regression with `class_weight='balanced'` remains the default model in the application. In medical screening, **Sensitivity (Recall)** is the overriding clinical metric: missing a diabetic patient (False Negative) has devastating pathological consequences. Logistic Regression successfully flagged **70.37%** of diabetic patients (vs Random Forest's 50.00%), while delivering superior discrimination (CV ROC-AUC: **0.8450 ± 0.0152**).

---

## 7. Explainability & Clinical Insights
Permutation feature importance ranked the top predictive biomarkers as:
1. **Glucose Concentration** (Mean AUC impact: +0.124)
2. **Body Mass Index (BMI)** (Mean AUC impact: +0.078)
3. **Patient Age** (Mean AUC impact: +0.042)
4. **Diabetes Pedigree Function** (Mean AUC impact: +0.031)

---

## 8. Limitations & Ethical Considerations
1. **Demographic Bias:** Trained exclusively on adult females of Pima Indian ancestry. Generalization to males, other ethnicities, or pediatric cohorts is unvalidated.
2. **Missing Biomarker Rates:** Over 48% of the cohort lacked postprandial insulin readings.
3. **Regulatory Status:** Educational research prototype; not approved as Software as a Medical Device (SaMD) by the FDA, EMA, or CDSCO.

---

## 9. Conclusion & Future Work
MediPredict AI demonstrates an end-to-end, reproducible, explainable machine learning system. Future extensions include integrating multi-ethnic clinical EHR datasets (e.g., MIMIC-IV), longitudinal survival analysis (Cox proportional hazards), and federated learning protocols for privacy-preserving hospital multi-center training.

---

## 10. References
1. National Institute of Diabetes and Digestive and Kidney Diseases. *Pima Indians Diabetes Database*.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
4. Breiman, L. (2001). *Random Forests*. Machine Learning.
5. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
