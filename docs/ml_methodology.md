# MediPredict AI: Machine Learning Methodology Guide

## Module Overview & Lifecycle Mapping (M1 - M6)

MediPredict AI follows a rigorous machine learning engineering lifecycle designed for healthcare classification systems.

---

## M1: Complete Machine Learning Lifecycle
1. **Raw Ingestion**: Loads the verified Pima Indians Diabetes Database and generates a cryptographic SHA-256 fingerprint (`698c203a...`) for provenance tracking.
2. **Data Audit**: Audits data quality without deleting valid patient records. Identifies 5 physiological zero markers (Glucose: 5, BloodPressure: 35, SkinThickness: 227, Insulin: 374, BMI: 11).
3. **Leakage Prevention**: Stratified 80/20 train/test splitting performed before any imputation or scaling.
4. **Pipeline Encapsulation**: Uses Scikit-learn `Pipeline([('zero_to_nan', ...), ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('classifier', ...)])`.
5. **Training-Serving Consistency**: The fitted pipeline is serialized as a unit into `models/medipredict_best_model.joblib`. Inference consumes this exact pipeline.

---

## M2: Linear Models & Mathematical Intuition

### Logistic Regression
Logistic Regression models the probability of binary outcome $Y \in \{0, 1\}$ using the sigmoid link function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$.

### Cross-Entropy (Log-Loss) Cost Function
$$J(\beta) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

### Regularization
- **L2 Regularization (Ridge)**: Adds penalty $\lambda \sum_{j=1}^p \beta_j^2$. Shrinks weights toward zero, mitigates multicollinearity between correlated biomarkers (e.g., Glucose and Insulin).
- **L1 Regularization (Lasso)**: Adds penalty $\lambda \sum_{j=1}^p |\beta_j|$. Drives non-essential coefficients to exact zero, performing automatic feature selection.

### Coefficient & Odds Ratio Interpretation
In standardized Logistic Regression, the exponentiated coefficient $\text{OR} = e^{\beta_j}$ represents the multiplicative change in the odds of diabetes per one standard deviation increase in biomarker $x_j$, holding all other biomarkers constant.

---

## M3: Tree-Based Models & Feature Importance

### Algorithms Evaluated
1. **Random Forest**: Ensemble of de-correlated decision trees trained via bootstrap aggregating (bagging) with random feature subspace sampling at each split ($m = \sqrt{p}$).
2. **Gradient Boosting**: Sequential ensemble that fits each new shallow decision tree to the negative gradient (pseudo-residuals) of the loss function:
   $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$
3. **XGBoost**: Extreme Gradient Boosting with second-order Taylor expansion of loss and tree complexity regularization ($\gamma T + \frac{1}{2} \lambda \sum w_j^2$).
4. **Support Vector Machine (SVM)**: Learns a maximum-margin decision boundary using scaled input features; an RBF or linear kernel is selected during cross-validated tuning.

### Permutation Feature Importance vs Impurity-Based Importance
Impurity-based (Mean Decrease in Impurity / MDI) importances are biased toward continuous features with high cardinality. MediPredict AI evaluates **Permutation Feature Importance** on held-out test data:
$$\text{Importance}(f) = \text{Score}_{\text{baseline}} - \text{Score}_{\text{permuted}(f)}$$
Biomarkers with highest permutation importance: **Glucose**, **BMI**, and **Age**.

---

## M4: Unsupervised Learning & Dimensionality Reduction

### K-Means Clustering
Partitions patient feature vectors into $k$ clusters by minimizing within-cluster sum of squares (inertia):
$$J = \sum_{j=1}^k \sum_{x_i \in S_j} ||x_i - \mu_j||^2$$
Evaluated across $k \in [2, 8]$ using **Elbow Analysis** and **Silhouette Scores**:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

### Principal Component Analysis (PCA)
Projects 8-dimensional standardized biomarker space onto 2 orthogonal axes maximizing variance. Demonstrates patient distribution clustering without using target labels.

---

## M5: Evaluation & Model Selection

### Stratified 5-Fold Cross-Validation
Ensures all cross-validation folds preserve the 65.1% negative / 34.9% positive class distribution.

### Multi-Criteria Clinical Model Selection Formula
In disease screening, false negatives carry a severe clinical penalty (missed diabetic patients). The winning model is chosen based on:
$$\text{Score} = (0.35 \times \text{ROC-AUC}) + (0.30 \times \text{Recall}) + (0.20 \times \text{F1}) + (0.15 \times (1 - \text{Brier Score}))$$

### Actual Held-Out Benchmark Results
| Model Architecture | Accuracy | Recall (Sensitivity) | Specificity | F1-Score | Test ROC-AUC | Brier Score | Clinical Selection Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Champion) | **0.7338** | **0.7037** | **0.7500** | **0.6496** | **0.8126** | **0.1814** | **0.7482** |
| SVM | 0.7013 | 0.4815 | 0.8200 | 0.5306 | 0.8137 | 0.1736 | 0.6593 |
| XGBoost | 0.7468 | 0.5741 | 0.8400 | 0.6139 | 0.8222 | 0.1636 | 0.7082 |
| Gradient Boosting | 0.7338 | 0.5370 | 0.8400 | 0.5859 | 0.8167 | 0.1691 | 0.6888 |
| Random Forest | 0.7273 | 0.5000 | 0.8500 | 0.5625 | 0.8043 | 0.1713 | 0.6683 |

**Why did Logistic Regression win?**
Because with `class_weight='balanced'`, Logistic Regression prioritized Sensitivity (Recall = 0.7037 vs Random Forest's 0.5000), identifying 70.4% of actual diabetic cases while maintaining strong discriminative ability (ROC-AUC = 0.8126 on test, 0.8450 in CV).

---

## M6: ML Engineering & Operational Monitoring
1. **FastAPI Microservice**: Asynchronous REST API with Pydantic validation, health checks, and batch triage.
2. **Streamlit UI**: Healthcare analytics clinical dashboard with interactive Plotly visualizer.
3. **Statistical Data Drift Monitoring**:
   - Two-sample Kolmogorov-Smirnov test ($D = \sup_x |F_1(x) - F_2(x)|$, $\alpha = 0.05$).
   - Population Stability Index (PSI) with 10 quantile bins.
4. **Privacy-Preserving Telemetry**: Logs latency and aggregate predictions without storing sensitive personal health identifiers.
