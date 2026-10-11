# MediPredict AI: Comprehensive Viva Voce Examination Guide
### 38 Essential Questions and Rigorous Academic Answers for BTech AI & DS Students

---

### Q1: What is the fundamental difference between supervised and unsupervised machine learning?
**Answer:** Supervised learning learns a mapping function $f: X \to Y$ using labeled training pairs $(x_i, y_i)$ to minimize a supervised loss function (e.g., cross-entropy for classification or mean squared error for regression). Unsupervised learning operates solely on unlabeled inputs $\{x_i\}$, seeking underlying structural patterns, data manifolds, or groupings (e.g., K-Means clustering, PCA) without reference to target ground-truth labels.

---

### Q2: Why is diabetes risk estimation framed as a binary classification problem?
**Answer:** Because the clinical target variable ($Y$) is discrete and dichotomous: $Y = 1$ (patient diagnosed with diabetes) or $Y = 0$ (patient without diabetes). The goal of the algorithm is to estimate the posterior class probability $P(Y=1 \mid X)$ given the patient's continuous and discrete biomarkers $X$.

---

### Q3: Explain the mathematical intuition of the Sigmoid function in Logistic Regression.
**Answer:** The linear combination of features $z = w^T x + b$ maps inputs to the unbounded real number line $(-\infty, +\infty)$. The sigmoid link function $\sigma(z) = \frac{1}{1 + e^{-z}}$ maps $(-\infty, +\infty)$ monotonically into the open probability interval $(0, 1)$, satisfying the Kolmogorov probability axioms.

---

### Q4: Why can't we use Mean Squared Error (MSE) to train Logistic Regression?
**Answer:** When the sigmoid function is plugged into MSE, the resulting objective function is non-convex with respect to weights $w$, containing numerous local minima and plateaus that hinder gradient descent convergence. Cross-Entropy (Log-Loss) guarantees a strictly convex loss surface with a unique global minimum.

---

### Q5: What is the cross-entropy loss function, and how does it penalize incorrect predictions?
**Answer:** The binary cross-entropy loss is defined as:
$$J(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$
When the true label is $y=1$, as the predicted probability $\hat{p} \to 0$, $-\ln(\hat{p}) \to \infty$. This applies an asymptotically infinite logarithmic penalty to confident misclassifications.

---

### Q6: Differentiate between L1 (Lasso) and L2 (Ridge) regularization.
**Answer:**
- **L1 Regularization (Lasso):** Adds penalty $\lambda \sum |w_j|$. Due to diamond-shaped geometric constraint contours, it forces less informative weights to exact zero, performing intrinsic feature selection.
- **L2 Regularization (Ridge):** Adds penalty $\lambda \sum w_j^2$. Due to spherical constraint contours, it shrinks weights asymptotically toward zero without zeroing them, effectively mitigating multicollinearity among correlated biomarkers.

---

### Q7: Why is feature scaling essential for Logistic Regression but optional for Decision Trees?
**Answer:** Logistic Regression relies on gradient descent and weight shrinkage penalties; unscaled features with larger numerical magnitudes (e.g., Insulin: 0–800) dominate the gradient updates and regularization penalties compared to smaller features (e.g., DiabetesPedigreeFunction: 0.1–2.0). Decision Trees split features orthogonally using rank-order monotonic thresholds ($x_j \le \theta$), making them completely invariant to monotonic affine scale transformations.

---

### Q8: What does a Logistic Regression coefficient represent?
**Answer:** In standardized Logistic Regression, each coefficient $\beta_j$ represents the change in log-odds of the positive class per one standard deviation increase in feature $x_j$, holding other variables constant. The exponentiated coefficient $e^{\beta_j}$ represents the Odds Ratio (OR).

---

### Q9: Does a large model coefficient prove medical causation?
**Answer:** No. Correlation or predictive association does not establish biological causation. A high positive coefficient indicates statistical dependency in the presence of existing confounders, not that manipulating the biomarker will alter disease risk.

---

### Q10: How does a Decision Tree decide where to split a node?
**Answer:** Decision trees evaluate candidate split thresholds across features to maximize impurity reduction. For classification, CART maximizes the Gini Impurity Gain:
$$\Delta I_G = I_G(\text{parent}) - \left[ \frac{N_L}{N} I_G(\text{left}) + \frac{N_R}{N} I_G(\text{right}) \right]$$
where $I_G = 1 - \sum p_k^2$.

---

### Q11: What is the primary difference between Bagging and Boosting?
**Answer:**
- **Bagging (Bootstrap Aggregation):** Trains multiple base estimators in parallel on independent bootstrap samples of the data and averages their outputs to reduce model variance (e.g., Random Forest).
- **Boosting:** Trains base estimators sequentially in series, where each subsequent model learns to correct the pseudo-residuals or errors of the preceding ensemble to reduce model bias (e.g., Gradient Boosting, XGBoost).

---

### Q12: How does Random Forest improve upon standard Bagged Decision Trees?
**Answer:** Standard bagging can produce correlated trees if a few strong features dominate every top split. Random Forest enforces **feature subspace sampling**: at each split, only a random subset of features (typically $m = \sqrt{p}$) is considered. This de-correlates the ensemble trees, substantially reducing variance upon averaging.

---

### Q13: Explain how Gradient Boosting constructs subsequent trees.
**Answer:** Gradient Boosting initializes with a base constant prediction (log-odds of positive class). In each iteration $m$, it computes the pseudo-residuals (negative gradient of the loss function with respect to current predictions), fits a shallow decision tree to predict these residuals, and updates predictions scaled by a learning rate $\eta \in (0, 1]$:
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

---

### Q14: What algorithmic enhancements does XGBoost introduce over standard Gradient Boosting?
**Answer:**
1. Second-order Taylor series expansion (gradient $g_i$ and Hessian $h_i$) for accurate loss approximation.
2. Built-in L1 ($\alpha$) and L2 ($\lambda$) regularization on leaf weights to curb overfitting.
3. Sparsity-aware split finding to handle missing values automatically.
4. Block structure for parallelized split finding.

---

### Q15: Why are zero values in Glucose and BloodPressure treated as missing rather than valid readings?
**Answer:** In living human physiology, plasma glucose concentration and arterial diastolic blood pressure cannot physically be zero. In the Pima database, zeroes represent omitted clinical tests or data collection entry codes for unmeasured values.

---

### Q16: Why did we choose median imputation over deleting rows with missing values?
**Answer:** In the Pima dataset, 374 out of 768 rows (48.7%) have zero Insulin, and 227 (29.6%) have zero SkinThickness. Deleting rows with zero values would discard almost half the dataset, destroying statistical sample size and introducing severe selection bias. Median imputation is robust to right-skewed physiological distributions and retains patient diversity.

---

### Q17: What is data leakage, and how did your pipeline prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as validation or test data) is inadvertently used to fit preprocessing transformations or models. We prevented leakage by performing Stratified 80/20 train/test splitting *before* any imputation or scaling. Imputer medians and standard scalers were fitted strictly on `X_train` and only applied via `.transform()` to `X_test`.

---

### Q18: What is Stratified K-Fold Cross-Validation, and why is it necessary here?
**Answer:** Stratified K-Fold partitions the dataset into $K$ equal subsets while guaranteeing that each fold preserves the exact class proportion (65.1% negative, 34.9% positive). Without stratification, random folds in small imbalanced datasets could suffer from high class distribution variance, biasing evaluation scores.

---

### Q19: Explain the Confusion Matrix terms: TN, FP, FN, and TP.
**Answer:**
- **True Positive (TP):** Diabetic patient correctly predicted as diabetic.
- **True Negative (TN):** Healthy individual correctly predicted as non-diabetic.
- **False Positive (FP):** Healthy individual incorrectly flagged as diabetic (Type I error).
- **False Negative (FN):** Diabetic patient incorrectly classified as healthy (Type II error).

---

### Q20: In medical screening, why is Recall (Sensitivity) often prioritized over Precision?
**Answer:** A False Negative (FN) means an individual with diabetes is told they are healthy, delaying medical intervention and leading to severe clinical complications (neuropathy, retinopathy, nephropathy). A False Positive (FP) merely triggers a harmless confirmatory laboratory assay (e.g., HbA1c blood test). Therefore, clinical screening systems must maximize Recall to minimize dangerous False Negatives.

---

### Q21: What is the difference between Specificity and Precision?
**Answer:**
- **Specificity (True Negative Rate):** $\frac{\text{TN}}{\text{TN} + \text{FP}}$ — the proportion of healthy individuals correctly identified as negative.
- **Precision (Positive Predictive Value):** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — the proportion of model-predicted positives that are genuinely positive.

---

### Q22: What is ROC-AUC, and what does an AUC of 0.81 signify?
**Answer:** The Receiver Operating Characteristic curve plots True Positive Rate vs False Positive Rate across all classification thresholds. ROC-AUC represents the probability that the classifier ranks a randomly chosen positive patient higher than a randomly chosen negative patient. An AUC of 0.8126 demonstrates strong discriminative ability.

---

### Q23: Why can ROC-AUC be overly optimistic in imbalanced datasets, and why should we also inspect PR-AUC?
**Answer:** In imbalanced datasets with a large negative class, False Positive Rate ($\frac{\text{FP}}{\text{TN} + \text{FP}}$) changes slowly because the large denominator $\text{TN}$ buffers the metric. Precision-Recall curves focus entirely on the minority positive class without including $\text{TN}$, providing a stricter assessment of positive class discrimination.

---

### Q24: What is Probability Calibration, and how is it measured?
**Answer:** A model is well-calibrated if its predicted probability reflects real empirical event frequency: among patients assigned a 70% risk, approximately 70% should actually have the disease. It is measured via **Calibration Curves** (reliability diagrams) and the **Brier Score** (mean squared difference between predicted probabilities and binary outcomes).

---

### Q25: Why did your evaluation select Logistic Regression over Random Forest as the champion model?
**Answer:** Our model selection used a multi-criteria Clinical Selection Score prioritizing Sensitivity/Recall. With `class_weight='balanced'`, Logistic Regression achieved **Recall = 0.7037** and ROC-AUC = 0.8126, whereas default Random Forest achieved higher specificity but a poor **Recall = 0.5000** (missing half the diabetic patients).

---

### Q26: What is Permutation Feature Importance, and why is it superior to MDI in trees?
**Answer:** Permutation importance randomly shuffles the values of a single feature on held-out test data and measures the drop in the evaluation metric (ROC-AUC). It is model-agnostic and avoids the impurity-based (MDI) bias that artificially inflates continuous features with high numerical variance.

---

### Q27: How does K-Means clustering work, and how was optimal $k$ selected?
**Answer:** K-Means iteratively assigns points to the nearest centroid using Euclidean distance and recalculates centroids as the cluster mean. Optimal $k$ is evaluated via the **Elbow Method** (rate of decrease in inertia) and **Silhouette Analysis** (cohesion vs separation ratio).

---

### Q28: Can K-Means clustering be used as a clinical diagnosis tool?
**Answer:** Absolutely not. K-Means partitions data strictly by mathematical proximity in geometric feature space. Clusters may reflect age cohorts or physiological clusters, but do not correspond to verified clinical disease etiologies without empirical validation.

---

### Q29: What is Principal Component Analysis (PCA), and what is its main limitation in 2D visualization?
**Answer:** PCA is an orthogonal linear transformation that projects data onto eigenvectors of the covariance matrix corresponding to the largest eigenvalues. Its main limitation is information loss: in our 8-feature dataset, 2 principal components capture approximately 45–55% of total variance, discarding higher-dimensional variations.

---

### Q30: What is Training-Serving Skew, and how did your architecture prevent it?
**Answer:** Training-serving skew is a discrepancy between model performance in development and production caused by inconsistent preprocessing or data transformations. We eliminated skew by packaging the imputer, scaler, and classifier into a single serialized Scikit-learn `Pipeline` object (`medipredict_best_model.joblib`), which is loaded directly by both FastAPI and Streamlit.

---

### Q31: How does FastAPI ensure input validation?
**Answer:** FastAPI utilizes **Pydantic v2 schemas** (`PatientInput`). It validates data types, enforces range boundaries (e.g., $18 \le \text{Age} \le 120$, $\text{Glucose} \ge 30$), and returns structured HTTP 422 error responses if inputs are malformed or out of bounds.

---

### Q32: What is the difference between Data Drift and Concept Drift?
**Answer:**
- **Data Drift (Covariate Shift):** Change in input feature distribution $P(X)$ over time while the conditional outcome distribution $P(Y \mid X)$ remains unchanged.
- **Concept Drift:** Change in the fundamental relationship between features and outcomes $P(Y \mid X)$ over time (e.g., medical definition of diabetes changing).

---

### Q33: How does your system detect data drift?
**Answer:** The monitoring module (`src/monitoring/drift.py`) applies:
1. Two-sample **Kolmogorov-Smirnov (KS) test** to detect continuous distribution shifts with statistical significance ($p < 0.05$).
2. **Population Stability Index (PSI)** to quantify quantile bin frequency shifts ($\text{PSI} \ge 0.20$ denotes significant drift).

---

### Q34: What role does MLflow play in this project?
**Answer:** MLflow provides reproducible experiment tracking. It logs model hyperparameters (C, max_depth, n_estimators), cross-validation scores, held-out evaluation metrics, dataset SHA-256 hashes, and model artifacts to an SQLite tracking backend.

---

### Q35: What are the main ethical limitations of the Pima Indians dataset?
**Answer:** The dataset comprises exclusively adult females of Pima Indian heritage. Because of genetic predispositions and specific socioeconomic factors, models trained on this cohort cannot be assumed to generalize safely to males, pediatric patients, or individuals of different ethnic ancestries.

---

### Q36: Why did you implement both direct local inference and FastAPI inference in Streamlit?
**Answer:** To demonstrate decoupled ML systems engineering. Direct local pipeline execution allows the dashboard to function standalone without network services, while the REST API demonstrates cloud microservice deployment where hospital EHR clients interact over standard HTTP.

---

### Q37: How does MediPredict AI protect patient privacy?
**Answer:** The system does not request or store patient names, addresses, phone numbers, or government IDs. Inference telemetry records only operational metadata (prediction latency, estimated probability, risk tier) to local logs without recording raw patient clinical measurements.

---

### Q38: Why is an educational disclaimer mandatory on medical AI prototypes?
**Answer:** Regulatory bodies (FDA, EMA, CDSCO) classify diagnostic software as Software as a Medical Device (SaMD), requiring extensive clinical trials and validation. An educational prototype has not undergone clinical trials, so prominent disclaimers prevent dangerous misuse for self-diagnosis or unverified treatment decisions.
