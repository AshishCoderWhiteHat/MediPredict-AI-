# MediPredict AI: Dataset Card

## 1. Dataset Summary
- **Dataset Name:** Pima Indians Diabetes Database
- **Originating Institution:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Primary Public Source:** Publicly hosted clinical research repository (Plotly / UCI Machine Learning Repository)
- **Cryptographic Fingerprint (SHA-256):** `698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72`
- **Total Records:** 768 patient observations
- **Total Features:** 8 numerical predictor features + 1 binary target feature (`Outcome`)

---

## 2. Feature Schema & Clinical Definitions

| Feature Name | Type | Units | Clinical Description & Operational Range |
| :--- | :---: | :---: | :--- |
| **Pregnancies** | Integer | Count | Number of times pregnant ($0 - 17$). Nulliparous ($0$) is valid. |
| **Glucose** | Float | mg/dL | 2-hour plasma glucose in oral glucose tolerance test ($44 - 199$). $0$ is physiological anomaly. |
| **BloodPressure** | Float | mm Hg | Diastolic blood pressure ($24 - 122$). $0$ represents absent circulatory measurement. |
| **SkinThickness** | Float | mm | Triceps skinfold thickness ($7 - 99$). $0$ represents omitted caliper test. |
| **Insulin** | Float | uU/mL | 2-Hour post-load serum insulin ($14 - 846$). $0$ represents unmeasured biomarker. |
| **BMI** | Float | kg/m² | Body Mass Index ($18.2 - 67.1$). $0$ represents unrecorded height/weight. |
| **DiabetesPedigreeFunction** | Float | Score | Genetic score summarizing family history and hereditary risk ($0.078 - 2.42$). |
| **Age** | Integer | Years | Patient age ($21 - 81$). All cohort subjects are adult females. |
| **Outcome** (Target) | Integer | Binary | $0 =$ Non-diabetic ($500$ cases, $65.1\%$), $1 =$ Diabetic ($268$ cases, $34.9\%$). |

---

## 3. Data Cleaning & Preprocessing Decisions

### Physiological Zero Handling
In human physiology, plasma glucose, blood pressure, insulin, skin thickness, and BMI cannot be zero in living individuals.
- **Glucose zeroes:** 5 instances (0.65%)
- **BloodPressure zeroes:** 35 instances (4.56%)
- **SkinThickness zeroes:** 227 instances (29.56%)
- **Insulin zeroes:** 374 instances (48.70%)
- **BMI zeroes:** 11 instances (1.43%)

**Methodological Decision:**
Deleting records with zero values would discard nearly 50% of the dataset, severely degrading statistical power and introducing survival bias. MediPredict AI replaces physiological zeroes with `np.nan` and applies median imputation fitted strictly on training folds, preserving sample diversity without data leakage.

---

## 4. Known Biases and Limitations
1. **Demographic Homogeneity:** All patients in this dataset are female of Pima Native American ancestry aged 21 and older. Models trained on this dataset cannot be assumed to generalize to male patients, pediatric populations, or other ethnicities.
2. **Missing Biomarker Prevalence:** Almost half of the cohort lacks 2-hour serum insulin measurements.
3. **Temporal Invariance:** Data was collected cross-sectionally without longitudinal follow-up tracking disease progression.

---

## 5. Ethical Considerations & Permissible Use
- **Permissible Use:** Academic demonstration, educational training, algorithm benchmarking, and software engineering coursework.
- **Prohibited Use:** Real-world clinical diagnosis, direct patient triage, prescription determination, or commercial medical devices.
