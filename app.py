"""
MediPredict AI - Clinical Machine Learning Dashboard
Author: MediPredict AI Team
Description: Interactive Streamlit application for explainable disease risk prediction,
             model evaluation, unsupervised clustering, and operational monitoring.
"""

import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="MediPredict AI | Healthcare Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    TARGET_NAME,
    get_feature_schema
)
from src.data.load_data import load_raw_data
from src.data.validate_data import generate_data_quality_report
from src.inference.predict import (
    predict_single,
    predict_batch_records,
    load_model_metadata,
    load_all_models,
    CLINICAL_DISCLAIMER,
    POPULATION_MEDIANS
)
from src.visualization.charts import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
    plot_feature_importance_bar,
    plot_logistic_regression_coefficients,
    plot_calibration_curve_chart,
    plot_learning_curve_chart,
    plot_kmeans_elbow_silhouette,
    plot_pca_clusters,
    plot_correlation_heatmap
)
from src.features.unsupervised import run_kmeans_analysis
from src.monitoring.logging_utils import get_monitoring_summary
from src.monitoring.drift import evaluate_feature_drift

# Custom Professional Clinical UI Styling
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Clinical Header Card */
    .clinical-header {
        background: linear-gradient(135deg, #0f766e 0%, #1e3a8a 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    
    .clinical-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    
    .clinical-header p {
        color: #e2e8f0;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Disclaimer Banner */
    .disclaimer-banner {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        color: #991b1b;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-bottom: 24px;
        line-height: 1.4;
    }

    /* Clinical Metric Card */
    .metric-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 6px 0;
    }
    .metric-sub {
        color: #0d9488;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Risk Badges */
    .risk-badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-mod {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Renders prominent clinical branding and educational disclaimer."""
    st.markdown("""
    <div class="clinical-header">
        <h1>🩺 MediPredict AI</h1>
        <p>Explainable Medical Disease Risk Prediction & Machine Learning Evaluation System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="disclaimer-banner">
        <b>⚠️ MEDICAL SAFETY & REGULATORY NOTICE:</b> {CLINICAL_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def get_cached_raw_data():
    """Cached loader for raw dataset."""
    df, sha = load_raw_data()
    return df, sha


@st.cache_data
def get_cached_metadata():
    """Cached loader for model metadata."""
    try:
        return load_model_metadata()
    except Exception:
        return None


@st.cache_data
def get_cached_eval_summary():
    """Cached loader for evaluation reports."""
    path = PROJECT_ROOT / "reports" / "evaluation_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_cached_feature_importance():
    """Cached loader for feature importance."""
    path = PROJECT_ROOT / "reports" / "feature_importance.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/stethoscope.png", width=64)
st.sidebar.title("MediPredict AI")
st.sidebar.caption("BTech AI & DS Capstone System")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home Dashboard",
        "🎯 Disease Risk Prediction",
        "📂 Batch Predictions",
        "📊 Dataset Explorer",
        "⚖️ Model Comparison",
        "🔍 Explainable AI",
        "🧬 Unsupervised Learning",
        "📈 Model Monitoring"
    ]
)

st.sidebar.divider()
inference_mode = st.sidebar.selectbox(
    "Inference Engine",
    ["Direct Scikit-Learn Pipeline", "FastAPI REST Service (localhost:8000)"]
)

# Main Application Pages
render_header()

# -------------------------------------------------------------
# 1. HOME DASHBOARD
# -------------------------------------------------------------
if nav_choice == "🏠 Home Dashboard":
    st.subheader("System Overview & Operational Health")

    raw_df, data_sha = get_cached_raw_data()
    metadata = get_cached_metadata()
    eval_summary = get_cached_eval_summary()

    # Top Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Patient Records</div>
            <div class="metric-value">{len(raw_df)}</div>
            <div class="metric-sub">Pima Clinical Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Clinical Biomarkers</div>
            <div class="metric-value">{len(FEATURE_NAMES)}</div>
            <div class="metric-sub">Standardized Inputs</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        best_model = metadata.get("selected_model_name", "N/A") if metadata else "Trained"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Champion Model</div>
            <div class="metric-value" style="font-size: 1.25rem;">{best_model}</div>
            <div class="metric-sub">Clinically Optimized</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        best_roc = "N/A"
        if metadata and eval_summary:
            best_roc = eval_summary.get(best_model, {}).get("roc_auc", "N/A")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test ROC-AUC</div>
            <div class="metric-value">{best_roc}</div>
            <div class="metric-sub">Held-Out Evaluation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset & Model Summary Cards
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📋 Clinical Dataset Fingerprint")
        st.write(
            "The system is trained on the National Institute of Diabetes and Digestive and Kidney Diseases "
            "(NIDDK) Pima Indians Diabetes Database."
        )
        st.markdown(f"**Dataset SHA-256 Fingerprint:** `{data_sha[:32]}...`")
        
        # Class distribution breakdown
        outcomes = raw_df["Outcome"].value_counts()
        st.markdown(f"- **Non-Diabetic (Class 0):** {outcomes.get(0, 0)} patients ({round(outcomes.get(0, 0)/len(raw_df)*100, 1)}%)")
        st.markdown(f"- **Diabetic (Class 1):** {outcomes.get(1, 0)} patients ({round(outcomes.get(1, 0)/len(raw_df)*100, 1)}%)")
        st.info("Class imbalance is managed through stratified sampling and class-weighted regularized training.")

    with col_right:
        st.markdown("### ⚙️ Machine Learning Pipeline Architecture")
        st.write(
            "MediPredict AI implements Scikit-learn Pipelines combining physiological zero imputation, "
            "standardization, multi-algorithm hyperparameter tuning, and probability calibration."
        )
        if metadata:
            st.markdown(f"- **Training Run:** `{metadata.get('timestamp')}`")
            st.markdown(f"- **Python Version:** `{metadata.get('environment', {}).get('python_version', '').split()[0]}`")
            st.markdown(f"- **Scikit-learn:** `{metadata.get('environment', {}).get('scikit_learn_version')}`")
            st.markdown(f"- **XGBoost Enabled:** `{metadata.get('environment', {}).get('has_xgboost')}`")

    # Pipeline lifecycle flowchart
    st.markdown("---")
    st.markdown("### 🔄 End-to-End ML Lifecycle Flow")
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style="filled,rounded", color="#0f766e", fontcolor=white, fontname="Arial"];
        edge [color="#1e3a8a", arrowhead=vee];
        
        RawData [label="Raw Clinical CSV", fillcolor="#1e3a8a"];
        Quality [label="Data Quality Audit\n(Zero Detection)", fillcolor="#0f766e"];
        Split [label="Stratified 80/20\nTrain-Test Split", fillcolor="#0f766e"];
        Pipeline [label="Sklearn Pipeline\n(Imputer + Scaler)", fillcolor="#0f766e"];
        Tuning [label="5-Fold Stratified CV\nGridSearch Tuning", fillcolor="#0f766e"];
        Eval [label="Held-Out Evaluation\n& Calibration", fillcolor="#0f766e"];
        Select [label="Clinical Selection\n(Recall + ROC-AUC)", fillcolor="#0f766e"];
        Deploy [label="FastAPI & Streamlit\nZero-Skew Serving", fillcolor="#16a34a"];
        
        RawData -> Quality -> Split -> Pipeline -> Tuning -> Eval -> Select -> Deploy;
    }
    """)

# -------------------------------------------------------------
# 2. DISEASE RISK PREDICTION (SINGLE PATIENT)
# -------------------------------------------------------------
elif nav_choice == "🎯 Disease Risk Prediction":
    st.subheader("Patient Biomarker Entry & Risk Assessment")
    st.write(
        "Enter verified patient clinical measurements below. The model will calculate the estimated "
        "diabetes risk probability, assign a risk tier, and outline local feature contributions."
    )

    all_models = load_all_models()
    model_names = list(all_models.keys())
    selected_model_name = st.selectbox("Select Model Architecture", model_names, index=0)

    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.slider("Pregnancies", 0, 20, 2, help="Number of times pregnant")
            glucose = st.number_input("Plasma Glucose (mg/dL)", 40.0, 300.0, 138.0, step=1.0, help="2h oral glucose tolerance test")
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", 30.0, 140.0, 78.0, step=1.0)
            skin_thickness = st.number_input("Triceps Skinfold Thickness (mm)", 0.0, 100.0, 28.0, step=1.0, help="Enter 0 if unmeasured")

        with col2:
            insulin = st.number_input("2-Hour Serum Insulin (uU/mL)", 0.0, 900.0, 120.0, step=1.0, help="Enter 0 if unmeasured")
            bmi = st.number_input("Body Mass Index (kg/m²)", 10.0, 70.0, 32.5, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function (Genetic Score)", 0.05, 3.0, 0.52, step=0.01)
            age = st.slider("Patient Age (Years)", 18, 100, 42)

        submit_btn = st.form_submit_button("⚡ Evaluate Disease Risk", use_container_width=True)

    if submit_btn:
        patient_dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        start_time = time.time()
        try:
            result = predict_single(patient_dict, model_name_override=selected_model_name)
            latency = round((time.time() - start_time) * 1000, 2)

            st.markdown("---")
            st.markdown("### 📋 Prediction Diagnostic Summary")

            # Risk Tier Card
            prob = result["probability"]
            risk_tier = result["risk_category"]
            pred_class = result["prediction"]

            res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
            with res_col1:
                badge_class = "risk-badge-low" if risk_tier == "Low Risk" else ("risk-badge-mod" if risk_tier == "Moderate Risk" else "risk-badge-high")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Calculated Risk Tier</div>
                    <div style="margin: 12px 0;"><span class="{badge_class}" style="font-size: 1.2rem;">{risk_tier}</span></div>
                    <div class="metric-sub">Outcome Class: {pred_class}</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Estimated Risk Probability</div>
                    <div class="metric-value">{round(prob * 100, 1)}%</div>
                    <div class="metric-sub">P(Outcome = 1)</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Evaluator Engine</div>
                    <div class="metric-value" style="font-size: 1.15rem;">{selected_model_name}</div>
                    <div class="metric-sub">Inference Latency: {latency} ms</div>
                </div>
                """, unsafe_allow_html=True)

            # Local Biomarker Explanations
            st.markdown("#### 🔬 Explainable Biomarker Breakdown")
            exp_col1, exp_col2 = st.columns(2)

            explanation = result.get("explanation", {})
            elevated = explanation.get("top_elevated_risk_factors", [])
            normal = explanation.get("protective_or_normal_factors", [])

            with exp_col1:
                st.markdown("**Elevated Clinical Risk Drivers:**")
                if elevated:
                    for item in elevated:
                        st.markdown(f"- 🔴 {item}")
                else:
                    st.write("No major biomarkers severely elevated above population thresholds.")

            with exp_col2:
                st.markdown("**Biomarkers Within Normal Reference Ranges:**")
                if normal:
                    for item in normal:
                        st.markdown(f"- 🟢 {item}")
                else:
                    st.write("Multiple clinical parameters deviate from median baseline.")

            # Biomarker deviations bar chart
            st.markdown("#### 📊 Patient Biomarker Deviation from Population Median")
            deviations = explanation.get("feature_deviations", {})
            dev_feats = list(deviations.keys())
            pct_diffs = [deviations[f]["percentage_difference"] for f in dev_feats]

            dev_df = pd.DataFrame({"Biomarker": dev_feats, "% Difference vs Median": pct_diffs})
            dev_df = dev_df.sort_values(by="% Difference vs Median", ascending=False)
            st.bar_chart(dev_df.set_index("Biomarker"))

        except Exception as e:
            st.error(f"Inference error: {e}")

# -------------------------------------------------------------
# 3. BATCH PREDICTIONS
# -------------------------------------------------------------
elif nav_choice == "📂 Batch Predictions":
    st.subheader("Batch Patient Evaluation & Triage Processing")
    st.write(
        "Upload a clinical CSV file containing patient measurements. The system validates the schema, "
        "processes all valid records, and produces downloadable predictions with estimated probabilities."
    )

    # Sample template download
    sample_df = pd.DataFrame([
        {"patient_id": "PAT_001", "Pregnancies": 2, "Glucose": 138, "BloodPressure": 76, "SkinThickness": 28, "Insulin": 120, "BMI": 32.4, "DiabetesPedigreeFunction": 0.52, "Age": 42},
        {"patient_id": "PAT_002", "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.35, "Age": 31},
        {"patient_id": "PAT_003", "Pregnancies": 7, "Glucose": 181, "BloodPressure": 84, "SkinThickness": 32, "Insulin": 192, "BMI": 35.9, "DiabetesPedigreeFunction": 0.68, "Age": 51}
    ])
    st.download_button(
        label="📥 Download Sample Batch Template (CSV)",
        data=sample_df.to_csv(index=False),
        file_name="medipredict_sample_batch.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Batch CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} patient records. Preview:")
            st.dataframe(batch_df.head(5))

            # Validate columns
            missing_cols = [c for c in FEATURE_NAMES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Batch file is missing required columns: {missing_cols}")
            else:
                if st.button("🚀 Run Batch Prediction Pipeline"):
                    with st.spinner("Processing batch records through trained pipeline..."):
                        records = batch_df.to_dict(orient="records")
                        results, scored_df = predict_batch_records(records)

                        st.success(f"Successfully generated predictions for {len(results)} records!")
                        st.dataframe(scored_df)

                        # Download button
                        csv_data = scored_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Scored Batch Results (CSV)",
                            data=csv_data,
                            file_name="medipredict_scored_batch.csv",
                            mime="text/csv"
                        )
        except Exception as err:
            st.error(f"Error processing batch file: {err}")

# -------------------------------------------------------------
# 4. DATASET EXPLORER
# -------------------------------------------------------------
elif nav_choice == "📊 Dataset Explorer":
    st.subheader("Exploratory Data Analysis & Quality Audit")

    raw_df, _ = get_cached_raw_data()
    quality_audit_path = PROJECT_ROOT / "reports" / "data_quality_audit.json"
    audit_data = None
    if quality_audit_path.exists():
        with open(quality_audit_path, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

    tab1, tab2, tab3 = st.tabs(["Data Preview & Summary", "Suspicious Zero Analysis", "Correlation Analysis"])

    with tab1:
        st.write("#### Clinical Dataset Sample")
        st.dataframe(raw_df.head(10))

        st.write("#### Descriptive Statistics")
        st.dataframe(raw_df.describe().T.round(2))

    with tab2:
        st.write("#### Biologically Implausible Zero Analysis")
        st.write(
            "Certain measurements (Glucose, Blood Pressure, Insulin, BMI, Skin Thickness) cannot be zero "
            "in living human physiology. In the raw dataset, zeroes represent missing or unrecorded values. "
            "MediPredict AI detects and converts these zeroes to NaN, imputing them via medians strictly on training folds."
        )

        if audit_data and "suspicious_zero_summary" in audit_data:
            zero_summary = audit_data["suspicious_zero_summary"]
            zero_records = []
            for feat, info in zero_summary.items():
                zero_records.append({
                    "Feature": feat,
                    "Zero Count": info["zero_count"],
                    "Percentage (%)": f"{info['percentage']}%",
                    "Physiological Reason": info["clinical_reason"]
                })
            st.table(pd.DataFrame(zero_records))

    with tab3:
        st.write("#### Feature Correlation Heatmap")
        st.write("Analyzes linear pairwise associations among clinical biomarkers and the diabetic outcome.")
        corr_fig = plot_correlation_heatmap(raw_df)
        st.plotly_chart(corr_fig, use_container_width=True)

# -------------------------------------------------------------
# 5. MODEL COMPARISON
# -------------------------------------------------------------
elif nav_choice == "⚖️ Model Comparison":
    st.subheader("Rigorous Model Comparison & Evaluation Metrics")
    st.write(
        "All models were trained with identical 5-fold stratified cross-validation on training data, "
        "and evaluated on the strictly held-out test set (20% of data). Zero data leakage."
    )

    eval_summary = get_cached_eval_summary()
    csv_summary_path = PROJECT_ROOT / "reports" / "evaluation_summary.csv"

    if csv_summary_path.exists():
        comp_df = pd.read_csv(csv_summary_path)
        st.write("### 🏆 Comprehensive Metrics Leaderboard")
        st.dataframe(comp_df)

    if eval_summary:
        col_roc, col_pr = st.columns(2)
        with col_roc:
            roc_fig = plot_roc_curves(eval_summary)
            st.plotly_chart(roc_fig, use_container_width=True)

        with col_pr:
            pr_fig = plot_pr_curves(eval_summary)
            st.plotly_chart(pr_fig, use_container_width=True)

        st.markdown("---")
        st.write("### 🔲 Confusion Matrices by Architecture")
        model_names = list(eval_summary.keys())
        selected_cm_model = st.selectbox("Select Model for Confusion Matrix Inspection", model_names)

        cm_fig = plot_confusion_matrix(eval_summary[selected_cm_model]["confusion_matrix"], selected_cm_model)
        st.plotly_chart(cm_fig, use_container_width=False)

        # Model selection methodology callout
        st.info(
            "💡 **Clinical Model Selection Rationale:** In diabetes screening, failing to flag an at-risk individual "
            "(False Negative) delays critical early clinical lifestyle or pharmacological intervention. "
            "Our multi-criteria Clinical Selection Score balances ROC-AUC (35%), Recall (30%), F1-Score (20%), "
            "and Probability Calibration Reliability (15% based on 1 - Brier Score)."
        )

# -------------------------------------------------------------
# 6. EXPLAINABLE AI
# -------------------------------------------------------------
elif nav_choice == "🔍 Explainable AI":
    st.subheader("Model Interpretability & Feature Importance")
    st.write(
        "Explainable AI (XAI) bridges the gap between machine learning predictions and clinical trust. "
        "We evaluate global model behavior via Permutation Importance and Logistic Regression Odds Ratios."
    )

    feat_imp = get_cached_feature_importance()
    if feat_imp:
        model_names = list(feat_imp.keys())
        selected_xai_model = st.selectbox("Select Model for Explanation Inspection", model_names)

        perm_data = feat_imp[selected_xai_model].get("permutation_importance", {})
        if perm_data:
            p_fig = plot_feature_importance_bar(perm_data, title=f"Permutation Importance on Held-Out Test Set: {selected_xai_model}")
            st.plotly_chart(p_fig, use_container_width=True)

        # Logistic Regression Coefficient / Odds Ratio inspection
        if selected_xai_model == "Logistic Regression":
            native_data = feat_imp[selected_xai_model].get("native_importance", {})
            if native_data:
                coef_fig = plot_logistic_regression_coefficients(native_data)
                st.plotly_chart(coef_fig, use_container_width=True)

    st.markdown("""
    > [!NOTE]
    > **Permutation Importance vs Impurity-Based Importance:**
    > Impurity-based feature importance in tree models (MDI) is heavily biased toward continuous features with high cardinality.
    > Permutation importance on held-out test data provides an unbiased, model-agnostic measurement of feature contribution.
    """)

# -------------------------------------------------------------
# 7. UNSUPERVISED LEARNING
# -------------------------------------------------------------
elif nav_choice == "🧬 Unsupervised Learning":
    st.subheader("Exploratory K-Means Clustering & PCA Projections")
    st.write(
        "Unsupervised learning partitions patients strictly based on feature similarity without utilizing ground truth target labels. "
        "This exploration assesses whether patient biomarker clusters naturally correlate with diabetic phenotypes."
    )

    raw_df, _ = get_cached_raw_data()
    with st.spinner("Executing K-Means clustering and PCA projection..."):
        kmeans_results = run_kmeans_analysis(raw_df)

    col1, col2 = st.columns(2)
    with col1:
        elbow_fig = plot_kmeans_elbow_silhouette(kmeans_results)
        st.plotly_chart(elbow_fig, use_container_width=True)

    with col2:
        pca_records = kmeans_results.get("pca_data", [])
        if pca_records:
            pca_fig = plot_pca_clusters(pca_records)
            st.plotly_chart(pca_fig, use_container_width=True)

    st.write("#### Cluster Centroids (Mean Biomarker Profile in Original Units)")
    centers = kmeans_results.get("cluster_centers", {})
    if centers:
        st.dataframe(pd.DataFrame(centers).T)

    st.warning(f"⚠️ {kmeans_results.get('disclaimer')}")

# -------------------------------------------------------------
# 8. MODEL MONITORING
# -------------------------------------------------------------
elif nav_choice == "📈 Model Monitoring":
    st.subheader("Operational Telemetry & Statistical Data Drift")
    st.write(
        "Continuous monitoring ensures models maintain reliability over time. MediPredict AI tracks "
        "prediction volume, inference latency, and statistical feature drift via Kolmogorov-Smirnov (KS) and PSI tests."
    )

    # Operational Telemetry
    telemetry = get_monitoring_summary()
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Inference Requests", telemetry.get("total_requests", 0))
    with m_col2:
        st.metric("Total Patients Scored", telemetry.get("total_predictions", 0))
    with m_col3:
        st.metric("Avg Prediction Latency", f"{telemetry.get('avg_latency_ms', 0)} ms")

    st.markdown("---")
    st.write("### 🧪 Statistical Data Drift Audit (KS-Test & Population Stability Index)")
    raw_df, _ = get_cached_raw_data()
    drift_res = evaluate_feature_drift(raw_df.iloc[:384], raw_df.iloc[384:])

    drift_col1, drift_col2 = st.columns(2)
    with drift_col1:
        st.markdown(f"**Drift Ratio:** `{drift_res['drift_ratio'] * 100}%`")
        st.markdown(f"**Drifted Features Count:** `{drift_res['drifted_features_count']}` / {drift_res['features_evaluated']}")
    with drift_col2:
        st.markdown(f"**Operational Recommendation:** {drift_res['recommendation']}")

    st.write("#### Biomarker Drift Breakdown")
    drift_records = []
    for feat, info in drift_res["feature_drift_details"].items():
        drift_records.append({
            "Feature": feat,
            "KS Statistic": info["ks_statistic"],
            "KS p-value": info["ks_pvalue"],
            "PSI Score": info["psi_score"],
            "PSI Status": info["psi_category"],
            "Drift Detected": "⚠️ YES" if info["has_drift"] else "✅ No"
        })
    st.dataframe(pd.DataFrame(drift_records))
