"""
MediPredict AI - Model Training & Lifecycle Orchestrator
Author: MediPredict AI Team
Description: Full reproducible ML training pipeline: Data Loading -> Quality Audit ->
             Stratified Split -> Hyperparameter Tuning -> 5-Fold Cross-Validation ->
             Held-out Evaluation -> Probability Calibration -> Model Selection ->
             Feature Importance -> Model Packaging -> MLflow Experiment Tracking.
"""

import sys
import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import joblib
import sklearn

from src.features.build_features import FEATURE_NAMES, TARGET_NAME
from src.data.load_data import load_raw_data
from src.data.validate_data import generate_data_quality_report
from src.data.preprocess_data import split_data
from src.training.tune import (
    tune_logistic_regression,
    tune_random_forest,
    tune_gradient_boosting,
    tune_svm,
    tune_xgboost,
    HAS_XGBOOST
)
from src.training.evaluate import (
    evaluate_classification_model,
    evaluate_cross_validation,
    compute_permutation_feature_importance,
    compute_learning_curve_data
)
from src.training.calibration import compute_calibration_curve, calibrate_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def calculate_clinical_selection_score(test_metrics: Dict[str, Any]) -> float:
    """
    Clinically informed model selection score prioritizing discrimination (ROC-AUC),
    sensitivity (Recall to minimize dangerous false negatives in diabetes screening),
    harmonic precision-recall balance (F1), and probability reliability (1 - Brier Score).
    
    Formula:
        Score = (0.35 * ROC_AUC) + (0.30 * Recall) + (0.20 * F1) + (0.15 * (1 - Brier_Score))
    """
    roc_auc = test_metrics.get("roc_auc") or 0.5
    recall = test_metrics.get("recall") or 0.0
    f1 = test_metrics.get("f1_score") or 0.0
    brier = test_metrics.get("brier_score") or 0.25
    brier_term = max(0.0, 1.0 - brier)

    score = (0.35 * roc_auc) + (0.30 * recall) + (0.20 * f1) + (0.15 * brier_term)
    return round(float(score), 4)


def extract_native_feature_importance(model, feature_names):
    """Extracts internal coefficients or impurity importances from pipeline."""
    clf = model.named_steps.get("classifier")
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
        return {f: round(float(v), 4) for f, v in zip(feature_names, imps)}
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        odds_ratios = np.exp(coefs)
        return {
            f: {
                "coefficient": round(float(c), 4),
                "odds_ratio": round(float(or_val), 4)
            }
            for f, c, or_val in zip(feature_names, coefs, odds_ratios)
        }
    return None


def run_training_pipeline(
    data_path=None,
    tune_models: bool = True,
    use_mlflow: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete machine learning lifecycle.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    logger.info("=== STARTING MEDIPREDICT AI TRAINING PIPELINE (%s) ===", timestamp)

    # 1. Load Data
    raw_df, data_hash = load_raw_data(data_path=data_path, auto_download=True)

    # 2. Audit Data Quality
    quality_report = generate_data_quality_report(raw_df)
    with open(REPORTS_DIR / "data_quality_audit.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    # 3. Train/Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = split_data(raw_df, test_size=0.20, random_state=random_state)

    # 4. Model Training & Tuning
    tuned_models = {}
    tuning_metadata = {}

    # Logistic Regression
    lr_pipe, lr_meta = tune_logistic_regression(X_train, y_train, cv_splits=5, random_state=random_state)
    tuned_models["Logistic Regression"] = lr_pipe
    tuning_metadata["Logistic Regression"] = lr_meta

    # Random Forest
    rf_pipe, rf_meta = tune_random_forest(X_train, y_train, cv_splits=5, random_state=random_state)
    tuned_models["Random Forest"] = rf_pipe
    tuning_metadata["Random Forest"] = rf_meta

    # Gradient Boosting
    gb_pipe, gb_meta = tune_gradient_boosting(X_train, y_train, cv_splits=5, random_state=random_state)
    tuned_models["Gradient Boosting"] = gb_pipe
    tuning_metadata["Gradient Boosting"] = gb_meta

    # Support Vector Machine (SVM)
    svm_pipe, svm_meta = tune_svm(X_train, y_train, cv_splits=5, random_state=random_state)
    tuned_models["SVM"] = svm_pipe
    tuning_metadata["SVM"] = svm_meta

    # XGBoost (optional)
    if HAS_XGBOOST:
        xgb_pipe, xgb_meta = tune_xgboost(X_train, y_train, cv_splits=5, random_state=random_state)
        if xgb_pipe is not None:
            tuned_models["XGBoost"] = xgb_pipe
            tuning_metadata["XGBoost"] = xgb_meta

    # 5. Model Evaluation (Cross-Validation + Held-out Test Set)
    evaluation_results = {}
    cv_results = {}
    feature_importances = {}
    selection_scores = {}
    calibration_reports = {}

    for name, model in tuned_models.items():
        logger.info("Evaluating model: %s", name)

        # 5-Fold Stratified Cross-Validation
        cv_res = evaluate_cross_validation(model, X_train, y_train, n_splits=5, random_state=random_state)
        cv_results[name] = cv_res

        # Held-out Test Set Evaluation
        test_res = evaluate_classification_model(model, X_test, y_test, model_name=name)
        evaluation_results[name] = test_res

        # Calibration Curve & Brier Score
        if hasattr(model, "predict_proba"):
            y_prob_test = model.predict_proba(X_test)[:, 1]
            cal_data = compute_calibration_curve(y_test.values, y_prob_test, n_bins=10)
            calibration_reports[name] = cal_data

        # Clinical Selection Score
        score = calculate_clinical_selection_score(test_res)
        selection_scores[name] = score

        # Feature Importance (Permutation on Test Set)
        perm_imp = compute_permutation_feature_importance(
            model, X_test, y_test, FEATURE_NAMES, n_repeats=10, random_state=random_state
        )
        native_imp = extract_native_feature_importance(model, FEATURE_NAMES)

        feature_importances[name] = {
            "permutation_importance": perm_imp,
            "native_importance": native_imp
        }

    # 6. Model Selection
    best_model_name = max(selection_scores, key=selection_scores.get)
    best_model_pipeline = tuned_models[best_model_name]
    logger.info(
        "=== BEST MODEL SELECTED: %s (Clinical Selection Score: %.4f) ===",
        best_model_name,
        selection_scores[best_model_name]
    )

    # Learning curve for best model
    learning_curve_res = compute_learning_curve_data(
        best_model_pipeline, X_train, y_train, cv=5, random_state=random_state
    )

    # 7. Model Persistence & Metadata
    model_metadata = {
        "project": "MediPredict AI",
        "timestamp": timestamp,
        "dataset_hash": data_hash,
        "selected_model_name": best_model_name,
        "clinical_selection_score": selection_scores[best_model_name],
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "models_evaluated": {},
        "environment": {
            "python_version": sys.version,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "has_xgboost": HAS_XGBOOST
        }
    }

    comparison_records = []
    for name in tuned_models.keys():
        t_eval = evaluation_results[name]
        c_eval = cv_results[name]
        model_metadata["models_evaluated"][name] = {
            "tuning_parameters": tuning_metadata[name],
            "cv_metrics_mean": {k: v["mean"] for k, v in c_eval.items()},
            "test_metrics": {
                "accuracy": t_eval["accuracy"],
                "balanced_accuracy": t_eval["balanced_accuracy"],
                "precision": t_eval["precision"],
                "recall": t_eval["recall"],
                "specificity": t_eval["specificity"],
                "f1_score": t_eval["f1_score"],
                "roc_auc": t_eval["roc_auc"],
                "pr_auc": t_eval["pr_auc"],
                "brier_score": t_eval["brier_score"]
            },
            "selection_score": selection_scores[name],
            "is_best": (name == best_model_name)
        }

        comparison_records.append({
            "Model": name,
            "Accuracy": t_eval["accuracy"],
            "Balanced Acc": t_eval["balanced_accuracy"],
            "Precision": t_eval["precision"],
            "Recall (Sensitivity)": t_eval["recall"],
            "Specificity": t_eval["specificity"],
            "F1-Score": t_eval["f1_score"],
            "ROC-AUC": t_eval["roc_auc"],
            "PR-AUC": t_eval["pr_auc"],
            "Brier Score": t_eval["brier_score"],
            "CV ROC-AUC (Mean±Std)": f"{c_eval['roc_auc']['mean']} ± {c_eval['roc_auc']['std']}",
            "Selection Score": selection_scores[name]
        })

    # Save Best Model Pipeline (includes Preprocessing + Estimator)
    best_model_path = MODELS_DIR / "medipredict_best_model.joblib"
    joblib.dump(best_model_pipeline, best_model_path)
    logger.info("Saved best model artifact to %s", best_model_path)

    # Save All Models
    all_models_path = MODELS_DIR / "all_trained_models.joblib"
    joblib.dump(tuned_models, all_models_path)

    # Save Metadata & Reports
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    with open(REPORTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)

    with open(REPORTS_DIR / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(calibration_reports, f, indent=2)

    with open(REPORTS_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feature_importances, f, indent=2)

    with open(REPORTS_DIR / "learning_curve.json", "w", encoding="utf-8") as f:
        json.dump(learning_curve_res, f, indent=2)

    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(REPORTS_DIR / "evaluation_summary.csv", index=False)
    logger.info("Comparison Summary Table:\n%s", comparison_df.to_string(index=False))

    # 8. Optional MLflow Experiment Tracking
    if use_mlflow:
        try:
            import mlflow
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            sqlite_db_path = (PROJECT_ROOT / "mlflow.db").resolve()
            mlflow.set_tracking_uri(f"sqlite:///{sqlite_db_path}")
            mlflow.set_experiment("MediPredictAI_Classification")
            with mlflow.start_run(run_name=f"Run_{best_model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("selected_model", best_model_name)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("data_hash", data_hash)

                best_test = evaluation_results[best_model_name]
                mlflow.log_metric("test_accuracy", best_test["accuracy"])
                mlflow.log_metric("test_recall", best_test["recall"])
                mlflow.log_metric("test_f1", best_test["f1_score"])
                mlflow.log_metric("test_roc_auc", best_test["roc_auc"])
                mlflow.log_metric("test_brier", best_test["brier_score"])
                mlflow.log_metric("selection_score", selection_scores[best_model_name])
                logger.info("MLflow logging completed successfully with SQLite backend.")
        except Exception as e:
            logger.warning("MLflow logging was bypassed or encountered an error: %s", e)

    logger.info("=== TRAINING PIPELINE FINISHED SUCCESSFULLY ===")
    return {
        "best_model_name": best_model_name,
        "selection_score": selection_scores[best_model_name],
        "comparison_table": comparison_df.to_dict(orient="records"),
        "best_model_path": str(best_model_path)
    }


if __name__ == "__main__":
    results = run_training_pipeline()
    print("\nTraining completed successfully! Best model:", results["best_model_name"])
