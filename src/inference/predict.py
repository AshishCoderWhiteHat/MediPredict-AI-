"""
MediPredict AI - Inference & Prediction Engine
Author: MediPredict AI Team
Description: Loads serialized model pipelines and generates single/batch predictions
             with probability calibration, risk tier categorization, and local biomarker explanations.
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import joblib

from src.features.build_features import FEATURE_NAMES, FEATURE_SCHEMA
from src.inference.schemas import (
    PatientInput,
    PredictionResponse,
    LocalExplanation,
    SingleBatchResult,
    BatchPredictionResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "medipredict_best_model.joblib"
ALL_MODELS_PATH = MODELS_DIR / "all_trained_models.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

CLINICAL_DISCLAIMER = (
    "EDUCATIONAL DISCLAIMER: MediPredict AI is an academic prototype developed for educational and research "
    "demonstrations only. It is NOT a certified medical diagnostic system and MUST NOT be used for medical decisions, "
    "prescriptions, or patient triage. Always consult a qualified healthcare professional."
)

# Reference population medians for biomarker comparison
POPULATION_MEDIANS = {
    "Pregnancies": 3.0,
    "Glucose": 117.0,
    "BloodPressure": 72.0,
    "SkinThickness": 29.0,
    "Insulin": 125.0,
    "BMI": 32.4,
    "DiabetesPedigreeFunction": 0.3825,
    "Age": 29.0
}

_CACHED_BEST_MODEL = None
_CACHED_ALL_MODELS = None
_CACHED_METADATA = None


def load_model_metadata() -> Dict[str, Any]:
    """Loads model training metadata and evaluated metrics."""
    global _CACHED_METADATA
    if _CACHED_METADATA is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Model metadata not found at {METADATA_PATH}. Run training first.")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _CACHED_METADATA = json.load(f)
    return _CACHED_METADATA


def load_best_model():
    """Loads and caches the best serialized model pipeline."""
    global _CACHED_BEST_MODEL
    if _CACHED_BEST_MODEL is None:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {BEST_MODEL_PATH}. "
                f"Please run the training pipeline first: python -m src.training.train"
            )
        logger.info("Loading best model from %s", BEST_MODEL_PATH)
        _CACHED_BEST_MODEL = joblib.load(BEST_MODEL_PATH)
    return _CACHED_BEST_MODEL


def load_all_models() -> Dict[str, Any]:
    """Loads all trained pipelines for comparative exploration."""
    global _CACHED_ALL_MODELS
    if _CACHED_ALL_MODELS is None:
        if not ALL_MODELS_PATH.exists():
            raise FileNotFoundError(f"All models artifact not found at {ALL_MODELS_PATH}.")
        _CACHED_ALL_MODELS = joblib.load(ALL_MODELS_PATH)
    return _CACHED_ALL_MODELS


def categorize_risk(probability: float) -> str:
    """Categorizes diabetes probability into clinically interpretable risk tiers."""
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.65:
        return "Moderate Risk"
    else:
        return "High Risk"


def compute_biomarker_explanation(patient_dict: Dict[str, Any]) -> LocalExplanation:
    """
    Computes a transparent local feature explanation by evaluating how the patient's
    biomarkers deviate from baseline population medians and clinical thresholds.
    """
    deviations = {}
    elevated = []
    normal = []

    for feat in FEATURE_NAMES:
        val = float(patient_dict.get(feat, 0.0))
        ref_med = POPULATION_MEDIANS.get(feat, 1.0)
        schema = FEATURE_SCHEMA[feat]

        pct_diff = round(((val - ref_med) / ref_med) * 100, 1) if ref_med > 0 else 0.0

        # Clinical elevation criteria
        is_elevated = False
        if feat == "Glucose" and val >= 140.0:
            is_elevated = True
            elevated.append(f"Elevated Blood Glucose: {val} {schema['unit']} (impaired/diabetic threshold is >=140)")
        elif feat == "BMI" and val >= 30.0:
            is_elevated = True
            elevated.append(f"Elevated BMI: {val} {schema['unit']} (clinical obesity threshold is >=30.0)")
        elif feat == "BloodPressure" and val >= 85.0:
            is_elevated = True
            elevated.append(f"Elevated Blood Pressure: {val} {schema['unit']} (pre-hypertension/hypertension)")
        elif feat == "Age" and val >= 45:
            is_elevated = True
            elevated.append(f"Age Risk Factor: {int(val)} years (elevated age bracket)")
        elif feat == "DiabetesPedigreeFunction" and val >= 0.70:
            is_elevated = True
            elevated.append(f"Strong Family History: DPF {val} (high hereditary predisposition)")
        elif pct_diff > 25.0 and val > 0:
            is_elevated = True
            elevated.append(f"{schema['display_name']} is {pct_diff}% above cohort median")
        else:
            normal.append(f"{schema['display_name']} ({val} {schema['unit']}) is within normal/reference range")

        deviations[feat] = {
            "patient_value": val,
            "population_median": ref_med,
            "percentage_difference": pct_diff,
            "unit": schema["unit"],
            "is_elevated": is_elevated
        }

    return LocalExplanation(
        feature_deviations=deviations,
        top_elevated_risk_factors=elevated[:4],
        protective_or_normal_factors=normal[:4]
    )


def predict_single(
    patient_data: Union[PatientInput, Dict[str, Any]],
    model_name_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes inference for a single patient record.
    """
    # Standardize dictionary
    if isinstance(patient_data, PatientInput):
        input_dict = patient_data.model_dump()
    else:
        input_dict = dict(patient_data)

    # Validate presence of features
    df_input = pd.DataFrame([input_dict])[FEATURE_NAMES]

    # Select pipeline
    metadata = load_model_metadata()
    if model_name_override:
        all_models = load_all_models()
        if model_name_override not in all_models:
            raise ValueError(f"Model '{model_name_override}' not found. Available: {list(all_models.keys())}")
        pipeline = all_models[model_name_override]
        active_model_name = model_name_override
    else:
        pipeline = load_best_model()
        active_model_name = metadata.get("selected_model_name", "Best Model")

    # Run inference through pipeline
    prediction_raw = int(pipeline.predict(df_input)[0])

    if hasattr(pipeline, "predict_proba"):
        prob_positive = float(pipeline.predict_proba(df_input)[0, 1])
    else:
        prob_positive = 1.0 if prediction_raw == 1 else 0.0

    risk_category = categorize_risk(prob_positive)
    explanation = compute_biomarker_explanation(input_dict)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "prediction": prediction_raw,
        "probability": round(prob_positive, 4),
        "risk_category": risk_category,
        "model_name": active_model_name,
        "model_version": "1.0.0",
        "timestamp": timestamp,
        "disclaimer": CLINICAL_DISCLAIMER,
        "explanation": explanation.model_dump()
    }


def predict_batch_records(
    records: List[Dict[str, Any]],
    model_name_override: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Executes batch inference preserving patient identifiers and row sequence.
    """
    if not records:
        return [], pd.DataFrame()

    results = []
    df_records = pd.DataFrame(records)

    # Validate features present
    for feat in FEATURE_NAMES:
        if feat not in df_records.columns:
            raise ValueError(f"Batch dataset missing required feature: '{feat}'")

    metadata = load_model_metadata()
    if model_name_override:
        all_models = load_all_models()
        pipeline = all_models[model_name_override]
        active_model_name = model_name_override
    else:
        pipeline = load_best_model()
        active_model_name = metadata.get("selected_model_name", "Best Model")

    X_batch = df_records[FEATURE_NAMES].copy()
    preds = pipeline.predict(X_batch)

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_batch)[:, 1]
    else:
        probs = [1.0 if p == 1 else 0.0 for p in preds]

    out_df = df_records.copy()
    out_df["Predicted_Outcome"] = preds
    out_df["Estimated_Probability"] = np.round(probs, 4)
    out_df["Risk_Category"] = [categorize_risk(p) for p in probs]
    out_df["Model_Used"] = active_model_name

    for idx, (p, prob) in enumerate(zip(preds, probs)):
        pat_id = str(df_records.iloc[idx].get("patient_id", f"PAT_{idx+1}"))
        results.append({
            "patient_id": pat_id,
            "row_index": idx,
            "prediction": int(p),
            "probability": round(float(prob), 4),
            "risk_category": categorize_risk(float(prob))
        })

    return results, out_df
