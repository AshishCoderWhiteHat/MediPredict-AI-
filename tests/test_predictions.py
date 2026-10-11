"""
MediPredict AI - Tests for Inference and Prediction Workflows
"""

import pytest
import pandas as pd
from pydantic import ValidationError

from src.inference.schemas import PatientInput
from src.inference.predict import (
    predict_single,
    predict_batch_records,
    categorize_risk,
    compute_biomarker_explanation
)


def test_valid_patient_single_prediction():
    """Verifies that a valid patient record produces an expected prediction structure."""
    patient = {
        "Pregnancies": 2,
        "Glucose": 140.0,
        "BloodPressure": 75.0,
        "SkinThickness": 28.0,
        "Insulin": 130.0,
        "BMI": 33.2,
        "DiabetesPedigreeFunction": 0.55,
        "Age": 45
    }

    res = predict_single(patient)

    assert res["prediction"] in (0, 1)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk_category"] in ("Low Risk", "Moderate Risk", "High Risk")
    assert "explanation" in res
    assert "disclaimer" in res
    assert len(res["explanation"]["top_elevated_risk_factors"]) > 0


def test_risk_categorization_thresholds():
    """Verifies risk tier categorization boundaries."""
    assert categorize_risk(0.15) == "Low Risk"
    assert categorize_risk(0.29) == "Low Risk"
    assert categorize_risk(0.30) == "Moderate Risk"
    assert categorize_risk(0.64) == "Moderate Risk"
    assert categorize_risk(0.65) == "High Risk"
    assert categorize_risk(0.95) == "High Risk"


def test_batch_prediction_workflow():
    """Verifies batch inference preserves row ordering and identifiers."""
    batch_records = [
        {"patient_id": "P1", "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 20, "Insulin": 50, "BMI": 24.5, "DiabetesPedigreeFunction": 0.25, "Age": 28},
        {"patient_id": "P2", "Pregnancies": 6, "Glucose": 165, "BloodPressure": 80, "SkinThickness": 30, "Insulin": 180, "BMI": 36.0, "DiabetesPedigreeFunction": 0.85, "Age": 55}
    ]

    results, out_df = predict_batch_records(batch_records)

    assert len(results) == 2
    assert results[0]["patient_id"] == "P1"
    assert results[1]["patient_id"] == "P2"
    assert "Estimated_Probability" in out_df.columns
    assert "Predicted_Outcome" in out_df.columns


def test_invalid_patient_input_validation():
    """Verifies that invalid input values are rejected by Pydantic."""
    # Negative age should fail validation
    with pytest.raises(ValidationError):
        PatientInput(
            Pregnancies=1,
            Glucose=100.0,
            BloodPressure=70.0,
            SkinThickness=20.0,
            Insulin=80.0,
            BMI=25.0,
            DiabetesPedigreeFunction=0.4,
            Age=-5
        )
