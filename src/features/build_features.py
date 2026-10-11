"""
MediPredict AI - Feature Engineering & Schema Definitions
Author: MediPredict AI Team
Description: Single source of truth for feature schema, constraints, units, and engineering transforms.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Single source of truth for feature order and types
FEATURE_NAMES: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET_NAME: str = "Outcome"

# Physiologically impossible zero columns that represent missing values
SUSPICIOUS_ZERO_COLUMNS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Clinical and operational range limits for validation and UI controls
FEATURE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "Pregnancies": {
        "display_name": "Pregnancies",
        "description": "Number of times pregnant",
        "unit": "count",
        "type": "int",
        "min_value": 0,
        "max_value": 20,
        "default": 1,
        "step": 1,
        "zero_is_valid": True,
        "clinical_context": "Nulliparous patients (0 pregnancies) are clinically valid. High parity (>=6) is associated with gestational stress."
    },
    "Glucose": {
        "display_name": "Plasma Glucose Concentration",
        "description": "2-hour plasma glucose concentration in an oral glucose tolerance test (OGTT)",
        "unit": "mg/dL",
        "type": "float",
        "min_value": 40.0,
        "max_value": 300.0,
        "default": 115.0,
        "step": 1.0,
        "zero_is_valid": False,
        "clinical_context": "Normal fasting: 70-99 mg/dL; 2h OGTT impaired: 140-199 mg/dL; Diabetic: >=200 mg/dL. Zero is biologically impossible in living patients."
    },
    "BloodPressure": {
        "display_name": "Diastolic Blood Pressure",
        "description": "Diastolic blood pressure",
        "unit": "mm Hg",
        "type": "float",
        "min_value": 30.0,
        "max_value": 140.0,
        "default": 72.0,
        "step": 1.0,
        "zero_is_valid": False,
        "clinical_context": "Normal: <80 mm Hg; Stage 1 hypertension: 80-89 mm Hg; Stage 2: >=90 mm Hg. Zero indicates absent circulatory pressure (measurement omission)."
    },
    "SkinThickness": {
        "display_name": "Triceps Skinfold Thickness",
        "description": "Triceps skinfold thickness used to estimate subcutaneous fat percentage",
        "unit": "mm",
        "type": "float",
        "min_value": 5.0,
        "max_value": 100.0,
        "default": 25.0,
        "step": 1.0,
        "zero_is_valid": False,
        "clinical_context": "Typical female adult range: 10-35 mm. Zero represents unrecorded or omitted caliper measurement."
    },
    "Insulin": {
        "display_name": "2-Hour Serum Insulin",
        "description": "2-Hour postprandial serum insulin level",
        "unit": "uU/mL",
        "type": "float",
        "min_value": 10.0,
        "max_value": 900.0,
        "default": 95.0,
        "step": 1.0,
        "zero_is_valid": False,
        "clinical_context": "Normal 2h post-load: 16-166 uU/mL. Zero is highly prevalent in the Pima dataset (approx 48%) due to omitted assay tests."
    },
    "BMI": {
        "display_name": "Body Mass Index",
        "description": "Weight in kilograms divided by height in meters squared",
        "unit": "kg/m^2",
        "type": "float",
        "min_value": 10.0,
        "max_value": 70.0,
        "default": 28.5,
        "step": 0.1,
        "zero_is_valid": False,
        "clinical_context": "Normal: 18.5-24.9 kg/m^2; Overweight: 25.0-29.9 kg/m^2; Obese: >=30.0 kg/m^2. Zero represents unrecorded patient stature/weight."
    },
    "DiabetesPedigreeFunction": {
        "display_name": "Diabetes Pedigree Function",
        "description": "Genetic scoring scoring the diabetes family history and ancestral risk",
        "unit": "score",
        "type": "float",
        "min_value": 0.05,
        "max_value": 3.0,
        "default": 0.45,
        "step": 0.01,
        "zero_is_valid": False,
        "clinical_context": "Higher values denote stronger hereditary predisposition based on degree of relationship of diabetic relatives."
    },
    "Age": {
        "display_name": "Patient Age",
        "description": "Patient age in years",
        "unit": "years",
        "type": "int",
        "min_value": 18,
        "max_value": 110,
        "default": 32,
        "step": 1,
        "zero_is_valid": False,
        "clinical_context": "Pima cohort consists of females >= 21 years old. Risk typically escalates markedly after age 35."
    }
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-informed risk flags for exploratory data analysis.
    Note: The core ML classification pipelines primarily use canonical standardized features 
    to prevent collinearity, but these flags provide rich contextual insights for analytics.
    """
    df_engineered = df.copy()

    # Clinical BMI Classification (WHO guidelines)
    # Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: >=30
    if "BMI" in df_engineered.columns:
        df_engineered["Is_Obese"] = (df_engineered["BMI"] >= 30.0).astype(int)

    # Impaired Fasting/OGTT Glucose (ADA guidelines)
    if "Glucose" in df_engineered.columns:
        df_engineered["High_Glucose"] = (df_engineered["Glucose"] >= 140.0).astype(int)

    # Age Risk Tier (Age >= 35)
    if "Age" in df_engineered.columns:
        df_engineered["Senior_Risk_Age"] = (df_engineered["Age"] >= 35).astype(int)

    # Insulin Resistance Index approximation (Glucose * Insulin / 405 proxy for HOMA-IR when non-zero)
    if "Glucose" in df_engineered.columns and "Insulin" in df_engineered.columns:
        df_engineered["Glucose_Insulin_Ratio"] = np.where(
            (df_engineered["Insulin"] > 0) & (df_engineered["Glucose"] > 0),
            (df_engineered["Glucose"] * df_engineered["Insulin"]) / 405.0,
            np.nan
        )

    return df_engineered


def get_feature_schema() -> Dict[str, Dict[str, Any]]:
    """Returns the immutable feature schema dictionary."""
    return FEATURE_SCHEMA


def get_feature_names() -> List[str]:
    """Returns ordered list of model input features."""
    return list(FEATURE_NAMES)
