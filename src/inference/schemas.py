"""
MediPredict AI - Pydantic Request & Response Schemas
Author: MediPredict AI Team
Description: Validated API schemas for single-patient and batch clinical prediction.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class PatientInput(BaseModel):
    """
    Validated clinical measurement inputs for a single patient.
    """
    Pregnancies: int = Field(
        default=1,
        ge=0,
        le=25,
        description="Number of times pregnant"
    )
    Glucose: float = Field(
        default=115.0,
        ge=30.0,
        le=400.0,
        description="2-Hour plasma glucose concentration (mg/dL)"
    )
    BloodPressure: float = Field(
        default=72.0,
        ge=30.0,
        le=200.0,
        description="Diastolic blood pressure (mm Hg)"
    )
    SkinThickness: float = Field(
        default=25.0,
        ge=0.0,
        le=120.0,
        description="Triceps skinfold thickness (mm). 0 denotes unmeasured."
    )
    Insulin: float = Field(
        default=95.0,
        ge=0.0,
        le=1000.0,
        description="2-Hour serum insulin (uU/mL). 0 denotes unmeasured."
    )
    BMI: float = Field(
        default=28.5,
        ge=10.0,
        le=80.0,
        description="Body Mass Index (kg/m^2)"
    )
    DiabetesPedigreeFunction: float = Field(
        default=0.45,
        ge=0.01,
        le=3.5,
        description="Diabetes pedigree function (genetic score)"
    )
    Age: int = Field(
        default=32,
        ge=18,
        le=120,
        description="Patient age in years"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 2,
                "Glucose": 138.0,
                "BloodPressure": 76.0,
                "SkinThickness": 28.0,
                "Insulin": 120.0,
                "BMI": 32.4,
                "DiabetesPedigreeFunction": 0.52,
                "Age": 42
            }
        }
    }


class LocalExplanation(BaseModel):
    feature_deviations: Dict[str, Dict[str, Any]]
    top_elevated_risk_factors: List[str]
    protective_or_normal_factors: List[str]


class PredictionResponse(BaseModel):
    """Structured response for a single patient prediction."""
    prediction: int = Field(..., description="Binary prediction: 0 (Low/No Diabetes Risk) or 1 (Elevated Risk)")
    probability: float = Field(..., description="Estimated probability of positive class (Outcome=1)")
    risk_category: str = Field(..., description="Risk tier: Low Risk, Moderate Risk, or High Risk")
    model_name: str = Field(..., description="Name of the classification algorithm used")
    model_version: str = Field(default="1.0.0", description="Semantic version of model pipeline")
    timestamp: str = Field(..., description="ISO 8601 UTC inference timestamp")
    disclaimer: str = Field(..., description="Mandatory educational clinical disclaimer")
    explanation: Optional[LocalExplanation] = None


class BatchPatientRecord(PatientInput):
    patient_id: Optional[str] = Field(default=None, description="Optional unique identifier for row tracking")


class BatchPredictionRequest(BaseModel):
    records: List[BatchPatientRecord] = Field(..., max_length=500, description="List of patient records (max 500)")


class SingleBatchResult(BaseModel):
    patient_id: Optional[str]
    row_index: int
    prediction: int
    probability: float
    risk_category: str


class BatchPredictionResponse(BaseModel):
    total_records_received: int
    successful_predictions: int
    failed_records: int
    predictions: List[SingleBatchResult]
    model_name: str
    model_version: str
    disclaimer: str


class ModelInfoResponse(BaseModel):
    project_name: str
    selected_model_name: str
    training_timestamp: str
    dataset_hash: str
    feature_names: List[str]
    metrics_summary: Dict[str, Any]
    environment: Dict[str, Any]
    disclaimer: str
