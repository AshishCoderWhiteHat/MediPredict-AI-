"""
MediPredict AI - Tests for Data Ingestion and Validation
"""

import pytest
import pandas as pd
import numpy as np

from src.features.build_features import FEATURE_NAMES, TARGET_NAME
from src.data.load_data import load_raw_data
from src.data.validate_data import (
    validate_schema,
    inspect_suspicious_zeros,
    inspect_duplicates,
    validate_target_distribution,
    generate_data_quality_report,
    DataValidationError
)


def test_valid_dataset_loads_successfully():
    """Verifies that the canonical dataset loads and has expected shape and columns."""
    df, sha = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 700
    assert df.shape[1] == len(FEATURE_NAMES) + 1
    assert all(col in df.columns for col in FEATURE_NAMES)
    assert TARGET_NAME in df.columns
    assert len(sha) == 64  # SHA-256 length


def test_invalid_schema_produces_error():
    """Verifies that missing columns are flagged in schema validation."""
    invalid_df = pd.DataFrame({"Pregnancies": [1, 2], "Age": [25, 30]})
    is_valid, issues = validate_schema(invalid_df, require_target=True)
    assert not is_valid
    assert len(issues) > 0


def test_invalid_target_labels_rejected():
    """Verifies that non-binary target labels trigger DataValidationError."""
    invalid_df = pd.DataFrame({
        TARGET_NAME: [0, 1, 2, 5]
    })
    with pytest.raises(DataValidationError):
        validate_target_distribution(invalid_df)


def test_suspicious_zero_inspection():
    """Verifies that physiological zeros are identified correctly."""
    mock_df = pd.DataFrame({
        "Glucose": [0, 120, 140],
        "BloodPressure": [0, 0, 75],
        "BMI": [25.0, 0, 32.0],
        "Insulin": [0, 0, 0],
        "SkinThickness": [0, 20, 30]
    })
    zero_report = inspect_suspicious_zeros(mock_df)
    assert zero_report["Glucose"]["zero_count"] == 1
    assert zero_report["BloodPressure"]["zero_count"] == 2
    assert zero_report["BMI"]["zero_count"] == 1
    assert zero_report["Insulin"]["zero_count"] == 3


def test_duplicate_detection():
    """Verifies duplicate row counting."""
    mock_df = pd.DataFrame({
        "Glucose": [100, 100, 120],
        "Age": [30, 30, 40]
    })
    dup_stats = inspect_duplicates(mock_df)
    assert dup_stats["duplicate_rows"] == 1


def test_data_quality_report_structure():
    """Verifies structure of comprehensive quality report."""
    df, _ = load_raw_data()
    report = generate_data_quality_report(df)
    assert report["total_rows"] == len(df)
    assert report["is_schema_valid"] is True
    assert "suspicious_zero_summary" in report
    assert "target_summary" in report
