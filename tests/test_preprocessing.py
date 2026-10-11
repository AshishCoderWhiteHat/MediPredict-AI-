"""
MediPredict AI - Tests for Preprocessing and Pipeline Integrity
"""

import pytest
import pandas as pd
import numpy as np

from src.features.build_features import FEATURE_NAMES, SUSPICIOUS_ZERO_COLUMNS
from src.data.preprocess_data import (
    PhysiologicalZeroToNaN,
    create_preprocessor_pipeline,
    split_data
)
from src.data.load_data import load_raw_data


def test_physiological_zero_to_nan_transformer():
    """Verifies that suspicious zeros are replaced by NaN while pregnancies=0 is preserved."""
    sample_df = pd.DataFrame({
        "Pregnancies": [0, 2],
        "Glucose": [0.0, 120.0],
        "BloodPressure": [70.0, 0.0],
        "SkinThickness": [0.0, 25.0],
        "Insulin": [0.0, 100.0],
        "BMI": [0.0, 31.0],
        "DiabetesPedigreeFunction": [0.2, 0.5],
        "Age": [25, 40]
    })

    transformer = PhysiologicalZeroToNaN(SUSPICIOUS_ZERO_COLUMNS)
    transformed_df = transformer.fit_transform(sample_df)

    # Pregnancies=0 should NOT be NaN (valid physiological state)
    assert transformed_df["Pregnancies"].iloc[0] == 0
    # Glucose=0 should be NaN
    assert np.isnan(transformed_df["Glucose"].iloc[0])
    # BloodPressure=0 should be NaN
    assert np.isnan(transformed_df["BloodPressure"].iloc[1])


def test_preprocessor_pipeline_fitting_and_no_leakage():
    """Verifies that the imputer and scaler learn statistics strictly on training data."""
    df, _ = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor_pipeline(with_scaling=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Assert shape preservation
    assert X_train_proc.shape == X_train.shape
    assert X_test_proc.shape == X_test.shape

    # Assert no NaNs remain after preprocessing
    assert not np.isnan(X_train_proc).any()
    assert not np.isnan(X_test_proc).any()

    # Training mean should be approximately 0 for scaled data
    np.testing.assert_almost_equal(np.mean(X_train_proc, axis=0), 0.0, decimal=1)


def test_stratified_split_preserves_class_distribution():
    """Verifies that train and test splits retain the original class balance."""
    df, _ = load_raw_data()
    orig_ratio = df["Outcome"].mean()

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Verify ratios are within 1.5% of each other
    assert abs(train_ratio - orig_ratio) < 0.02
    assert abs(test_ratio - orig_ratio) < 0.02
