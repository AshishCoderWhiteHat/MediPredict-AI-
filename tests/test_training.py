"""
MediPredict AI - Tests for Model Training, Evaluation, and Calibration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.build_features import FEATURE_NAMES
from src.data.load_data import load_raw_data
from src.data.preprocess_data import split_data, create_preprocessor_pipeline
from src.training.tune import tune_logistic_regression, tune_decision_tree
from src.training.evaluate import (
    evaluate_classification_model,
    evaluate_cross_validation,
    compute_permutation_feature_importance
)
from src.training.calibration import compute_calibration_curve
from src.inference.predict import load_best_model, load_model_metadata


def test_saved_model_artifact_loads():
    """Verifies that the serialized production model artifact can be loaded successfully."""
    model = load_best_model()
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_metadata_structure():
    """Verifies that saved metadata contains required engineering provenance fields."""
    meta = load_model_metadata()
    assert "project" in meta
    assert "selected_model_name" in meta
    assert "dataset_hash" in meta
    assert "models_evaluated" in meta
    assert len(meta["feature_names"]) == 8


def test_evaluation_metric_bounds():
    """Verifies that classification metrics fall strictly within valid mathematical bounds."""
    df, _ = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    model = load_best_model()
    eval_res = evaluate_classification_model(model, X_test, y_test, model_name="BestModel")

    assert 0.0 <= eval_res["accuracy"] <= 1.0
    assert 0.0 <= eval_res["precision"] <= 1.0
    assert 0.0 <= eval_res["recall"] <= 1.0
    assert 0.0 <= eval_res["f1_score"] <= 1.0
    assert 0.0 <= eval_res["specificity"] <= 1.0
    assert 0.0 <= eval_res["roc_auc"] <= 1.0
    assert 0.0 <= eval_res["brier_score"] <= 1.0


def test_calibration_curve_calculation():
    """Verifies probability calibration curve calculations."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.85, 0.15])
    cal_res = compute_calibration_curve(y_true, y_prob, n_bins=5)

    assert "prob_true" in cal_res
    assert "prob_pred" in cal_res
    assert 0.0 <= cal_res["brier_score"] <= 1.0
