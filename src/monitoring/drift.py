"""
MediPredict AI - Statistical Data Drift Monitoring Module
Author: MediPredict AI Team
Description: Detects covariate data drift using the two-sample Kolmogorov-Smirnov (KS) test
             and Population Stability Index (PSI) against baseline training distributions.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from src.features.build_features import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline (expected) and current (actual).
    
    Interpretation:
        PSI < 0.10: No significant population shift.
        0.10 <= PSI < 0.20: Moderate population shift; monitor closely.
        PSI >= 0.20: Substantial distribution drift; retraining or calibration review recommended.
    """
    if len(expected) < 5 or len(actual) < 5:
        return 0.0

    # Determine quantiles based on expected reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    # Ensure unique bucket bounds
    bucket_bounds = np.unique(bucket_bounds)
    if len(bucket_bounds) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bucket_bounds)
    actual_counts, _ = np.histogram(actual, bins=bucket_bounds)

    # Convert to fractions with small epsilon to avoid log(0) or division by 0
    eps = 1e-4
    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps

    # Normalize fractions
    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi_val, 4))


def evaluate_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Executes two-sample Kolmogorov-Smirnov test and Population Stability Index across features.
    
    Args:
        reference_df: Baseline training feature distribution.
        current_df: Current batch or inference feature distribution.
        features: List of feature names to inspect.
        alpha: Statistical significance threshold for KS test (default 0.05).
        
    Returns:
        Structured drift assessment dictionary with feature-level metrics and recommendations.
    """
    feats = features or [f for f in FEATURE_NAMES if f in reference_df.columns and f in current_df.columns]
    feature_reports = {}
    drifted_count = 0

    for feat in feats:
        ref_vals = reference_df[feat].dropna().values
        curr_vals = current_df[feat].dropna().values

        if len(ref_vals) < 5 or len(curr_vals) < 5:
            continue

        # Two-sample Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, curr_vals)
        psi_score = calculate_psi(ref_vals, curr_vals)

        is_ks_drift = bool(ks_pvalue < alpha)
        is_psi_drift = bool(psi_score >= 0.20)
        has_drift = is_ks_drift or is_psi_drift

        if has_drift:
            drifted_count += 1

        psi_category = "Stable"
        if psi_score >= 0.20:
            psi_category = "Significant Drift"
        elif psi_score >= 0.10:
            psi_category = "Moderate Drift"

        feature_reports[feat] = {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 4),
            "is_statistically_significant": is_ks_drift,
            "psi_score": psi_score,
            "psi_category": psi_category,
            "has_drift": has_drift,
            "ref_mean": round(float(np.mean(ref_vals)), 2),
            "curr_mean": round(float(np.mean(curr_vals)), 2)
        }

    drift_ratio = round(drifted_count / len(feats), 3) if feats else 0.0
    recommendation = "Distribution stable. Continue routine monitoring."
    if drift_ratio >= 0.40:
        recommendation = (
            "Severe feature drift detected across >40% of biomarkers. "
            "Inspect clinical data acquisition protocols, assess population demographic shifts, "
            "and consider triggering a model retraining and evaluation cycle."
        )
    elif drift_ratio > 0.0:
        recommendation = (
            "Moderate drift detected in isolated features. "
            "Verify calibration stability on recent patient batches."
        )

    return {
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
        "features_evaluated": len(feats),
        "drifted_features_count": drifted_count,
        "drift_ratio": drift_ratio,
        "overall_drift_detected": drifted_count > 0,
        "recommendation": recommendation,
        "feature_drift_details": feature_reports,
        "retraining_workflow": {
            "retraining_command": "python -m src.training.train",
            "validation_requirement": "New model must achieve clinical selection score >= current champion model."
        }
    }
