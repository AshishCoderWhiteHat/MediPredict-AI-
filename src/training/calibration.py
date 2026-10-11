"""
MediPredict AI - Probability Calibration Module
Author: MediPredict AI Team
Description: Probability calibration analysis using Platt scaling (sigmoid) and isotonic regression,
             computing calibration curves and Brier score loss.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes calibration curve coordinates and Brier score.
    
    A perfectly calibrated model has prob_true == prob_pred (along the 45-degree diagonal).
    Brier score is the mean squared difference between predicted probability and actual binary label.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": round(brier, 4),
        "n_bins": n_bins
    }


def calibrate_pipeline(
    pipeline,
    X_train,
    y_train,
    method: str = "sigmoid",
    cv: int = 5
):
    """
    Applies CalibratedClassifierCV to an existing model pipeline using internal cross-validation
    to prevent calibration leakage.
    
    Args:
        pipeline: Trained or configured Scikit-learn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        method: 'sigmoid' (Platt scaling) or 'isotonic'.
        cv: Number of cross-validation folds for calibration.
        
    Returns:
        Fitted CalibratedClassifierCV wrapper.
    """
    logger.info("Calibrating pipeline with method='%s', cv=%d", method, cv)
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=cv
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model
