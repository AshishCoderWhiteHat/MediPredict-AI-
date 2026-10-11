"""
MediPredict AI - Evaluation Module
Author: MediPredict AI Team
Description: Rigorous classification metrics, confusion matrices, ROC/PR curves,
             cross-validation evaluation, and permutation feature importance.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_classification_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics on held-out test data.
    """
    y_pred = model.predict(X_test)

    # Probabilities for positive class (1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-d))
    else:
        y_prob = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
    pr_auc = float(average_precision_score(y_test, y_prob)) if y_prob is not None else None
    brier = float(brier_score_loss(y_test, y_prob)) if y_prob is not None else None

    # Curves for interactive plotting
    roc_data = {}
    pr_data = {}
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
        pr_data = {
            "precision": p_curve.tolist(),
            "recall": r_curve.tolist()
        }

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    result = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "classification_report": report_dict
    }

    logger.info(
        "Evaluated '%s': Acc=%.4f, Rec=%.4f, Prec=%.4f, F1=%.4f, ROC-AUC=%s",
        model_name, acc, rec, prec, f1, str(result["roc_auc"])
    )
    return result


def evaluate_cross_validation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs stratified 5-fold cross-validation on training data.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    scores = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, return_train_score=False)

    cv_results = {}
    for metric in scoring.keys():
        test_scores = scores[f"test_{metric}"]
        cv_results[metric] = {
            "mean": round(float(np.mean(test_scores)), 4),
            "std": round(float(np.std(test_scores)), 4),
            "scores": [round(float(s), 4) for s in test_scores]
        }

    return cv_results


def compute_permutation_feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Calculates model-agnostic permutation feature importance on held-out test data.
    Avoids impurity-based tree importance bias for continuous high-cardinality features.
    """
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc"
    )

    importance_dict = {}
    for i, col in enumerate(feature_names):
        importance_dict[col] = {
            "importance_mean": round(float(perm.importances_mean[i]), 4),
            "importance_std": round(float(perm.importances_std[i]), 4)
        }

    # Sorted by mean importance descending
    sorted_importance = dict(sorted(
        importance_dict.items(),
        key=lambda item: item[1]["importance_mean"],
        reverse=True
    ))

    return sorted_importance


def compute_learning_curve_data(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Generates learning curves to diagnose bias (underfitting) vs variance (overfitting).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=skf,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="roc_auc",
        n_jobs=-1
    )

    return {
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": [round(float(s), 4) for s in np.mean(train_scores, axis=1)],
        "train_scores_std": [round(float(s), 4) for s in np.std(train_scores, axis=1)],
        "val_scores_mean": [round(float(s), 4) for s in np.mean(val_scores, axis=1)],
        "val_scores_std": [round(float(s), 4) for s in np.std(val_scores, axis=1)]
    }
