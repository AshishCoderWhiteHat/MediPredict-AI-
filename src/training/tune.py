"""
MediPredict AI - Hyperparameter Tuning Module
Author: MediPredict AI Team
Description: Efficient hyperparameter search across linear and ensemble models
             using Stratified 5-Fold Cross Validation without held-out test contamination.
"""

import logging
from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from src.data.preprocess_data import create_preprocessor_pipeline

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tune_logistic_regression(X_train, y_train, cv_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tunes Logistic Regression (L1 and L2 regularization) over standardized features."""
    logger.info("Tuning Logistic Regression...")
    preprocessor = create_preprocessor_pipeline(with_scaling=True)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="liblinear", max_iter=1000, random_state=random_state))
    ])

    param_grid = {
        "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"]
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info("Best LR Params: %s | Best ROC-AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, {"best_params": grid.best_params_, "best_cv_score": round(grid.best_score_, 4)}


def tune_decision_tree(X_train, y_train, cv_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tunes baseline Decision Tree with pruning hyperparameters."""
    logger.info("Tuning Decision Tree Classifier...")
    preprocessor = create_preprocessor_pipeline(with_scaling=False)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=random_state))
    ])

    param_grid = {
        "classifier__max_depth": [3, 4, 5, 7, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"]
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info("Best DT Params: %s | Best ROC-AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, {"best_params": grid.best_params_, "best_cv_score": round(grid.best_score_, 4)}


def tune_random_forest(X_train, y_train, cv_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tunes Random Forest Classifier across tree counts and tree depths."""
    logger.info("Tuning Random Forest Classifier...")
    preprocessor = create_preprocessor_pipeline(with_scaling=False)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=random_state))
    ])

    param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"]
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info("Best RF Params: %s | Best ROC-AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, {"best_params": grid.best_params_, "best_cv_score": round(grid.best_score_, 4)}


def tune_gradient_boosting(X_train, y_train, cv_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tunes Scikit-learn Gradient Boosting Classifier."""
    logger.info("Tuning Gradient Boosting Classifier...")
    preprocessor = create_preprocessor_pipeline(with_scaling=False)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=random_state))
    ])

    param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__learning_rate": [0.02, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4],
        "classifier__subsample": [0.8, 1.0]
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info("Best GB Params: %s | Best ROC-AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, {"best_params": grid.best_params_, "best_cv_score": round(grid.best_score_, 4)}



def tune_svm(X_train, y_train, cv_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tunes a probability-enabled Support Vector Machine on scaled features."""
    logger.info("Tuning Support Vector Machine (SVM)...")
    preprocessor = create_preprocessor_pipeline(with_scaling=True)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(probability=True, random_state=random_state))
    ])
    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__class_weight": [None, "balanced"]
    }
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    logger.info("Best SVM Params: %s | Best ROC-AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, {"best_params": grid.best_params_, "best_cv_score": round(grid.best_score_, 4)}


def tune_xgboost(X_train, y_train, cv_splits: int = 5, random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    """Tunes XGBoost Classifier if installed, else returns None."""
    if not HAS_XGBOOST:
        logger.warning("XGBoost is not installed. Skipping XGBoost tuning.")
        return None, {}

    logger.info("Tuning XGBoost Classifier...")
    preprocessor = create_preprocessor_pipeline(with_scaling=False)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=random_state
        ))
    ])

    param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__learning_rate": [0.02, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0]
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info("Best XGB Params: %s | Best ROC-AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, {"best_params": grid.best_params_, "best_cv_score": round(grid.best_score_, 4)}
