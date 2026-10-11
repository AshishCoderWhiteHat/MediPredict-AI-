"""
MediPredict AI - Data Preprocessing Module
Author: MediPredict AI Team
Description: Leakage-free preprocessing with Scikit-learn Pipeline, handling physiological zeros
             via median imputation and feature scaling.
"""

import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.features.build_features import (
    FEATURE_NAMES,
    TARGET_NAME,
    SUSPICIOUS_ZERO_COLUMNS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PhysiologicalZeroToNaN(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that replaces biologically implausible zero values
    with np.nan for designated clinical features (e.g., Glucose, Blood Pressure, Insulin, BMI).
    """

    def __init__(self, zero_columns: Optional[List[str]] = None):
        self.zero_columns = zero_columns if zero_columns is not None else SUSPICIOUS_ZERO_COLUMNS
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = list(FEATURE_NAMES)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            for col in self.zero_columns:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].replace(0, np.nan)
            return X_copy
        else:
            # Numpy array: map column names based on fitted feature order
            X_arr = np.array(X, dtype=float, copy=True)
            for col in self.zero_columns:
                if col in self.feature_names_in_:
                    col_idx = self.feature_names_in_.index(col)
                    X_arr[:, col_idx] = np.where(X_arr[:, col_idx] == 0, np.nan, X_arr[:, col_idx])
            return X_arr

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.feature_names_in_


def create_preprocessor_pipeline(with_scaling: bool = True) -> Pipeline:
    """
    Creates a robust Scikit-learn preprocessing pipeline:
    1. PhysiologicalZeroToNaN: Converts clinical zeroes to NaN.
    2. SimpleImputer: Imputes NaNs using median learned from the training set.
    3. StandardScaler (optional): Standardizes features to mean=0, std=1.
    """
    steps = [
        ("zero_to_nan", PhysiologicalZeroToNaN(SUSPICIOUS_ZERO_COLUMNS)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if with_scaling:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split ensuring class balance preservation
    and preventing data leakage.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target column '{TARGET_NAME}' not found in dataframe.")

    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_NAME].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(
        "Split data with random_state=%d, test_size=%.2f. Train: %d records, Test: %d records.",
        random_state,
        test_size,
        len(X_train),
        len(X_test)
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    df, _ = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    preprocessor = create_preprocessor_pipeline(with_scaling=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    print(f"X_train original shape: {X_train.shape}, processed: {X_train_proc.shape}")
    print(f"X_test original shape: {X_test.shape}, processed: {X_test_proc.shape}")
    print(f"Imputer medians: {preprocessor.named_steps['imputer'].statistics_}")
