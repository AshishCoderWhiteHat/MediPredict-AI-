"""
MediPredict AI - Data Validation Module
Author: MediPredict AI Team
Description: Validates clinical schema integrity, identifies suspicious physiological zeros,
             detects duplicates, and produces structured data quality audit reports.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.features.build_features import (
    FEATURE_NAMES,
    TARGET_NAME,
    SUSPICIOUS_ZERO_COLUMNS,
    FEATURE_SCHEMA
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when critical dataset validation checks fail."""
    pass


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, List[str]]:
    """
    Validates that the dataframe contains all expected feature columns and the target variable.
    
    Returns:
        Tuple[bool, List[str]]: Success status and list of validation issue descriptions.
    """
    issues = []
    missing_features = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_features:
        issues.append(f"Missing required feature columns: {missing_features}")

    if require_target and TARGET_NAME not in df.columns:
        issues.append(f"Missing required target column '{TARGET_NAME}'")

    return (len(issues) == 0, issues)


def inspect_suspicious_zeros(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyzes columns where a value of 0 is physiologically implausible in living patients.
    Records count and percentage of suspicious values.
    """
    zero_report = {}
    total_rows = len(df)
    for col in SUSPICIOUS_ZERO_COLUMNS:
        if col in df.columns:
            zero_count = int((df[col] == 0).sum())
            pct = (zero_count / total_rows * 100) if total_rows > 0 else 0.0
            zero_report[col] = {
                "zero_count": zero_count,
                "percentage": round(pct, 2),
                "is_suspicious": True,
                "clinical_reason": FEATURE_SCHEMA[col]["clinical_context"]
            }
    return zero_report


def inspect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Detects exact duplicate records."""
    dup_count = int(df.duplicated().sum())
    return {
        "duplicate_rows": dup_count,
        "duplicate_percentage": round((dup_count / len(df) * 100), 2) if len(df) > 0 else 0.0
    }


def validate_target_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates the binary target distribution."""
    if TARGET_NAME not in df.columns:
        return {"error": f"Target column '{TARGET_NAME}' absent."}

    target_series = df[TARGET_NAME].dropna()
    unique_vals = set(target_series.unique())

    if not unique_vals.issubset({0, 1}):
        raise DataValidationError(f"Target column contains non-binary values: {unique_vals}")

    counts = target_series.value_counts().to_dict()
    total = len(target_series)
    neg_count = int(counts.get(0, 0))
    pos_count = int(counts.get(1, 0))

    return {
        "total_records": total,
        "negative_class_0_count": neg_count,
        "positive_class_1_count": pos_count,
        "positive_ratio": round(pos_count / total, 4) if total > 0 else 0.0,
        "class_balance": f"{round((neg_count / total) * 100, 1)}% vs {round((pos_count / total) * 100, 1)}%"
    }


def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compiles a comprehensive data quality audit report without altering raw records.
    """
    is_valid_schema, schema_issues = validate_schema(df, require_target=(TARGET_NAME in df.columns))
    dup_stats = inspect_duplicates(df)
    suspicious_zeros = inspect_suspicious_zeros(df)
    target_stats = validate_target_distribution(df) if TARGET_NAME in df.columns else None

    # Null analysis
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(sum(null_counts.values()))

    # Basic numeric range check
    range_anomalies = {}
    for col in FEATURE_NAMES:
        if col in df.columns:
            cfg = FEATURE_SCHEMA[col]
            negatives = int((df[col] < 0).sum())
            extreme_highs = int((df[col] > cfg["max_value"] * 1.5).sum())
            if negatives > 0 or extreme_highs > 0:
                range_anomalies[col] = {
                    "negative_values": negatives,
                    "extreme_outliers": extreme_highs
                }

    report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "is_schema_valid": is_valid_schema,
        "schema_issues": schema_issues,
        "total_nulls": total_nulls,
        "null_counts": {k: int(v) for k, v in null_counts.items()},
        "duplicate_summary": dup_stats,
        "suspicious_zero_summary": suspicious_zeros,
        "range_anomalies": range_anomalies,
        "target_summary": target_stats
    }

    logger.info(
        "Data quality audit completed: %d rows, %d cols, %d duplicates, %d explicit nulls.",
        report["total_rows"],
        report["total_columns"],
        dup_stats["duplicate_rows"],
        total_nulls
    )
    return report


if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    df, _ = load_raw_data()
    report = generate_data_quality_report(df)
    import pprint
    pprint.pprint(report)
