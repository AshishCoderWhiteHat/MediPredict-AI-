"""
MediPredict AI - Data Ingestion Module
Author: MediPredict AI Team
Description: Secure and reproducible data loading with SHA-256 verification and automatic fallback.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expected canonical columns for the Pima Indians Diabetes dataset
EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

PRIMARY_DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
FALLBACK_DATASET_URL = "https://gist.githubusercontent.com/ktisha/c21e73a1a9f3f88c471a/raw/89523b1404166582d6f6d19b4e726576d1d44467/pima-data.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"


def calculate_file_hash(file_path: Path) -> str:
    """Computes SHA-256 checksum for provenance tracking."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_dataset(target_path: Path) -> Path:
    """
    Downloads dataset from verified public repository if not locally present.
    Ensures directory structure exists and raw data is saved without modification.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [PRIMARY_DATASET_URL, FALLBACK_DATASET_URL]

    for url in urls:
        try:
            logger.info("Attempting to download dataset from %s", url)
            headers = {"User-Agent": "MediPredictAI-Educational-System/1.0"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            with open(target_path, "wb") as f:
                f.write(response.content)

            logger.info("Successfully saved raw dataset to %s", target_path)
            return target_path
        except Exception as exc:
            logger.warning("Failed to download from %s: %s", url, exc)

    raise RuntimeError(
        f"Unable to download dataset from remote repositories. "
        f"Please manually download the Pima Indians Diabetes CSV and place it at: {target_path}"
    )


def load_raw_data(data_path: Optional[str or Path] = None, auto_download: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Loads the raw dataset, optionally downloading if missing.
    
    Returns:
        Tuple[pd.DataFrame, str]: The loaded DataFrame and its SHA-256 fingerprint.
    """
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    if not path.exists():
        if auto_download:
            logger.info("Dataset not found at %s. Triggering automatic download...", path)
            download_dataset(path)
        else:
            raise FileNotFoundError(f"Raw data file not found at: {path}")

    # Compute SHA256 fingerprint
    data_hash = calculate_file_hash(path)
    logger.info("Dataset fingerprint (SHA-256): %s", data_hash)

    # Read CSV
    df = pd.read_csv(path)
    logger.info("Loaded raw dataset: %d rows, %d columns", df.shape[0], df.shape[1])

    # Validate columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset at {path} is missing expected columns: {missing_cols}")

    return df, data_hash


if __name__ == "__main__":
    df, sha = load_raw_data()
    print(f"Data successfully loaded. Shape: {df.shape}, Hash: {sha}")
