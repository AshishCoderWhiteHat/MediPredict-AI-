"""
MediPredict AI - Unsupervised Learning & Clustering Module
Author: MediPredict AI Team
Description: Exploratory K-Means clustering and Principal Component Analysis (PCA) projection.
             Evaluates cluster inertia, silhouette scores, and feature groupings strictly
             WITHOUT using target labels during fitting.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.features.build_features import FEATURE_NAMES, TARGET_NAME
from src.data.preprocess_data import create_preprocessor_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_kmeans_analysis(
    df: pd.DataFrame,
    k_range: range = range(2, 9),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs K-Means clustering across candidate values of k:
    1. Imputes and scales features (no target leakage).
    2. Computes inertia for Elbow plot.
    3. Computes Silhouette scores.
    4. Fits optimal / default k=2 cluster model.
    5. Projects high-dimensional feature space onto 2 principal components via PCA.
    """
    if len(df) < 10:
        return {
            "error": "Insufficient data to perform stable K-Means clustering (need >= 10 samples)."
        }

    X = df[[col for col in FEATURE_NAMES if col in df.columns]].copy()

    # Preprocess: Median Imputation + StandardScaler
    preprocessor = create_preprocessor_pipeline(with_scaling=True)
    X_scaled = preprocessor.fit_transform(X)

    inertias = []
    silhouette_scores_list = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(round(float(kmeans.inertia_), 2))

        try:
            sil = float(silhouette_score(X_scaled, labels))
            silhouette_scores_list.append(round(sil, 4))
        except Exception:
            silhouette_scores_list.append(0.0)

    # Fit default k=2 (exploratory binary comparison)
    best_k = 2
    kmeans_k2 = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    cluster_labels_k2 = kmeans_k2.fit_predict(X_scaled)

    # 2D PCA Projection
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    var_explained = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

    pca_df = pd.DataFrame({
        "PCA_1": pca_coords[:, 0],
        "PCA_2": pca_coords[:, 1],
        "Cluster": [f"Cluster {c}" for c in cluster_labels_k2]
    })

    if TARGET_NAME in df.columns:
        pca_df["True_Outcome"] = df[TARGET_NAME].values

    # Cluster feature centers (in original scale)
    imputer = preprocessor.named_steps["imputer"]
    scaler = preprocessor.named_steps["scaler"]
    cluster_centers_orig = scaler.inverse_transform(kmeans_k2.cluster_centers_)

    centers_dict = {}
    for i in range(best_k):
        centers_dict[f"Cluster {i}"] = {
            feat: round(float(val), 2)
            for feat, val in zip(FEATURE_NAMES, cluster_centers_orig[i])
        }

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "best_k": best_k,
        "explained_variance_ratio": var_explained,
        "total_variance_explained": round(sum(var_explained), 2),
        "pca_data": pca_df.to_dict(orient="records"),
        "cluster_centers": centers_dict,
        "pca_components": {
            f"PC_{i+1}": {feat: round(float(weight), 4) for feat, weight in zip(FEATURE_NAMES, comp)}
            for i, comp in enumerate(pca.components_)
        },
        "disclaimer": (
            "Unsupervised K-Means clustering partitions patients solely based on Euclidean distance in "
            "standardized feature space. It is strictly exploratory and MUST NOT be interpreted as a clinical diagnosis."
        )
    }
