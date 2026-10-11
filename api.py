"""
MediPredict AI - Production FastAPI Inference Service
Author: MediPredict AI Team
Description: REST API for explainable disease risk prediction, health telemetry,
             batch processing, and model metadata inspection.
"""

import time
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from src.inference.schemas import (
    PatientInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse
)
from src.inference.predict import (
    predict_single,
    predict_batch_records,
    load_model_metadata,
    load_best_model,
    load_all_models,
    CLINICAL_DISCLAIMER
)
from src.monitoring.logging_utils import log_prediction_event, get_monitoring_summary
from src.monitoring.drift import evaluate_feature_drift
from src.data.load_data import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediPredict AI REST API",
    description="Explainable Medical Disease Risk Prediction & Machine Learning Evaluation Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()


@app.get("/", tags=["General"])
def root(request: Request):
    """Serves the React single page app if browser navigation, or API status overview for API clients."""
    from pathlib import Path
    accept_header = request.headers.get("accept", "")
    dist_index = Path(__file__).resolve().parent / "frontend" / "dist" / "index.html"
    if "text/html" in accept_header and dist_index.exists():
        return FileResponse(dist_index)
    return {
        "service": "MediPredict AI API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
        "disclaimer": CLINICAL_DISCLAIMER
    }


@app.get("/api", tags=["General"])
def api_status_overview():
    """API overview and documentation link."""
    return {
        "service": "MediPredict AI API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
        "disclaimer": CLINICAL_DISCLAIMER
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Returns system health, uptime, and model loading status."""
    uptime_seconds = round(time.time() - _START_TIME, 2)
    model_loaded = False
    model_name = "None"

    try:
        model = load_best_model()
        model_loaded = model is not None
        meta = load_model_metadata()
        model_name = meta.get("selected_model_name", "Loaded")
    except Exception as e:
        logger.warning("Model health check warning: %s", e)

    return {
        "status": "healthy" if model_loaded else "setup_required",
        "uptime_seconds": uptime_seconds,
        "model_loaded": model_loaded,
        "active_model": model_name
    }


@app.get("/models", tags=["Model Metadata"])
def get_available_models():
    """Return the trained models available for the prediction dropdown."""
    try:
        all_models = load_all_models()
        metadata = load_model_metadata()
        return {
            "models": list(all_models.keys()),
            "default_model": metadata.get("selected_model_name", next(iter(all_models), None))
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Metadata"])
def get_model_information():
    """Retrieves versioned metadata, training timestamp, and evaluated metrics."""
    try:
        meta = load_model_metadata()
        best_name = meta["selected_model_name"]
        best_metrics = meta["models_evaluated"][best_name]["test_metrics"]

        return ModelInfoResponse(
            project_name=meta.get("project", "MediPredict AI"),
            selected_model_name=best_name,
            training_timestamp=meta.get("timestamp", "Unknown"),
            dataset_hash=meta.get("dataset_hash", "Unknown"),
            feature_names=meta.get("feature_names", []),
            metrics_summary=best_metrics,
            environment=meta.get("environment", {}),
            disclaimer=CLINICAL_DISCLAIMER
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata artifact is not yet available. Please execute model training first."
        )
    except Exception as e:
        logger.error("Error retrieving model metadata: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving model metadata."
        )


@app.get("/model-comparison", tags=["Model Metadata"])
def get_model_comparison():
    """Return precomputed evaluation metrics for the dashboard."""
    from pathlib import Path
    import pandas as pd

    comparison_path = Path(__file__).resolve().parent / "reports" / "evaluation_summary.csv"
    if not comparison_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model comparison report is not available. Run model training first.",
        )
    try:
        records = pd.read_csv(comparison_path).replace({float("nan"): None}).to_dict(orient="records")
        return {"models": records}
    except Exception as exc:
        logger.error("Unable to load model comparison report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load model comparison metrics.",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_patient_risk(patient: PatientInput, model_name: str | None = None):
    """
    Predicts disease risk for an individual patient and provides local biomarker explanations.
    """
    start_time = time.time()
    try:
        result = predict_single(patient, model_name_override=model_name)
        latency_ms = (time.time() - start_time) * 1000.0

        # Log operational metrics asynchronously/in telemetry
        log_prediction_event(
            latency_ms=latency_ms,
            prediction=result["prediction"],
            probability=result["probability"],
            risk_category=result["risk_category"],
            model_name=result["model_name"],
            batch_size=1,
            status="success"
        )

        return PredictionResponse(**result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not available. Please train models before making predictions."
        )
    except Exception as e:
        logger.error("Inference failure: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to generate prediction. Verify input feature values and types."
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_patient_risk(batch_request: BatchPredictionRequest):
    """
    Executes batch inference across a list of patient records (maximum 500).
    """
    start_time = time.time()
    try:
        records_data = [r.model_dump() for r in batch_request.records]
        results, _ = predict_batch_records(records_data)
        latency_ms = (time.time() - start_time) * 1000.0

        meta = load_model_metadata()
        active_model = meta.get("selected_model_name", "Champion Model")

        # Telemetry
        log_prediction_event(
            latency_ms=latency_ms,
            prediction=-1,
            probability=-1.0,
            risk_category="Batch",
            model_name=active_model,
            batch_size=len(results),
            status="success"
        )

        return BatchPredictionResponse(
            total_records_received=len(batch_request.records),
            successful_predictions=len(results),
            failed_records=0,
            predictions=results,
            model_name=active_model,
            model_version="1.0.0",
            disclaimer=CLINICAL_DISCLAIMER
        )
    except Exception as e:
        logger.error("Batch inference failure: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/drift-report", tags=["Monitoring"])
def get_data_drift_report():
    """
    Computes Kolmogorov-Smirnov and PSI data drift against baseline dataset.
    """
    try:
        ref_df, _ = load_raw_data()
        # For demonstration, compare first half against second half or recent batches
        drift_results = evaluate_feature_drift(ref_df.iloc[:400], ref_df.iloc[400:])
        return drift_results
    except Exception as e:
        logger.error("Data drift calculation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute drift report."
        )


@app.get("/telemetry", tags=["Monitoring"])
def get_inference_telemetry():
    """Returns aggregate prediction volume, latency, and risk distributions."""
    return get_monitoring_summary()


@app.get("/feature-importance", tags=["Model Metadata"])
def get_feature_importance():
    """Return precomputed permutation and native feature importances across all models."""
    from pathlib import Path
    import json

    report_path = Path(__file__).resolve().parent / "reports" / "feature_importance.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature importance report not found. Run model training first.",
        )
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Unable to load feature importance report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load feature importance metrics.",
        )


@app.get("/evaluation-details", tags=["Model Metadata"])
def get_evaluation_details():
    """Return detailed evaluation metrics including confusion matrices, ROC curves, and PR curves."""
    from pathlib import Path
    import json

    report_path = Path(__file__).resolve().parent / "reports" / "evaluation_summary.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation details report not found. Run model training first.",
        )
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Unable to load evaluation details report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load detailed evaluation metrics.",
        )


@app.get("/dataset-insights", tags=["Data Analysis"])
def get_dataset_insights():
    """Return distribution summaries, medians, and biomarker ranges from baseline dataset."""
    try:
        df, _ = load_raw_data()
        feature_cols = [c for c in df.columns if c != "Outcome"]
        summary = {}
        for col in feature_cols:
            vals = df[col].dropna()
            summary[col] = {
                "mean": round(float(vals.mean()), 2),
                "std": round(float(vals.std()), 2),
                "median": round(float(vals.median()), 2),
                "min": round(float(vals.min()), 2),
                "max": round(float(vals.max()), 2),
                "q25": round(float(vals.quantile(0.25)), 2),
                "q75": round(float(vals.quantile(0.75)), 2),
            }
        
        outcome_counts = df["Outcome"].value_counts().to_dict() if "Outcome" in df.columns else {}
        return {
            "total_records": len(df),
            "feature_stats": summary,
            "target_distribution": {
                "negative": int(outcome_counts.get(0, 0)),
                "positive": int(outcome_counts.get(1, 0)),
                "positive_percentage": round(float(outcome_counts.get(1, 0) / len(df)) * 100, 1) if len(df) else 0.0
            }
        }
    except Exception as exc:
        logger.error("Error computing dataset insights: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to compute dataset insights.",
        )


@app.get("/clustering", tags=["Data Analysis"])
def get_clustering_analysis():
    """Return unsupervised K-Means clustering and 2D PCA projection results."""
    from src.features.unsupervised import run_kmeans_analysis
    try:
        df, _ = load_raw_data()
        results = run_kmeans_analysis(df)
        return results
    except Exception as exc:
        logger.error("Error computing clustering analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to compute clustering analysis.",
        )


# Mount built React frontend for unified full-stack single-port deployment (Render, Docker, Hugging Face)
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
if _DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")


