"""
MediPredict AI - Inference Monitoring & Latency Logger
Author: MediPredict AI Team
Description: Privacy-preserving aggregate metric logging for prediction counts,
             latency, risk tier distributions, and error tracking without recording raw patient PII.
"""

import os
import json
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "artifacts"
INFERENCE_LOG_PATH = LOGS_DIR / "inference_logs.jsonl"


def log_prediction_event(
    latency_ms: float,
    prediction: int,
    probability: float,
    risk_category: str,
    model_name: str,
    batch_size: int = 1,
    status: str = "success"
) -> None:
    """
    Logs an aggregate prediction event. Never stores patient names, IDs, or clinical measurements.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "latency_ms": round(latency_ms, 2),
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_category": risk_category,
        "model_name": model_name,
        "batch_size": batch_size,
        "status": status
    }

    try:
        with open(INFERENCE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning("Failed to record inference log event: %s", e)


def get_monitoring_summary() -> Dict[str, Any]:
    """
    Reads recorded prediction events to compile real operational telemetry metrics.
    """
    if not INFERENCE_LOG_PATH.exists():
        return {
            "total_requests": 0,
            "total_predictions": 0,
            "avg_latency_ms": 0.0,
            "risk_distribution": {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0},
            "recent_events": []
        }

    events = []
    with open(INFERENCE_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

    if not events:
        return {
            "total_requests": 0,
            "total_predictions": 0,
            "avg_latency_ms": 0.0,
            "risk_distribution": {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0},
            "recent_events": []
        }

    total_reqs = len(events)
    total_preds = sum(e.get("batch_size", 1) for e in events)
    latencies = [e["latency_ms"] for e in events if "latency_ms" in e]
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    risk_counts = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0}
    for e in events:
        r = e.get("risk_category")
        if r in risk_counts:
            risk_counts[r] += e.get("batch_size", 1)

    return {
        "total_requests": total_reqs,
        "total_predictions": total_preds,
        "avg_latency_ms": avg_lat,
        "risk_distribution": risk_counts,
        "recent_events": events[-10:]
    }
