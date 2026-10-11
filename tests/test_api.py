"""
MediPredict AI - Tests for FastAPI REST Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_endpoint(client):
    """Verifies that the root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MediPredict AI API"
    assert "disclaimer" in data


def test_health_endpoint(client):
    """Verifies system health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "setup_required")
    assert "uptime_seconds" in data
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    """Verifies model metadata retrieval endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "selected_model_name" in data
    assert "metrics_summary" in data
    assert len(data["feature_names"]) == 8


def test_model_comparison_endpoint(client):
    """The React dashboard can load the saved model comparison report."""
    response = client.get("/model-comparison")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 3
    assert "Model" in data["models"][0]


def test_predict_single_endpoint(client):
    """Verifies single patient risk prediction endpoint."""
    payload = {
        "Pregnancies": 2,
        "Glucose": 142.0,
        "BloodPressure": 78.0,
        "SkinThickness": 28.0,
        "Insulin": 115.0,
        "BMI": 31.8,
        "DiabetesPedigreeFunction": 0.48,
        "Age": 38
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_category"] in ("Low Risk", "Moderate Risk", "High Risk")
    assert "explanation" in data


def test_predict_invalid_input_rejected(client):
    """Verifies that malformed input is rejected with HTTP 422."""
    payload = {
        "Pregnancies": 2,
        "Glucose": "invalid_glucose",
        "Age": -10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_endpoint(client):
    """Verifies batch inference endpoint."""
    payload = {
        "records": [
            {
                "patient_id": "P_001",
                "Pregnancies": 1,
                "Glucose": 90.0,
                "BloodPressure": 65.0,
                "SkinThickness": 20.0,
                "Insulin": 60.0,
                "BMI": 23.5,
                "DiabetesPedigreeFunction": 0.22,
                "Age": 25
            },
            {
                "patient_id": "P_002",
                "Pregnancies": 5,
                "Glucose": 175.0,
                "BloodPressure": 85.0,
                "SkinThickness": 35.0,
                "Insulin": 210.0,
                "BMI": 38.2,
                "DiabetesPedigreeFunction": 0.75,
                "Age": 52
            }
        ]
    }
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records_received"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["patient_id"] == "P_001"
    assert data["predictions"][1]["patient_id"] == "P_002"


def test_models_endpoint_lists_available_algorithms(client):
    response = client.get("/models")
    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert "Logistic Regression" in payload["models"]



def test_predict_uses_selected_model(client):
    payload = {
        "Pregnancies": 2, "Glucose": 142.0, "BloodPressure": 78.0,
        "SkinThickness": 28.0, "Insulin": 115.0, "BMI": 31.8,
        "DiabetesPedigreeFunction": 0.48, "Age": 38
    }
    for model_name in ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "XGBoost"]:
        response = client.post("/predict", params={"model_name": model_name}, json=payload)
        assert response.status_code == 200
        assert response.json()["model_name"] == model_name

