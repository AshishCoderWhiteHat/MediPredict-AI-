#!/usr/bin/env bash
set -e

echo "========================================================"
echo "      MediPredict AI - Quickstart Launcher (Unix/Mac)   "
echo "========================================================"

echo "[1/4] Running automated tests with Pytest..."
python3 -m pytest -v

echo "[2/4] Verifying model training artifacts..."
if [ ! -f "models/medipredict_best_model.joblib" ]; then
    echo "Model artifacts not found. Starting training pipeline..."
    python3 -m src.training.train
else
    echo "Found existing trained model artifact."
fi

echo "[3/4] Launching FastAPI REST Server in background (Port 8000)..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "[4/4] Launching Streamlit Clinical Dashboard (Port 8501)..."
streamlit run app.py
