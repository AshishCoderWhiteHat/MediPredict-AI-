/**
 * MediPredict AI - API Service Layer
 * Centralized HTTP client connecting the React frontend to the FastAPI backend.
 */

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

async function handleResponse(response) {
  if (!response.ok) {
    let errorMessage = `Request failed with status ${response.status}`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map(e => e.msg || JSON.stringify(e)).join(', ');
        }
      }
    } catch {
      // Fallback to default status text if not JSON
      errorMessage = response.statusText || errorMessage;
    }
    throw new Error(errorMessage);
  }
  return response.json();
}

export const api = {
  getBaseUrl() {
    return API_BASE;
  },

  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse(res);
  },

  async getModelInfo() {
    const res = await fetch(`${API_BASE}/model-info`);
    return handleResponse(res);
  },

  async getAvailableModels() {
    const res = await fetch(`${API_BASE}/models`);
    return handleResponse(res);
  },

  async getModelComparison() {
    const res = await fetch(`${API_BASE}/model-comparison`);
    return handleResponse(res);
  },

  async getFeatureImportance() {
    const res = await fetch(`${API_BASE}/feature-importance`);
    return handleResponse(res);
  },

  async getEvaluationDetails() {
    const res = await fetch(`${API_BASE}/evaluation-details`);
    return handleResponse(res);
  },

  async getDatasetInsights() {
    const res = await fetch(`${API_BASE}/dataset-insights`);
    return handleResponse(res);
  },

  async getClusteringAnalysis() {
    const res = await fetch(`${API_BASE}/clustering`);
    return handleResponse(res);
  },

  async getTelemetry() {
    const res = await fetch(`${API_BASE}/telemetry`);
    return handleResponse(res);
  },

  async getDriftReport() {
    const res = await fetch(`${API_BASE}/drift-report`);
    return handleResponse(res);
  },

  async predictRisk(patientData, modelName = null) {
    const url = modelName
      ? `${API_BASE}/predict?model_name=${encodeURIComponent(modelName)}`
      : `${API_BASE}/predict`;

    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patientData),
    });
    return handleResponse(res);
  },

  async predictBatch(records) {
    const res = await fetch(`${API_BASE}/predict-batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ records }),
    });
    return handleResponse(res);
  },
};
