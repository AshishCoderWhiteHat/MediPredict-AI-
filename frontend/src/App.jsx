import React, { useState, useEffect } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import { api } from './services/api';
import { Sidebar } from './components/Sidebar';
import { Topbar } from './components/Topbar';
import { HomePage } from './pages/HomePage';
import { DashboardPage } from './pages/DashboardPage';
import { AssessmentPage } from './pages/AssessmentPage';
import { ResultsPage } from './pages/ResultsPage';
import { ReportsPage } from './pages/ReportsPage';
import { InsightsPage } from './pages/InsightsPage';
import { ModelLabPage } from './pages/ModelLabPage';
import { MonitoringPage } from './pages/MonitoringPage';

const LOCAL_STORAGE_REPORTS_KEY = 'medipredict_saved_reports';

function MainApp() {
  const [currentPage, setCurrentPage] = useState('home');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('Logistic Regression');

  // Prediction State
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [submittedPatientData, setSubmittedPatientData] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState('');

  // Persisted Reports State (localStorage)
  const [savedReports, setSavedReports] = useState(() => {
    try {
      const data = localStorage.getItem(LOCAL_STORAGE_REPORTS_KEY);
      return data ? JSON.parse(data) : [];
    } catch {
      return [];
    }
  });

  const syncReports = updated => {
    setSavedReports(updated);
    try {
      localStorage.setItem(LOCAL_STORAGE_REPORTS_KEY, JSON.stringify(updated));
    } catch (e) {
      console.warn('Failed to save to localStorage:', e);
    }
  };

  const handleSaveReport = newReport => {
    const updated = [newReport, ...savedReports.filter(r => r.id !== newReport.id)];
    syncReports(updated);
  };

  const handleDeleteReport = id => {
    const updated = savedReports.filter(r => r.id !== id);
    syncReports(updated);
  };

  const handleClearReports = () => {
    if (window.confirm('Are you sure you want to delete all saved reports? This cannot be undone.')) {
      syncReports([]);
    }
  };

  // Initial Data Fetch
  const fetchSystemData = async () => {
    setLoading(true);
    try {
      const [health, info, models] = await Promise.allSettled([
        api.getHealth(),
        api.getModelInfo(),
        api.getAvailableModels(),
      ]);

      if (health.status === 'fulfilled') {
        setApiOnline(health.value.status === 'healthy' || health.value.model_loaded === true);
      } else {
        setApiOnline(false);
      }

      if (info.status === 'fulfilled') {
        setModelInfo(info.value);
        if (info.value.selected_model_name) {
          setSelectedModel(info.value.selected_model_name);
        }
      }

      if (models.status === 'fulfilled' && models.value.models) {
        setAvailableModels(models.value.models);
      }
    } catch (err) {
      console.error('System initialization failed:', err);
      setApiOnline(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemData();
  }, []);

  // Handle Assessment Prediction Submission
  const handleRunPrediction = async patientPayload => {
    setPredictionLoading(true);
    setPredictionError('');
    try {
      const result = await api.predictRisk(patientPayload, selectedModel);
      setCurrentPrediction(result);
      setSubmittedPatientData(patientPayload);
      setCurrentPage('results');
    } catch (err) {
      setPredictionError(err.message || 'Prediction execution failed. Please verify the input values.');
    } finally {
      setPredictionLoading(false);
    }
  };

  const activeChampion = modelInfo?.selected_model_name || 'Logistic Regression';
  const testAuc = modelInfo?.metrics_summary?.roc_auc
    ? `${(Number(modelInfo.metrics_summary.roc_auc) * 100).toFixed(1)}%`
    : '81.3%';

  return (
    <div className="app-shell">
      {/* Sidebar Navigation */}
      <Sidebar
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        apiOnline={apiOnline}
        activeModel={activeChampion}
        savedReportsCount={savedReports.length}
      />

      {/* Main App Layout */}
      <div className="main-wrapper">
        <Topbar
          currentPage={currentPage}
          onOpenSidebar={() => setSidebarOpen(true)}
          onNavigate={setCurrentPage}
          apiOnline={apiOnline}
        />

        <main className="page-content">
          {currentPage === 'home' && (
            <HomePage
              onNavigate={setCurrentPage}
              championModel={activeChampion}
              testAuc={testAuc}
            />
          )}

          {currentPage === 'dashboard' && (
            <DashboardPage
              onNavigate={setCurrentPage}
              modelInfo={modelInfo}
              onRefresh={fetchSystemData}
              loading={loading}
              latestAssessment={currentPrediction || savedReports[0] || null}
              savedReports={savedReports}
            />
          )}

          {currentPage === 'assessment' && (
            <AssessmentPage
              availableModels={availableModels}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
              onSubmitPrediction={handleRunPrediction}
              loading={predictionLoading}
              apiOnline={apiOnline}
              errorMessage={predictionError}
              onClearError={() => setPredictionError('')}
            />
          )}

          {currentPage === 'results' && (
            <ResultsPage
              prediction={currentPrediction}
              patientData={submittedPatientData}
              onNavigate={setCurrentPage}
              onSaveReport={handleSaveReport}
              isSaved={savedReports.some(r => r.id === currentPrediction?.id)}
            />
          )}

          {currentPage === 'reports' && (
            <ReportsPage
              savedReports={savedReports}
              onDeleteReport={handleDeleteReport}
              onClearReports={handleClearReports}
              onNavigate={setCurrentPage}
            />
          )}

          {currentPage === 'insights' && <InsightsPage />}

          {currentPage === 'models' && (
            <ModelLabPage championModel={activeChampion} />
          )}

          {currentPage === 'monitoring' && <MonitoringPage />}
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <MainApp />
    </ThemeProvider>
  );
}
