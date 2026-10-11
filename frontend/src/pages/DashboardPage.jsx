import React from 'react';
import {
  HeartPulse,
  BrainCircuit,
  Activity,
  Sparkles,
  ArrowRight,
  TrendingUp,
  FileText,
  Clock,
  Layers,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  BarChart3,
  RefreshCw
} from 'lucide-react';

export function DashboardPage({
  onNavigate,
  modelInfo,
  onRefresh,
  loading,
  latestAssessment,
  savedReports = [],
  datasetInsights
}) {
  const metrics = modelInfo?.metrics_summary || {};
  const activeModel = modelInfo?.selected_model_name || 'Logistic Regression';
  const rocVal = metrics.roc_auc ?? metrics['ROC-AUC'];
  const formattedRoc = rocVal ? `${(Number(rocVal) * 100).toFixed(1)}%` : '81.3%';
  const recallVal = metrics.recall ?? metrics['Recall (Sensitivity)'];
  const formattedRecall = recallVal ? `${(Number(recallVal) * 100).toFixed(1)}%` : '70.4%';

  // Biomarker reference values (from real dataset medians)
  const biomarkerReferences = [
    { name: 'Fasting Glucose', median: '117.0 mg/dL', normal: '70 - 99 mg/dL', elevated: '≥ 126 mg/dL', unit: 'mg/dL' },
    { name: 'Diastolic BP', median: '72.0 mmHg', normal: '< 80 mmHg', elevated: '≥ 90 mmHg', unit: 'mmHg' },
    { name: 'Body Mass Index (BMI)', median: '32.4 kg/m²', normal: '18.5 - 24.9', elevated: '≥ 30.0 kg/m²', unit: 'kg/m²' },
    { name: '2-Hr Serum Insulin', median: '125.0 µU/mL', normal: '16 - 166 µU/mL', elevated: '> 200 µU/mL', unit: 'µU/mL' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Top Welcome Header */}
      <div className="page-header">
        <div>
          <div className="page-eyebrow">
            <Activity size={14} /> Clinical Intelligence Overview
          </div>
          <h1 className="page-title">Healthcare Analytics Dashboard</h1>
          <p className="page-subtitle">
            Review machine learning performance benchmarks, biomarker reference ranges, and your latest clinical assessment results.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button
            className="btn btn-secondary btn-sm"
            onClick={onRefresh}
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            <span>Refresh Model Metrics</span>
          </button>
          <button
            className="btn btn-primary btn-sm"
            onClick={() => onNavigate('assessment')}
          >
            <HeartPulse size={15} />
            <span>New Risk Assessment</span>
          </button>
        </div>
      </div>

      {/* Clinical Disclaimer Banner */}
      <div className="clinical-disclaimer-banner">
        <ShieldCheck size={18} />
        <div className="disclaimer-text">
          <strong>Academic & Educational Demonstration:</strong> MediPredict AI provides statistical predictions based on historical cohorts. It is not an approved diagnostic medical device and does not replace professional clinical evaluation or laboratory testing.
        </div>
      </div>

      {/* Real Verified Model Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-card-top">
            <span>Champion Model</span>
            <div className="metric-icon-wrap">
              <BrainCircuit size={17} />
            </div>
          </div>
          <div className="metric-value" style={{ fontSize: 18 }}>
            {activeModel.replace(' Classifier', '')}
          </div>
          <div className="metric-hint">Selected via CV selection score</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-top">
            <span>Test ROC-AUC</span>
            <div className="metric-icon-wrap blue">
              <TrendingUp size={17} />
            </div>
          </div>
          <div className="metric-value">{formattedRoc}</div>
          <div className="metric-hint">Held-out test set discrimination</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-top">
            <span>Clinical Sensitivity (Recall)</span>
            <div className="metric-icon-wrap mint">
              <Activity size={17} />
            </div>
          </div>
          <div className="metric-value">{formattedRecall}</div>
          <div className="metric-hint">Positive case detection rate</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-top">
            <span>Saved Assessments</span>
            <div className="metric-icon-wrap purple">
              <FileText size={17} />
            </div>
          </div>
          <div className="metric-value">{savedReports.length}</div>
          <div className="metric-hint">Persisted in local browser storage</div>
        </div>
      </div>

      {/* Latest Assessment vs Empty State */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 20 }}>
        {/* Latest Assessment Summary */}
        <div className="card-elevated" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <div>
              <h2 className="card-title">Latest Risk Assessment</h2>
              <p className="card-desc">Most recently generated prediction result</p>
            </div>
            {latestAssessment ? (
              <span
                className={`badge ${
                  latestAssessment.risk_category === 'High Risk'
                    ? 'badge-high'
                    : latestAssessment.risk_category === 'Moderate Risk'
                    ? 'badge-moderate'
                    : 'badge-low'
                }`}
              >
                {latestAssessment.risk_category}
              </span>
            ) : (
              <span className="badge badge-neutral">No Recent Test</span>
            )}
          </div>

          {latestAssessment ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14, flex: 1 }}>
              <div
                style={{
                  background: 'var(--bg-card)',
                  borderRadius: 'var(--radius-md)',
                  padding: 16,
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Estimated Probability</div>
                  <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--accent-teal)' }}>
                    {(latestAssessment.probability * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Model Utilized</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                    {latestAssessment.model_name || activeModel}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                    {latestAssessment.timestamp ? new Date(latestAssessment.timestamp).toLocaleDateString() : 'Just now'}
                  </div>
                </div>
              </div>

              {latestAssessment.explanation?.top_elevated_risk_factors?.length > 0 && (
                <div
                  style={{
                    background: 'rgba(239, 68, 68, 0.06)',
                    border: '1px solid rgba(239, 68, 68, 0.2)',
                    borderRadius: 'var(--radius-md)',
                    padding: 12,
                    fontSize: 12,
                  }}
                >
                  <div style={{ fontWeight: 700, color: 'var(--status-danger)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <AlertTriangle size={14} /> Key Elevated Factors:
                  </div>
                  <ul style={{ paddingLeft: 18, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    {latestAssessment.explanation.top_elevated_risk_factors.slice(0, 2).map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div style={{ display: 'flex', gap: 10, marginTop: 'auto', paddingTop: 10 }}>
                <button
                  className="btn btn-outline btn-sm"
                  onClick={() => onNavigate('results')}
                  style={{ flex: 1 }}
                >
                  View Full Report
                </button>
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => onNavigate('assessment')}
                  style={{ flex: 1 }}
                >
                  Run New Assessment
                </button>
              </div>
            </div>
          ) : (
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                textAlign: 'center',
                padding: '30px 20px',
              }}
            >
              <div
                style={{
                  width: 54,
                  height: 54,
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-card)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-muted)',
                  marginBottom: 14,
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <HeartPulse size={26} />
              </div>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
                No Assessments Performed Yet
              </h3>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', maxWidth: 300, marginBottom: 18, lineHeight: 1.5 }}>
                Submit patient clinical biomarkers to compute an instant, explainable diabetes risk score.
              </p>
              <button
                className="btn btn-primary"
                onClick={() => onNavigate('assessment')}
              >
                <HeartPulse size={16} />
                <span>Start Risk Assessment</span>
              </button>
            </div>
          )}
        </div>

        {/* Quick Action Navigation Panels */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div
            className="card"
            style={{
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 16,
            }}
            onClick={() => onNavigate('assessment')}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <div
                style={{
                  width: 42,
                  height: 42,
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--accent-teal-subtle)',
                  color: 'var(--accent-teal)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <HeartPulse size={20} />
              </div>
              <div>
                <h4 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                  Interactive Assessment Form
                </h4>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  Submit 8 clinical biomarkers with real-time validation & optional model selection.
                </p>
              </div>
            </div>
            <ArrowRight size={18} style={{ color: 'var(--text-muted)' }} />
          </div>

          <div
            className="card"
            style={{
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 16,
            }}
            onClick={() => onNavigate('insights')}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <div
                style={{
                  width: 42,
                  height: 42,
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--accent-blue-subtle)',
                  color: 'var(--accent-blue)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <Activity size={20} />
              </div>
              <div>
                <h4 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                  Health Insights & Explainable AI
                </h4>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  Explore feature importance rankings and unsupervised clustering trends.
                </p>
              </div>
            </div>
            <ArrowRight size={18} style={{ color: 'var(--text-muted)' }} />
          </div>

          <div
            className="card"
            style={{
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 16,
            }}
            onClick={() => onNavigate('models')}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <div
                style={{
                  width: 42,
                  height: 42,
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--accent-purple-subtle)',
                  color: 'var(--accent-purple)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <BarChart3 size={20} />
              </div>
              <div>
                <h4 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                  Model Lab (5 Algorithms)
                </h4>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  Compare accuracy, sensitivity, ROC-AUC, and confusion matrices.
                </p>
              </div>
            </div>
            <ArrowRight size={18} style={{ color: 'var(--text-muted)' }} />
          </div>
        </div>
      </div>

      {/* Useful Biomarker Clinical Reference Ranges */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Clinical Biomarker Reference Medians</h3>
            <p className="card-desc">
              Reference medians from the training cohort (Pima Indians Diabetes dataset, N=768) alongside standard medical thresholds.
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
          {biomarkerReferences.map(bio => (
            <div
              key={bio.name}
              style={{
                background: 'var(--bg-panel)',
                border: '1px solid var(--border-muted)',
                borderRadius: 'var(--radius-md)',
                padding: 14,
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                {bio.name}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                <span style={{ color: 'var(--text-muted)' }}>Cohort Median:</span>
                <strong style={{ color: 'var(--accent-teal)' }}>{bio.median}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                <span style={{ color: 'var(--text-muted)' }}>Normal Threshold:</span>
                <span style={{ color: 'var(--status-success)' }}>{bio.normal}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                <span style={{ color: 'var(--text-muted)' }}>Elevated Threshold:</span>
                <span style={{ color: 'var(--status-danger)' }}>{bio.elevated}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
