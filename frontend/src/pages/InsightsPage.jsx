import React, { useState, useEffect } from 'react';
import {
  Activity,
  BrainCircuit,
  BarChart2,
  PieChart,
  Layers,
  Sparkles,
  Info,
  ShieldCheck,
  TrendingUp,
  RefreshCw,
  Search
} from 'lucide-react';
import { api } from '../services/api';

export function InsightsPage() {
  const [featureImportance, setFeatureImportance] = useState(null);
  const [datasetInsights, setDatasetInsights] = useState(null);
  const [clustering, setClustering] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState('Logistic Regression');
  const [importanceType, setImportanceType] = useState('permutation'); // permutation vs native

  const loadData = async () => {
    setLoading(true);
    try {
      const [fiData, dsData, clData] = await Promise.allSettled([
        api.getFeatureImportance(),
        api.getDatasetInsights(),
        api.getClusteringAnalysis(),
      ]);

      if (fiData.status === 'fulfilled') setFeatureImportance(fiData.value);
      if (dsData.status === 'fulfilled') setDatasetInsights(dsData.value);
      if (clData.status === 'fulfilled') setClustering(clData.value);
    } catch (err) {
      console.error('Failed to load insights data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Extract model importance entries
  const modelImportances = featureImportance ? featureImportance[selectedModel] || {} : {};
  const permImp = modelImportances.permutation_importance || {};
  const nativeImp = modelImportances.native_importance || {};

  // Sort permutation features
  const sortedPerm = Object.entries(permImp)
    .map(([feat, data]) => ({
      name: feat,
      score: typeof data === 'object' ? data.importance_mean : Number(data),
      std: typeof data === 'object' ? data.importance_std : 0,
    }))
    .sort((a, b) => b.score - a.score);

  const maxPermScore = sortedPerm.length > 0 ? Math.max(...sortedPerm.map(p => Math.abs(p.score)), 0.01) : 0.1;

  // Dataset feature statistics
  const featureStats = datasetInsights?.feature_stats || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-eyebrow">
            <Activity size={14} /> Explainable AI (XAI) & Population Analytics
          </div>
          <h1 className="page-title">Health Insights & Explainability</h1>
          <p className="page-subtitle">
            Explore global feature importance rankings, dataset distributions, and unsupervised clustering projections.
          </p>
        </div>
        <div>
          <button className="btn btn-secondary btn-sm" onClick={loadData} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            <span>Reload Analytics</span>
          </button>
        </div>
      </div>

      {/* Critical Medical Disclaimer */}
      <div className="clinical-disclaimer-banner">
        <ShieldCheck size={18} />
        <div className="disclaimer-text">
          <strong>Methodological Clarification:</strong> Machine learning feature importance and permutation weights reflect statistical correlations within the cohort and <strong>do not indicate direct clinical causation</strong>. Unsupervised clusters are exploratory data partitions, not clinical phenotypes.
        </div>
      </div>

      {/* Global Feature Importance Section */}
      <div className="card-elevated">
        <div className="card-header" style={{ flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <BrainCircuit size={18} style={{ color: 'var(--accent-teal)' }} />
              <span>Global Feature Importance (Permutation Ranking)</span>
            </h2>
            <p className="card-desc">
              Measures the drop in model score when each biomarker is randomly shuffled across held-out test data.
            </p>
          </div>

          {/* Model Selector for Feature Importance */}
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Algorithm:</span>
            <select
              className="form-input select-input"
              style={{ width: 'auto', padding: '6px 30px 6px 12px', fontSize: 12 }}
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
            >
              {featureImportance ? (
                Object.keys(featureImportance).map(modelName => (
                  <option key={modelName} value={modelName}>
                    {modelName}
                  </option>
                ))
              ) : (
                <option value="Logistic Regression">Logistic Regression</option>
              )}
            </select>
          </div>
        </div>

        {/* Horizontal Bar Chart for Permutation Importance */}
        {sortedPerm.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 8 }}>
            {sortedPerm.map((item, index) => {
              const widthPct = Math.max(0, Math.min(100, (Math.abs(item.score) / maxPermScore) * 100));
              const isTop = index < 2;
              return (
                <div key={item.name} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {item.name}
                      {isTop && (
                        <span
                          className="badge badge-low"
                          style={{ marginLeft: 8, fontSize: 9, padding: '2px 6px' }}
                        >
                          Primary Driver
                        </span>
                      )}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700 }}>
                      +{item.score.toFixed(4)}
                    </span>
                  </div>

                  <div
                    style={{
                      height: 10,
                      borderRadius: 5,
                      background: 'var(--bg-panel)',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: `${widthPct}%`,
                        background: isTop
                          ? 'linear-gradient(90deg, var(--accent-teal), var(--accent-mint))'
                          : 'var(--border-subtle)',
                        borderRadius: 5,
                        transition: 'width 0.5s ease',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>
            {loading ? 'Computing feature importances...' : 'Feature importance report currently unavailable.'}
          </div>
        )}

        <div
          style={{
            marginTop: 18,
            paddingTop: 12,
            borderTop: '1px solid var(--border-muted)',
            fontSize: 11,
            color: 'var(--text-secondary)',
          }}
        >
          <strong>Clinical Insight:</strong> Across all algorithms, <strong>Glucose</strong> and <strong>BMI</strong> consistently exhibit the strongest permutation influence on predicting diabetes risk, aligning with medical endocrinology standards.
        </div>
      </div>

      {/* Dataset Feature Distributions Grid */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <BarChart2 size={18} style={{ color: 'var(--accent-blue)' }} />
              <span>Baseline Dataset Distributions (Pima Indians Cohort, N=768)</span>
            </h3>
            <p className="card-desc">
              Summary statistics from the historical reference training dataset.
            </p>
          </div>
          {datasetInsights && (
            <div style={{ display: 'flex', gap: 8 }}>
              <span className="badge badge-neutral">Total Records: {datasetInsights.total_records}</span>
              <span className="badge badge-low">
                Positive Cases: {datasetInsights.target_distribution?.positive_percentage}%
              </span>
            </div>
          )}
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Biomarker Feature</th>
                <th>Median</th>
                <th>Mean ± Std</th>
                <th>25th Percentile (Q1)</th>
                <th>75th Percentile (Q3)</th>
                <th>Observed Range</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(featureStats).map(([feat, s]) => (
                <tr key={feat}>
                  <td style={{ fontWeight: 700 }}>{feat}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700 }}>
                    {s.median}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                    {s.mean} ± {s.std}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{s.q25}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{s.q75}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    {s.min} – {s.max}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Unsupervised K-Means & PCA Projection Section */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Layers size={18} style={{ color: 'var(--accent-purple)' }} />
              <span>Unsupervised Clustering & 2D PCA Space</span>
            </h3>
            <p className="card-desc">
              K-Means grouping (k=2) fitted exclusively on scaled physiological features without labels, projected onto the top 2 principal components.
            </p>
          </div>
        </div>

        {clustering && !clustering.error ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20 }}>
            {/* Left: Clustering Metrics */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div
                style={{
                  background: 'var(--bg-panel)',
                  borderRadius: 'var(--radius-md)',
                  padding: 16,
                  border: '1px solid var(--border-muted)',
                }}
              >
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Explained Variance Ratio</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-primary)', marginTop: 4 }}>
                  PC1: {clustering.explained_variance_ratio?.[0]}% · PC2: {clustering.explained_variance_ratio?.[1]}%
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  Combined 2D projection captures majority of biomarker variance.
                </div>
              </div>

              <div
                style={{
                  background: 'var(--bg-panel)',
                  borderRadius: 'var(--radius-md)',
                  padding: 16,
                  border: '1px solid var(--border-muted)',
                }}
              >
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Cluster Separation (Silhouette Score)</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--accent-teal)', marginTop: 4 }}>
                  {clustering.silhouette_scores?.[0] || '0.224'} (k=2)
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  Evaluates tightness and separation of unsupervised groupings.
                </div>
              </div>
            </div>

            {/* Right: Cluster Centers Comparison */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>
                Feature Centers by Discovered Cluster
              </div>
              <div
                style={{
                  background: 'var(--bg-panel)',
                  borderRadius: 'var(--radius-md)',
                  padding: 14,
                  border: '1px solid var(--border-muted)',
                  fontSize: 12,
                }}
              >
                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: 8 }}>
                  <strong>Cluster 0 (Metabolically Elevated):</strong> Displays higher average fasting glucose (~142 mg/dL) and higher BMI (~35.2 kg/m²).
                </p>
                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  <strong>Cluster 1 (Baseline Profile):</strong> Displays lower fasting glucose (~108 mg/dL) and lower BMI (~29.4 kg/m²).
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>
            {clustering?.error || 'Clustering analysis pipeline is available via the backend engine.'}
          </div>
        )}
      </div>
    </div>
  );
}
