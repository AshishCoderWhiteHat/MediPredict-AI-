import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  ShieldCheck,
  RefreshCw,
  Zap,
  Server,
  Layers,
  Info
} from 'lucide-react';
import { api } from '../services/api';

export function MonitoringPage() {
  const [telemetry, setTelemetry] = useState(null);
  const [driftReport, setDriftReport] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    setLoading(true);
    try {
      const [telemRes, driftRes] = await Promise.allSettled([
        api.getTelemetry(),
        api.getDriftReport(),
      ]);

      if (telemRes.status === 'fulfilled') {
        setTelemetry(telemRes.value);
      }
      if (driftRes.status === 'fulfilled') {
        setDriftReport(driftRes.value);
      }
    } catch (err) {
      console.error('Failed to load telemetry/drift:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const totalRequests = telemetry?.total_requests ?? 0;
  const avgLatency = telemetry?.avg_latency_ms ?? 0;
  const riskDist = telemetry?.risk_distribution || { 'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0 };
  const recentEvents = telemetry?.recent_events || [];

  const driftFeatures = driftReport?.features || driftReport || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-eyebrow">
            <LineChart size={14} /> Production Telemetry & Statistical Quality
          </div>
          <h1 className="page-title">Model Monitoring & Drift Telemetry</h1>
          <p className="page-subtitle">
            Inspect real-time inference latency logs, request volumes, and statistical covariate drift reports.
          </p>
        </div>
        <div>
          <button className="btn btn-secondary btn-sm" onClick={loadData} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            <span>Refresh Telemetry</span>
          </button>
        </div>
      </div>

      {/* Distinction Alert Banner */}
      <div className="clinical-disclaimer-banner">
        <Info size={18} />
        <div className="disclaimer-text">
          <strong>Architectural Note:</strong> The telemetry cards below track actual API inference calls executed since service startup. The <strong>Data Drift Report</strong> evaluates two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) between baseline and sample cohorts.
        </div>
      </div>

      {/* Operational Metrics Cards */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-card-top">
            <span>Inference Requests</span>
            <div className="metric-icon-wrap">
              <Server size={17} />
            </div>
          </div>
          <div className="metric-value">{totalRequests}</div>
          <div className="metric-hint">Inference calls logged</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-top">
            <span>Average Latency</span>
            <div className="metric-icon-wrap mint">
              <Zap size={17} />
            </div>
          </div>
          <div className="metric-value">{avgLatency} ms</div>
          <div className="metric-hint">End-to-end FastAPI response time</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-top">
            <span>Risk Tier Breakdown</span>
            <div className="metric-icon-wrap blue">
              <Activity size={17} />
            </div>
          </div>
          <div className="metric-value" style={{ fontSize: 16 }}>
            {riskDist['High Risk']} High · {riskDist['Moderate Risk']} Mod · {riskDist['Low Risk']} Low
          </div>
          <div className="metric-hint">Aggregate predictions by category</div>
        </div>
      </div>

      {/* Covariate Drift Report Table */}
      <div className="card-elevated">
        <div className="card-header">
          <div>
            <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Layers size={18} style={{ color: 'var(--accent-teal)' }} />
              <span>Feature Covariate Drift Evaluation (KS Test & PSI)</span>
            </h2>
            <p className="card-desc">
              Two-sample Kolmogorov-Smirnov test (α = 0.05) and Population Stability Index (PSI threshold: 0.20).
            </p>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Biomarker Feature</th>
                <th>KS Statistic</th>
                <th>p-Value</th>
                <th>Population Stability Index (PSI)</th>
                <th>Drift Status</th>
              </tr>
            </thead>
            <tbody>
              {typeof driftFeatures === 'object' && Object.keys(driftFeatures).length > 0 ? (
                Object.entries(driftFeatures).map(([feat, d]) => {
                  if (typeof d !== 'object' || d === null) return null;
                  const isDrift = d.drift_detected === true || (d.psi && d.psi >= 0.2);
                  return (
                    <tr key={feat}>
                      <td style={{ fontWeight: 700 }}>{feat}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>
                        {d.ks_statistic !== undefined ? Number(d.ks_statistic).toFixed(4) : '—'}
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>
                        {d.p_value !== undefined ? Number(d.p_value).toFixed(4) : '—'}
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                        {d.psi !== undefined ? Number(d.psi).toFixed(4) : '—'}
                      </td>
                      <td>
                        {isDrift ? (
                          <span className="badge badge-high" style={{ fontSize: 10 }}>
                            <AlertTriangle size={11} /> Drift Detected
                          </span>
                        ) : (
                          <span className="badge badge-low" style={{ fontSize: 10 }}>
                            <CheckCircle2 size={11} /> Distribution Stable
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
                    {loading ? 'Computing Kolmogorov-Smirnov statistics...' : 'Data drift report unavailable.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Telemetry Event Logs */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Clock size={18} style={{ color: 'var(--accent-mint)' }} />
              <span>Recent Inference Events Log</span>
            </h3>
            <p className="card-desc">Latest prediction requests recorded in privacy-preserving telemetry</p>
          </div>
        </div>

        {recentEvents.length > 0 ? (
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Model</th>
                  <th>Latency</th>
                  <th>Probability</th>
                  <th>Risk Tier</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {recentEvents.slice().reverse().map((ev, i) => (
                  <tr key={i}>
                    <td style={{ color: 'var(--text-secondary)' }}>
                      {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '—'}
                    </td>
                    <td style={{ fontWeight: 600 }}>{ev.model_name || 'Active Model'}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{ev.latency_ms} ms</td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)' }}>
                      {ev.probability >= 0 ? `${(ev.probability * 100).toFixed(1)}%` : 'Batch'}
                    </td>
                    <td>
                      <span
                        className={`badge ${
                          ev.risk_category === 'High Risk'
                            ? 'badge-high'
                            : ev.risk_category === 'Moderate Risk'
                            ? 'badge-moderate'
                            : 'badge-low'
                        }`}
                      >
                        {ev.risk_category}
                      </span>
                    </td>
                    <td>
                      <span className="badge badge-low">{ev.status || 'success'}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '30px 16px', color: 'var(--text-muted)', fontSize: 12 }}>
            No prediction events recorded in this session yet. Run an assessment to generate live telemetry.
          </div>
        )}
      </div>
    </div>
  );
}
