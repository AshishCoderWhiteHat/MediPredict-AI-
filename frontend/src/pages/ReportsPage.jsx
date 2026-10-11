import React, { useState } from 'react';
import {
  FileText,
  Download,
  Trash2,
  Printer,
  Eye,
  Calendar,
  AlertTriangle,
  CheckCircle2,
  ShieldCheck,
  HeartPulse,
  X,
  FileSpreadsheet,
  FileCode,
  Sparkles
} from 'lucide-react';

export function ReportsPage({
  savedReports = [],
  onDeleteReport,
  onClearReports,
  onNavigate,
  onViewReportDetails,
}) {
  const [selectedReport, setSelectedReport] = useState(null);

  const handleExportJSON = () => {
    if (!savedReports.length) return;
    const blob = new Blob([JSON.stringify(savedReports, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medipredict_assessments_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportCSV = () => {
    if (!savedReports.length) return;
    const headers = ['ID', 'Date', 'Model', 'RiskCategory', 'Probability', 'Glucose', 'BMI', 'BloodPressure', 'Insulin', 'Age'];
    const rows = savedReports.map(r => {
      const p = r.patientData || {};
      return [
        r.id,
        new Date(r.date).toLocaleDateString(),
        `"${r.model_name || 'Champion'}"`,
        `"${r.risk_category}"`,
        (r.probability * 100).toFixed(1),
        p.Glucose ?? '',
        p.BMI ?? '',
        p.BloodPressure ?? '',
        p.Insulin ?? '',
        p.Age ?? '',
      ].join(',');
    });

    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medipredict_assessments_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-eyebrow">
            <FileText size={14} /> Local Assessment History
          </div>
          <h1 className="page-title">My Saved Clinical Reports</h1>
          <p className="page-subtitle">
            Review, print, and export your historical diabetes risk assessments stored privately in browser storage.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }} className="no-print">
          {savedReports.length > 0 && (
            <>
              <button className="btn btn-secondary btn-sm" onClick={handleExportCSV}>
                <FileSpreadsheet size={15} />
                <span>Export CSV</span>
              </button>
              <button className="btn btn-secondary btn-sm" onClick={handleExportJSON}>
                <FileCode size={15} />
                <span>Export JSON</span>
              </button>
              <button className="btn btn-danger btn-sm" onClick={onClearReports}>
                <Trash2 size={15} />
                <span>Clear All</span>
              </button>
            </>
          )}
          <button className="btn btn-primary btn-sm" onClick={() => onNavigate('assessment')}>
            <HeartPulse size={15} />
            <span>New Assessment</span>
          </button>
        </div>
      </div>

      {/* Privacy Notice Banner */}
      <div className="clinical-disclaimer-banner">
        <ShieldCheck size={18} />
        <div className="disclaimer-text">
          <strong>Privacy Architecture:</strong> All assessments are persisted strictly in your local browser storage (localStorage). No patient identifiers, blood glucose values, or clinical measurements are uploaded to external databases.
        </div>
      </div>

      {/* Reports Table or Empty State */}
      {savedReports.length > 0 ? (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div className="table-wrapper" style={{ border: 'none' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date & Time</th>
                  <th>Risk Tier</th>
                  <th>Probability</th>
                  <th>Model Used</th>
                  <th>Key Parameters</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {savedReports.map(report => {
                  const p = report.patientData || {};
                  return (
                    <tr key={report.id}>
                      <td style={{ color: 'var(--text-secondary)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <Calendar size={13} style={{ color: 'var(--accent-teal)' }} />
                          <span>{new Date(report.date).toLocaleString()}</span>
                        </div>
                      </td>
                      <td>
                        <span
                          className={`badge ${
                            report.risk_category === 'High Risk'
                              ? 'badge-high'
                              : report.risk_category === 'Moderate Risk'
                              ? 'badge-moderate'
                              : 'badge-low'
                          }`}
                        >
                          {report.risk_category}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                        {(report.probability * 100).toFixed(1)}%
                      </td>
                      <td style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                        {report.model_name || 'Champion Model'}
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                        Glucose: {p.Glucose ?? '—'} · BMI: {p.BMI ?? '—'} · Age: {p.Age ?? '—'}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => setSelectedReport(report)}
                            title="Inspect Assessment"
                          >
                            <Eye size={13} />
                            <span>Details</span>
                          </button>
                          <button
                            className="btn btn-danger btn-sm"
                            onClick={() => onDeleteReport(report.id)}
                            title="Delete this record"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div
          className="card-elevated"
          style={{
            textAlign: 'center',
            padding: '60px 20px',
            maxWidth: 550,
            margin: '20px auto',
          }}
        >
          <FileText size={48} style={{ color: 'var(--text-muted)', marginBottom: 16 }} />
          <h3 style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-primary)', marginBottom: 6 }}>
            No Saved Assessments Found
          </h3>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 20, lineHeight: 1.6 }}>
            When you complete an assessment, select "Save to My Reports" on the results screen to record it locally for future comparison.
          </p>
          <button className="btn btn-primary" onClick={() => onNavigate('assessment')}>
            <HeartPulse size={16} />
            <span>Start Risk Assessment</span>
          </button>
        </div>
      )}

      {/* Assessment Details Modal */}
      {selectedReport && (
        <div className="modal-overlay" onClick={() => setSelectedReport(null)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3 className="modal-title">Assessment Report Details</h3>
                <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Record ID: {selectedReport.id} · Generated: {new Date(selectedReport.date).toLocaleString()}
                </p>
              </div>
              <button className="close-btn" onClick={() => setSelectedReport(null)}>
                <X size={18} />
              </button>
            </div>

            <div
              style={{
                background: 'var(--bg-card)',
                borderRadius: 'var(--radius-md)',
                padding: 16,
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div>
                <span
                  className={`badge ${
                    selectedReport.risk_category === 'High Risk'
                      ? 'badge-high'
                      : selectedReport.risk_category === 'Moderate Risk'
                      ? 'badge-moderate'
                      : 'badge-low'
                  }`}
                  style={{ marginBottom: 6 }}
                >
                  {selectedReport.risk_category}
                </span>
                <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-primary)' }}>
                  Probability: {(selectedReport.probability * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Model</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent-teal)' }}>
                  {selectedReport.model_name || 'Champion Model'}
                </div>
              </div>
            </div>

            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
              Submitted Biomarkers
            </div>
            <div
              style={{
                background: 'var(--bg-card)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
                overflow: 'hidden',
              }}
            >
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(selectedReport.patientData || {}).map(([key, val]) => (
                    <tr key={key}>
                      <td style={{ fontWeight: 600 }}>{key}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700 }}>
                        {val}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {selectedReport.explanation?.top_elevated_risk_factors?.length > 0 && (
              <div
                style={{
                  background: 'rgba(239, 68, 68, 0.06)',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  borderRadius: 'var(--radius-md)',
                  padding: 12,
                  fontSize: 12,
                }}
              >
                <div style={{ fontWeight: 700, color: 'var(--status-danger)', marginBottom: 4 }}>
                  Elevated Biomarkers Identified:
                </div>
                <ul style={{ paddingLeft: 18, color: 'var(--text-secondary)' }}>
                  {selectedReport.explanation.top_elevated_risk_factors.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => {
                  window.print();
                }}
              >
                <Printer size={14} />
                <span>Print This Report</span>
              </button>
              <button className="btn btn-primary btn-sm" onClick={() => setSelectedReport(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
