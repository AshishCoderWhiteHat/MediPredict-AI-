import React, { useState } from 'react';
import {
  HeartPulse,
  BrainCircuit,
  ClipboardCheck,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Settings2,
  Sliders,
  X
} from 'lucide-react';

const FEATURE_CONFIGS = [
  // Section 1: Glycemic & Metabolic
  {
    key: 'Glucose',
    label: 'Fasting Plasma Glucose',
    unit: 'mg/dL',
    section: 'Glycemic & Metabolic',
    min: 30,
    max: 400,
    step: 1,
    default: 115,
    description: '2-hour plasma glucose concentration from oral glucose tolerance test. Normal fasting is < 100 mg/dL.',
  },
  {
    key: 'Insulin',
    label: '2-Hour Serum Insulin',
    unit: 'µU/mL',
    section: 'Glycemic & Metabolic',
    min: 0,
    max: 1000,
    step: 1,
    default: 95,
    description: 'Serum insulin concentration. Enter 0 if test was not performed (the ML pipeline handles zero-imputation).',
  },

  // Section 2: Cardiovascular & Body Composition
  {
    key: 'BloodPressure',
    label: 'Diastolic Blood Pressure',
    unit: 'mmHg',
    section: 'Cardiovascular & Body',
    min: 30,
    max: 200,
    step: 1,
    default: 72,
    description: 'Resting diastolic pressure in mm Hg. Normal resting pressure is generally below 80 mm Hg.',
  },
  {
    key: 'BMI',
    label: 'Body Mass Index (BMI)',
    unit: 'kg/m²',
    section: 'Cardiovascular & Body',
    min: 10.0,
    max: 80.0,
    step: 0.1,
    default: 28.5,
    description: 'Weight (kg) divided by height squared (m²). Overweight range starts at 25.0; clinical obesity at ≥ 30.0.',
  },
  {
    key: 'SkinThickness',
    label: 'Triceps Skinfold Thickness',
    unit: 'mm',
    section: 'Cardiovascular & Body',
    min: 0,
    max: 120,
    step: 1,
    default: 25,
    description: 'Measurement estimating body fat reserve. Enter 0 if not measured (handled by pipeline median imputer).',
  },

  // Section 3: Demographics & Genetics
  {
    key: 'Age',
    label: 'Patient Age',
    unit: 'years',
    section: 'Demographics & Genetics',
    min: 18,
    max: 120,
    step: 1,
    default: 32,
    description: 'Patient age in years (cohort dataset evaluated adult patients aged 18 to 120).',
  },
  {
    key: 'Pregnancies',
    label: 'Number of Pregnancies',
    unit: 'count',
    section: 'Demographics & Genetics',
    min: 0,
    max: 25,
    step: 1,
    default: 1,
    description: 'Total number of gestational pregnancies. Gestational history is a recognized clinical risk factor.',
  },
  {
    key: 'DiabetesPedigreeFunction',
    label: 'Diabetes Pedigree Function (DPF)',
    unit: 'genetic score',
    section: 'Demographics & Genetics',
    min: 0.01,
    max: 3.50,
    step: 0.01,
    default: 0.45,
    description: 'Synthesis of genetic pedigree and family history of diabetes mellitus. Cohort median is ~0.38.',
  },
];

const PRESETS = [
  {
    name: 'Typical Baseline',
    desc: 'Average reference patient',
    values: {
      Pregnancies: 1,
      Glucose: 115,
      BloodPressure: 72,
      SkinThickness: 25,
      Insulin: 95,
      BMI: 28.5,
      DiabetesPedigreeFunction: 0.45,
      Age: 32,
    },
  },
  {
    name: 'Metabolic Syndrome Risk',
    desc: 'Elevated glucose, BMI, and family history',
    values: {
      Pregnancies: 3,
      Glucose: 158,
      BloodPressure: 84,
      SkinThickness: 33,
      Insulin: 180,
      BMI: 35.8,
      DiabetesPedigreeFunction: 0.78,
      Age: 51,
    },
  },
  {
    name: 'Low Risk Profile',
    desc: 'Normal glucose, optimal BMI, younger age',
    values: {
      Pregnancies: 0,
      Glucose: 88,
      BloodPressure: 68,
      SkinThickness: 20,
      Insulin: 70,
      BMI: 22.4,
      DiabetesPedigreeFunction: 0.25,
      Age: 26,
    },
  },
];

export function AssessmentPage({
  availableModels = [],
  selectedModel,
  onSelectModel,
  onSubmitPrediction,
  loading,
  apiOnline,
  errorMessage,
  onClearError,
}) {
  const [formData, setFormData] = useState(() => {
    const init = {};
    FEATURE_CONFIGS.forEach(f => {
      init[f.key] = f.default;
    });
    return init;
  });

  const [fieldErrors, setFieldErrors] = useState({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showReviewModal, setShowReviewModal] = useState(false);

  const sections = ['Glycemic & Metabolic', 'Cardiovascular & Body', 'Demographics & Genetics'];

  const handleInputChange = (key, value, config) => {
    const numVal = value === '' ? '' : Number(value);
    setFormData(prev => ({ ...prev, [key]: numVal }));

    // Instant validation
    if (value === '') {
      setFieldErrors(prev => ({ ...prev, [key]: 'Value is required' }));
    } else if (Number(value) < config.min || Number(value) > config.max) {
      setFieldErrors(prev => ({
        ...prev,
        [key]: `Must be between ${config.min} and ${config.max}`,
      }));
    } else {
      setFieldErrors(prev => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
    if (onClearError) onClearError();
  };

  const applyPreset = preset => {
    setFormData({ ...preset.values });
    setFieldErrors({});
    if (onClearError) onClearError();
  };

  const handleReset = () => {
    const reset = {};
    FEATURE_CONFIGS.forEach(f => {
      reset[f.key] = f.default;
    });
    setFormData(reset);
    setFieldErrors({});
    if (onClearError) onClearError();
  };

  const validateAll = () => {
    const errors = {};
    FEATURE_CONFIGS.forEach(f => {
      const val = formData[f.key];
      if (val === '' || val === null || val === undefined) {
        errors[f.key] = 'Required';
      } else if (Number(val) < f.min || Number(val) > f.max) {
        errors[f.key] = `Range: ${f.min} - ${f.max}`;
      }
    });
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleOpenReview = e => {
    e.preventDefault();
    if (validateAll()) {
      setShowReviewModal(true);
    }
  };

  const handleConfirmSubmit = () => {
    setShowReviewModal(false);
    // Ensure exact types expected by FastAPI schema
    const payload = {
      Pregnancies: parseInt(formData.Pregnancies, 10),
      Glucose: parseFloat(formData.Glucose),
      BloodPressure: parseFloat(formData.BloodPressure),
      SkinThickness: parseFloat(formData.SkinThickness),
      Insulin: parseFloat(formData.Insulin),
      BMI: parseFloat(formData.BMI),
      DiabetesPedigreeFunction: parseFloat(formData.DiabetesPedigreeFunction),
      Age: parseInt(formData.Age, 10),
    };
    onSubmitPrediction(payload);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24, position: 'relative' }}>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-eyebrow">
            <HeartPulse size={14} /> Clinical Risk Inference
          </div>
          <h1 className="page-title">Diabetes Risk Assessment Form</h1>
          <p className="page-subtitle">
            Enter the patient's 8 validated physiological measurements to generate an explainable machine learning risk estimate.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={handleReset}
            disabled={loading}
          >
            <RotateCcw size={14} />
            <span>Reset Form</span>
          </button>
        </div>
      </div>

      {/* Clinical Disclaimer */}
      <div className="clinical-disclaimer-banner">
        <ShieldCheck size={18} />
        <div className="disclaimer-text">
          <strong>Validation Rules:</strong> All eight features are required by the ML model pipeline. Measurements must represent standard clinical units. This prototype provides an educational estimate, not a diagnosis.
        </div>
      </div>

      {/* Error alert */}
      {errorMessage && (
        <div
          style={{
            background: 'var(--status-danger-bg)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 'var(--radius-md)',
            padding: '12px 16px',
            color: 'var(--status-danger)',
            fontSize: 13,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertCircle size={16} />
            <span>{errorMessage}</span>
          </div>
          <button onClick={onClearError} style={{ color: 'inherit' }}>
            <X size={16} />
          </button>
        </div>
      )}

      {/* Quick Preset Selector */}
      <div className="card" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
            <Sparkles size={16} style={{ color: 'var(--accent-teal)' }} />
            <span>Load Reference Clinical Presets:</span>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {PRESETS.map(preset => (
              <button
                key={preset.name}
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => applyPreset(preset)}
                title={preset.desc}
              >
                <span>{preset.name}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Assessment Form */}
      <form onSubmit={handleOpenReview}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {sections.map(section => {
            const sectionFields = FEATURE_CONFIGS.filter(f => f.section === section);
            return (
              <div key={section} className="card">
                <div className="form-section-title">
                  <CheckCircle2 size={16} />
                  <span>{section}</span>
                </div>

                <div className="form-grid">
                  {sectionFields.map(field => {
                    const error = fieldErrors[field.key];
                    return (
                      <div key={field.key} className="form-group">
                        <label className="form-label" htmlFor={`input-${field.key}`}>
                          <span>{field.label}</span>
                          <span className="form-label-unit">{field.unit}</span>
                        </label>
                        <div className="input-container">
                          <input
                            id={`input-${field.key}`}
                            type="number"
                            className={`form-input ${error ? 'has-error' : ''}`}
                            value={formData[field.key]}
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            required
                            disabled={loading}
                            onChange={e => handleInputChange(field.key, e.target.value, field)}
                          />
                        </div>
                        {error ? (
                          <span className="form-error-msg">{error}</span>
                        ) : (
                          <span className="form-helper">{field.description}</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}

          {/* Advanced Model Selection Settings */}
          <div className="card" style={{ padding: '16px 20px' }}>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                color: 'var(--text-primary)',
                fontWeight: 700,
                fontSize: 13,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Settings2 size={16} style={{ color: 'var(--accent-teal)' }} />
                <span>Advanced Inference Settings (Model Selection)</span>
                <span className="badge badge-neutral" style={{ fontSize: 10 }}>
                  Active: {selectedModel}
                </span>
              </div>
              {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {showAdvanced && (
              <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border-muted)' }}>
                <div className="form-group" style={{ maxWidth: 420 }}>
                  <label className="form-label" htmlFor="model-select">
                    <span>Select Classification Algorithm</span>
                    <span className="form-label-unit">5 trained pipelines available</span>
                  </label>
                  <select
                    id="model-select"
                    className="form-input select-input"
                    value={selectedModel}
                    onChange={e => onSelectModel(e.target.value)}
                    disabled={loading || !availableModels.length}
                  >
                    {availableModels.map(name => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                  <span className="form-helper">
                    Override the default champion model to evaluate inference across alternative architectures (e.g., XGBoost, Random Forest, SVM).
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Form Actions */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '16px 0',
              flexWrap: 'wrap',
              gap: 12,
            }}
          >
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              All 8 features validated. Click Review to inspect parameters before submission.
            </div>

            <div style={{ display: 'flex', gap: 12 }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Reset Values
              </button>
              <button
                type="submit"
                className="btn btn-primary btn-lg"
                disabled={loading || !apiOnline}
              >
                <ClipboardCheck size={18} />
                <span>Review & Predict Risk</span>
              </button>
            </div>
          </div>
        </div>
      </form>

      {/* Review Modal Step */}
      {showReviewModal && (
        <div className="modal-overlay" onClick={() => setShowReviewModal(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--accent-teal-subtle)',
                    color: 'var(--accent-teal)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <ClipboardCheck size={20} />
                </div>
                <div>
                  <h3 className="modal-title">Review Clinical Parameters</h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    Confirm measurements before sending payload to FastAPI inference service.
                  </p>
                </div>
              </div>
              <button className="close-btn" onClick={() => setShowReviewModal(false)}>
                <X size={18} />
              </button>
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
                    <th>Feature Measurement</th>
                    <th>Entered Value</th>
                    <th>Standard Unit</th>
                  </tr>
                </thead>
                <tbody>
                  {FEATURE_CONFIGS.map(f => (
                    <tr key={f.key}>
                      <td style={{ fontWeight: 600 }}>{f.label}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700 }}>
                        {formData[f.key]}
                      </td>
                      <td style={{ color: 'var(--text-muted)' }}>{f.unit}</td>
                    </tr>
                  ))}
                  <tr>
                    <td style={{ fontWeight: 600 }}>Selected Model</td>
                    <td colSpan={2} style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
                      {selectedModel}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end', marginTop: 10 }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowReviewModal(false)}
              >
                Back to Edit
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleConfirmSubmit}
                disabled={loading}
              >
                <HeartPulse size={16} />
                <span>Confirm & Generate Estimate</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
