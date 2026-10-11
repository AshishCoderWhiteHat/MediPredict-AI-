import React from 'react';
import {
  HeartPulse,
  BrainCircuit,
  Activity,
  ShieldCheck,
  ArrowRight,
  CheckCircle2,
  Sparkles,
  Lock,
  Layers,
  ChevronRight,
  TrendingUp,
  FileCheck2,
  BarChart2
} from 'lucide-react';

export function HomePage({ onNavigate, championModel = 'Logistic Regression', testAuc = '81.3%' }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 48, paddingBottom: 24 }}>
      {/* Hero Section */}
      <section
        style={{
          background: 'linear-gradient(180deg, rgba(27, 41, 64, 0.4) 0%, rgba(15, 23, 42, 0) 100%)',
          borderRadius: 'var(--radius-xl)',
          border: '1px solid var(--border-subtle)',
          padding: '48px 36px',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: -100,
            right: -100,
            width: 380,
            height: 380,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(45, 212, 191, 0.12) 0%, rgba(0,0,0,0) 70%)',
            pointerEvents: 'none',
          }}
        />

        <div style={{ maxWidth: 740, position: 'relative', zIndex: 2 }}>
          <div className="page-eyebrow">
            <Sparkles size={14} />
            <span>AI-Driven Healthcare Intelligence</span>
          </div>
          <h1
            style={{
              fontSize: 'clamp(28px, 4vw, 42px)',
              fontWeight: 800,
              lineHeight: 1.15,
              letterSpacing: '-1px',
              color: 'var(--text-primary)',
              marginBottom: 16,
            }}
          >
            Understand Your <span style={{ color: 'var(--accent-teal)' }}>Diabetes Risk</span> with Transparent Machine Learning
          </h1>
          <p
            style={{
              fontSize: 16,
              lineHeight: 1.6,
              color: 'var(--text-secondary)',
              marginBottom: 28,
              maxWidth: 620,
            }}
          >
            MediPredict AI evaluates patient biomarkers using rigorous, calibrated classification models.
            Receive an explainable risk probability, comparative cohort deviations, and transparent feature attributions in seconds.
          </p>

          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center' }}>
            <button
              className="btn btn-primary btn-lg"
              onClick={() => onNavigate('assessment')}
              style={{ gap: 10 }}
            >
              <HeartPulse size={18} />
              <span>Start Risk Assessment</span>
              <ArrowRight size={16} />
            </button>
            <button
              className="btn btn-secondary btn-lg"
              onClick={() => onNavigate('dashboard')}
            >
              <span>View Analytics Dashboard</span>
            </button>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 24,
              marginTop: 32,
              paddingTop: 24,
              borderTop: '1px solid var(--border-muted)',
              flexWrap: 'wrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
              <CheckCircle2 size={16} style={{ color: 'var(--accent-teal)' }} />
              <span>5 Validated ML Algorithms</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
              <CheckCircle2 size={16} style={{ color: 'var(--accent-teal)' }} />
              <span>Instant Biomarker Discrepancy</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
              <CheckCircle2 size={16} style={{ color: 'var(--accent-teal)' }} />
              <span>Zero PHI Retention (Privacy-First)</span>
            </div>
          </div>
        </div>

        {/* Dashboard Preview Mockup Card */}
        <div
          style={{
            marginTop: 40,
            background: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-lg)',
            padding: 24,
            boxShadow: 'var(--shadow-lg)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: 18,
              borderBottom: '1px solid var(--border-muted)',
              paddingBottom: 12,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#EF4444' }} />
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#F59E0B' }} />
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#10B981' }} />
              <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8, fontFamily: 'var(--font-mono)' }}>
                dashboard.medipredict.ai/preview
              </span>
            </div>
            <span className="badge badge-low">
              <Sparkles size={11} /> Live Architecture Preview
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
            <div
              style={{
                background: 'var(--bg-panel)',
                borderRadius: 'var(--radius-md)',
                padding: 16,
                border: '1px solid var(--border-muted)',
              }}
            >
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Champion Model</div>
              <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--text-primary)' }}>{championModel}</div>
              <div style={{ fontSize: 11, color: 'var(--accent-teal)', marginTop: 4 }}>ROC-AUC: {testAuc}</div>
            </div>

            <div
              style={{
                background: 'var(--bg-panel)',
                borderRadius: 'var(--radius-md)',
                padding: 16,
                border: '1px solid var(--border-muted)',
              }}
            >
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Calibrated Risk Tiers</div>
              <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                <span className="badge badge-low" style={{ fontSize: 10 }}>&lt;30% Low</span>
                <span className="badge badge-moderate" style={{ fontSize: 10 }}>30-65% Mod</span>
                <span className="badge badge-high" style={{ fontSize: 10 }}>&gt;65% High</span>
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
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Biomarkers Tracked</div>
              <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--accent-mint)' }}>8 Clinical Metrics</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>Glucose, BMI, Insulin, DPF +4</div>
            </div>

            <div
              style={{
                background: 'var(--bg-panel)',
                borderRadius: 'var(--radius-md)',
                padding: 16,
                border: '1px solid var(--border-muted)',
              }}
            >
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Explainability Tech</div>
              <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--accent-blue)' }}>Cohort Deviations</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>Population Median Reference</div>
            </div>
          </div>
        </div>
      </section>

      {/* Core Platform Capabilities */}
      <section>
        <div style={{ textAlign: 'center', maxWidth: 600, margin: '0 auto 36px' }}>
          <div className="page-eyebrow" style={{ justifyContent: 'center' }}>
            <Layers size={14} /> Core Capabilities
          </div>
          <h2 style={{ fontSize: 26, fontWeight: 800, color: 'var(--text-primary)', marginBottom: 10 }}>
            Engineered for Precision & Medical Explainability
          </h2>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
            MediPredict AI is not a black-box predictor. Every score is linked to actionable clinical parameters and reference comparisons.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 20 }}>
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-teal-subtle)',
                color: 'var(--accent-teal)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <HeartPulse size={22} />
            </div>
            <h3 style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-primary)' }}>
              1. Calibrated Risk Prediction
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              Receive an estimated probability of diabetes risk derived from trained statistical and gradient boosting ensembles. Results are segmented into clear clinical tiers with visual indicator bars.
            </p>
            <button
              className="btn btn-outline btn-sm"
              style={{ alignSelf: 'flex-start', marginTop: 'auto' }}
              onClick={() => onNavigate('assessment')}
            >
              Try Risk Assessment <ChevronRight size={14} />
            </button>
          </div>

          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-blue-subtle)',
                color: 'var(--accent-blue)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <BrainCircuit size={22} />
            </div>
            <h3 style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-primary)' }}>
              2. Explainable AI & Biomarkers
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              Understand why the model predicted a particular tier. Discover which specific biomarkers (such as impaired fasting glucose or high BMI) elevated the patient's individual risk above cohort norms.
            </p>
            <button
              className="btn btn-outline btn-sm"
              style={{ alignSelf: 'flex-start', marginTop: 'auto' }}
              onClick={() => onNavigate('insights')}
            >
              Explore Insights <ChevronRight size={14} />
            </button>
          </div>

          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-purple-subtle)',
                color: 'var(--accent-purple)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <BarChart2 size={22} />
            </div>
            <h3 style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-primary)' }}>
              3. Model Lab & Benchmarking
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              Inspect and compare all 5 supervised models—Logistic Regression, Random Forest, Gradient Boosting, SVM, and XGBoost. Evaluate held-out ROC curves, confusion matrices, and Brier scores.
            </p>
            <button
              className="btn btn-outline btn-sm"
              style={{ alignSelf: 'flex-start', marginTop: 'auto' }}
              onClick={() => onNavigate('models')}
            >
              Open Model Lab <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section
        style={{
          background: 'var(--bg-panel)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-xl)',
          padding: '36px 30px',
        }}
      >
        <div style={{ textAlign: 'center', maxWidth: 600, margin: '0 auto 32px' }}>
          <div className="page-eyebrow" style={{ justifyContent: 'center' }}>
            <TrendingUp size={14} /> Streamlined Clinical Workflow
          </div>
          <h2 style={{ fontSize: 24, fontWeight: 800, color: 'var(--text-primary)' }}>
            How MediPredict AI Operates
          </h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 20 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-teal)',
                color: '#0B1120',
                fontWeight: 800,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              01
            </div>
            <h4 style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
              Enter Clinical Features
            </h4>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              Provide standard clinical indicators across glycemia, body mass, blood pressure, and demographic history. Validated ranges ensure reliable inputs.
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-teal)',
                color: '#0B1120',
                fontWeight: 800,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              02
            </div>
            <h4 style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
              FastAPI Inference Engine
            </h4>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              The payload is serialized to the backend pipeline. Scalers, median imputers, and your chosen classification algorithm compute probability in under 50ms.
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-teal)',
                color: '#0B1120',
                fontWeight: 800,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              03
            </div>
            <h4 style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
              Transparent Report & Export
            </h4>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              View the final risk tier, compare your values to population medians, save the assessment locally, and print an educational summary report.
            </p>
          </div>
        </div>
      </section>

      {/* Privacy and Educational Disclaimer Section */}
      <section
        style={{
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg)',
          padding: '24px 28px',
          background: 'var(--bg-card)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: 16,
        }}
      >
        <Lock size={28} style={{ color: 'var(--accent-teal)', flexShrink: 0, marginTop: 2 }} />
        <div>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
            Privacy Architecture & Academic Medical Disclaimer
          </h4>
          <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 8 }}>
            <strong>Privacy Protection:</strong> MediPredict AI operates with a zero-retention policy for personally identifiable health information. Any saved assessments in "My Reports" reside solely within your local browser storage and can be exported or purged at any time.
          </p>
          <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <strong>Clinical Disclaimer:</strong> This web application is an educational and research prototype. Model outputs are statistical estimates based on benchmark datasets and do not constitute a formal diagnosis, treatment plan, or medical advice. Always consult a qualified medical physician for diagnostic evaluations.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingTop: 20,
          borderTop: '1px solid var(--border-muted)',
          fontSize: 12,
          color: 'var(--text-muted)',
          flexWrap: 'wrap',
          gap: 12,
        }}
      >
        <div>
          © MediPredict AI · Machine Learning for Healthcare Analytics
        </div>
        <div style={{ display: 'flex', gap: 16 }}>
          <span>React 18 + Vite</span>
          <span>·</span>
          <span>Python FastAPI</span>
          <span>·</span>
          <span>Scikit-Learn & XGBoost</span>
        </div>
      </footer>
    </div>
  );
}
