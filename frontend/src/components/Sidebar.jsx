import React from 'react';
import {
  LayoutDashboard,
  HeartPulse,
  FileText,
  Activity,
  Cpu,
  LineChart,
  Home,
  ShieldCheck,
  Stethoscope,
  X,
  ChevronRight,
  Database
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'home', label: 'Home', icon: Home, section: 'Overview' },
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, section: 'Overview' },
  { id: 'assessment', label: 'Risk Assessment', icon: HeartPulse, section: 'Clinical Inference', badge: 'Interactive' },
  { id: 'reports', label: 'My Reports', icon: FileText, section: 'Clinical Inference' },
  { id: 'insights', label: 'Health Insights', icon: Activity, section: 'Analytics & XAI' },
  { id: 'models', label: 'Model Lab', icon: Cpu, section: 'Machine Learning', badge: '5 Models' },
  { id: 'monitoring', label: 'Model Monitoring', icon: LineChart, section: 'Machine Learning' },
];

export function Sidebar({ currentPage, onNavigate, isOpen, onClose, apiOnline, activeModel, savedReportsCount = 0 }) {
  // Group items by section
  const sections = ['Overview', 'Clinical Inference', 'Analytics & XAI', 'Machine Learning'];

  return (
    <>
      {isOpen && (
        <div
          className="modal-overlay"
          style={{ zIndex: 35 }}
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside className={`app-sidebar ${isOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-brand">
          <div className="brand-icon-box">
            <Stethoscope size={22} />
          </div>
          <div className="brand-titles">
            <div className="brand-name">
              MediPredict <span>AI</span>
            </div>
            <div className="brand-badge">Healthcare Analytics</div>
          </div>
          {isOpen && (
            <button
              className="close-btn"
              style={{ marginLeft: 'auto' }}
              onClick={onClose}
              aria-label="Close navigation"
            >
              <X size={18} />
            </button>
          )}
        </div>

        <div className="sidebar-nav-container">
          {sections.map(section => {
            const items = NAV_ITEMS.filter(item => item.section === section);
            return (
              <div key={section} style={{ marginBottom: 10 }}>
                <div className="nav-section-title">{section}</div>
                {items.map(item => {
                  const Icon = item.icon;
                  const isActive = currentPage === item.id;
                  return (
                    <button
                      key={item.id}
                      className={`nav-item ${isActive ? 'active' : ''}`}
                      onClick={() => {
                        onNavigate(item.id);
                        if (isOpen) onClose();
                      }}
                    >
                      <Icon size={18} />
                      <span>{item.label}</span>
                      {item.id === 'reports' && savedReportsCount > 0 && (
                        <span className="nav-badge teal">{savedReportsCount}</span>
                      )}
                      {item.badge && item.id !== 'reports' && (
                        <span className="nav-badge">{item.badge}</span>
                      )}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>

        <div className="sidebar-footer">
          <div className="api-status-card">
            <div className={`status-dot ${apiOnline ? 'online' : 'offline'}`} />
            <div className="status-text">
              <span className="status-label">
                {apiOnline ? 'FastAPI Connected' : 'Service Offline'}
              </span>
              <span className="status-sub">
                {apiOnline ? `Model: ${activeModel || 'Loaded'}` : 'Run uvicorn api:app'}
              </span>
            </div>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              fontSize: 10,
              color: 'var(--text-muted)',
              lineHeight: 1.4,
              padding: '0 4px',
            }}
          >
            <ShieldCheck size={14} style={{ flexShrink: 0, color: 'var(--accent-teal)' }} />
            <span>Educational Research Prototype · Non-Diagnostic</span>
          </div>
        </div>
      </aside>
    </>
  );
}
