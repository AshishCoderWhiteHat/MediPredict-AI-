import React from 'react';
import { Menu, Sun, Moon, ChevronRight, Activity, PlusCircle } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const PAGE_NAMES = {
  home: 'Home & Overview',
  dashboard: 'Analytics Dashboard',
  assessment: 'Diabetes Risk Assessment',
  results: 'Prediction Results & Biomarkers',
  reports: 'My Saved Reports',
  insights: 'Health Insights & Explainability',
  models: 'Model Lab & Benchmarks',
  monitoring: 'Model Monitoring & Telemetry',
};

export function Topbar({ currentPage, onOpenSidebar, onNavigate, apiOnline }) {
  const { theme, toggleTheme, isDark } = useTheme();

  return (
    <header className="app-topbar">
      <div className="topbar-left">
        <button
          className="mobile-nav-toggle"
          onClick={onOpenSidebar}
          aria-label="Open navigation menu"
        >
          <Menu size={20} />
        </button>
        <div className="breadcrumbs">
          <span>MediPredict AI</span>
          <ChevronRight size={14} />
          <span className="current">{PAGE_NAMES[currentPage] || currentPage}</span>
        </div>
      </div>

      <div className="topbar-right">
        {/* Working Theme Toggle */}
        <button
          className="theme-toggle-btn"
          onClick={toggleTheme}
          aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
          title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        >
          {isDark ? <Sun size={15} /> : <Moon size={15} />}
          <span>{isDark ? 'Light' : 'Dark'}</span>
        </button>

        {/* Quick Action */}
        {currentPage !== 'assessment' && (
          <button
            className="btn btn-primary btn-sm"
            onClick={() => onNavigate('assessment')}
            style={{ gap: 6 }}
          >
            <PlusCircle size={14} />
            <span>New Assessment</span>
          </button>
        )}
      </div>
    </header>
  );
}
