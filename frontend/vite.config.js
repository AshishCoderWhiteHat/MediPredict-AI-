import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
      '/model-info': 'http://127.0.0.1:8000',
      '/model-comparison': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/predict-batch': 'http://127.0.0.1:8000',
      '/drift-report': 'http://127.0.0.1:8000',
      '/telemetry': 'http://127.0.0.1:8000',
      '/feature-importance': 'http://127.0.0.1:8000',
      '/evaluation-details': 'http://127.0.0.1:8000',
      '/dataset-insights': 'http://127.0.0.1:8000',
      '/clustering': 'http://127.0.0.1:8000',
    },
  },
});
