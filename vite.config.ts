import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// BASE_PATH wird im Deploy-Workflow aus actions/configure-pages gesetzt
// (z. B. "/Inside-GenAI/"). Lokal läuft die Seite unter "/".
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    assetsInlineLimit: 0,
  },
});
