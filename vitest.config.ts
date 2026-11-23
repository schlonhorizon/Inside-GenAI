import { defineConfig } from 'vitest/config';

// Tests prüfen Logik und Inhalte (Node-Umgebung, ohne die Vite-Plugins der App).
export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
  },
});
