import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

// Schriften (lokal über @fontsource, keine externen Requests) – Entscheidung: docs/design-decisions.md
import '@fontsource-variable/fraunces/full.css';
import '@fontsource-variable/fraunces/full-italic.css';
import '@fontsource-variable/inter/index.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import 'katex/dist/katex.min.css';

import './styles/tokens.css';
import './styles/global.css';
import './styles/components.css';
import './styles/app.css';

import { App } from './app/App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
