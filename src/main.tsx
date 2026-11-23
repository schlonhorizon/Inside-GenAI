import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

// Schriften (lokal über @fontsource, keine externen Requests)
import '@fontsource-variable/fraunces/full.css';
import '@fontsource-variable/fraunces/full-italic.css';
import '@fontsource/instrument-serif/400.css';
import '@fontsource/instrument-serif/400-italic.css';
import '@fontsource/young-serif/400.css';
import '@fontsource-variable/inter/index.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/400-italic.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import 'katex/dist/katex.min.css';

import './styles/tokens.css';
import './styles/global.css';
import './styles/styletile.css';

import { StyleTile } from './pages/StyleTile';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <StyleTile />
  </StrictMode>,
);
