import { lazy, Suspense, useEffect } from 'react';
import { MotionConfig } from 'motion/react';
import { MDXProvider } from '@mdx-js/react';
import { useApplySettings } from '../state/settings';
import { useRoute, useScrollOnRoute } from './router';
import { Layout } from './Layout';
import { Home } from '../pages/Home';
import { ChapterPage } from '../pages/ChapterPage';
import { GlossaryPage } from '../pages/GlossaryPage';
import { CreditsPage } from '../pages/CreditsPage';
import { NotFound } from '../pages/NotFound';
import { mdxComponents } from '../components/learn/Learn';
import { CHAPTERS, loadChapter } from '../content/chapters';

// selten besuchte Seiten erst bei Bedarf laden
const DiagnosePage = lazy(() => import('../pages/DiagnosePage'));
const StyleguidePage = lazy(() => import('../pages/StyleguidePage'));
const PlaygroundPage = lazy(() => import('../pages/PlaygroundPage'));

/**
 * Nach dem ersten Laden im Leerlauf vorladen: fertige Kapitel, Playground, Tokenizer, Toy-Modell.
 * Damit funktioniert die Navigation zwischen den MVP-Routen ohne Netz, solange der Tab offen ist
 * (kein Service Worker – ein Neuladen offline funktioniert nicht).
 */
function usePrefetch() {
  useEffect(() => {
    const run = () => {
      for (const c of CHAPTERS) if (c.ready) loadChapter(c.slug)?.catch(() => {});
      import('../pages/PlaygroundPage').catch(() => {});
      // dynamisch, damit die Engine nicht im Haupt-Bundle landet
      import('../engine/lm').then((m) => m.loadToyModel('toy')).catch(() => {});
      import('../engine/tokenizer/load')
        .then((m) => Promise.all([m.loadTokenizer('gpt2'), m.loadTokenizer('smollm2')]))
        .catch(() => {});
    };
    const w = window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number };
    if (w.requestIdleCallback) w.requestIdleCallback(run, { timeout: 4000 });
    else setTimeout(run, 1500);
  }, []);
}

export function App() {
  const { reduced } = useApplySettings();
  const route = useRoute();
  useScrollOnRoute(route);
  usePrefetch();

  let page;
  switch (route.name) {
    case 'home':
      page = <Home />;
      break;
    case 'chapter':
      page = <ChapterPage slug={route.slug} />;
      break;
    case 'glossary':
      page = <GlossaryPage activeId={route.id} />;
      break;
    case 'credits':
      page = <CreditsPage />;
      break;
    case 'diagnose':
      page = <DiagnosePage />;
      break;
    case 'styleguide':
      page = <StyleguidePage />;
      break;
    case 'playground':
      page = <PlaygroundPage />;
      break;
    default:
      page = <NotFound path={route.path} />;
  }

  return (
    <MotionConfig reducedMotion={reduced ? 'always' : 'never'}>
      <MDXProvider components={mdxComponents}>
        <Layout route={route}>
          <Suspense fallback={<p className="meta loading">Lädt …</p>}>{page}</Suspense>
        </Layout>
      </MDXProvider>
    </MotionConfig>
  );
}
