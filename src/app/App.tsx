import { lazy, Suspense } from 'react';
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

// selten besuchte Seiten erst bei Bedarf laden
const DiagnosePage = lazy(() => import('../pages/DiagnosePage'));
const StyleguidePage = lazy(() => import('../pages/StyleguidePage'));

export function App() {
  const { reduced } = useApplySettings();
  const route = useRoute();
  useScrollOnRoute(route);

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
