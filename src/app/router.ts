import { useEffect, useSyncExternalStore } from 'react';

/**
 * Minimaler Hash-Router. GitHub Pages hat kein SPA-Fallback, darum liegen alle
 * Routen hinter "#/…" – Reload und Deep-Links funktionieren ohne Server-Konfiguration.
 *
 *   #/                      Start
 *   #/kapitel/<slug>        Kapitel
 *   #/kapitel/<slug>/<anker> Kapitel, zu Abschnitt scrollen
 *   #/glossar[/<id>]        Glossar
 *   #/credits · #/diagnose · #/styleguide
 */

export type Route =
  | { name: 'home' }
  | { name: 'chapter'; slug: string; anchor?: string }
  | { name: 'glossary'; id?: string }
  | { name: 'credits' }
  | { name: 'diagnose' }
  | { name: 'styleguide' }
  | { name: 'notfound'; path: string };

export function parseHash(hash: string): Route {
  const path = decodeURIComponent(hash.replace(/^#/, '')).replace(/\/+$/, '') || '/';
  const parts = path.split('/').filter(Boolean);
  if (parts.length === 0) return { name: 'home' };
  const [head, a, b] = parts;
  switch (head) {
    case 'kapitel':
      return a ? { name: 'chapter', slug: a, anchor: b } : { name: 'home' };
    case 'glossar':
      return { name: 'glossary', id: a };
    case 'credits':
      return { name: 'credits' };
    case 'diagnose':
      return { name: 'diagnose' };
    case 'styleguide':
      return { name: 'styleguide' };
    default:
      return { name: 'notfound', path };
  }
}

export const href = {
  home: () => '#/',
  chapter: (slug: string, anchor?: string) => `#/kapitel/${slug}${anchor ? `/${anchor}` : ''}`,
  glossary: (id?: string) => `#/glossar${id ? `/${id}` : ''}`,
  credits: () => '#/credits',
  diagnose: () => '#/diagnose',
  styleguide: () => '#/styleguide',
};

function subscribe(cb: () => void) {
  window.addEventListener('hashchange', cb);
  return () => window.removeEventListener('hashchange', cb);
}

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash);
  return parseHash(hash);
}

/** Nach Routenwechsel nach oben bzw. zum Anker scrollen. */
export function useScrollOnRoute(route: Route) {
  const key = JSON.stringify(route);
  useEffect(() => {
    const anchor = route.name === 'chapter' ? route.anchor : route.name === 'glossary' ? route.id : undefined;
    if (anchor) {
      // nach dem Rendern des (evtl. nachgeladenen) Inhalts
      const t = setTimeout(() => document.getElementById(anchor)?.scrollIntoView({ block: 'start' }), 60);
      return () => clearTimeout(t);
    }
    window.scrollTo({ top: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
}
