import { useState, type ReactNode } from 'react';
import { CHAPTERS, pad } from '../content/chapters';
import { useProgress } from '../state/progress';
import { useSettings, type MotionPref, type ThemePref } from '../state/settings';
import { Segmented, Toggle } from '../components/primitives';
import { href, type Route } from './router';

function ProgressMark({ slug }: { slug: string }) {
  const p = useProgress((s) => s.chapters[slug]);
  const state = p?.completed ? 'done' : p?.visited ? 'seen' : 'new';
  const label = state === 'done' ? 'abgeschlossen' : state === 'seen' ? 'gelesen' : 'noch nicht besucht';
  return <span className={`pmark pmark--${state}`} aria-label={label} title={label} />;
}

function Sidebar({ route, onNavigate }: { route: Route; onNavigate: () => void }) {
  const current = route.name === 'chapter' ? route.slug : undefined;
  return (
    <nav className="sidebar" aria-label="Kapitel">
      <a className={`sidebar__home ${route.name === 'home' ? 'is-current' : ''}`} href={href.home()} onClick={onNavigate}>
        Übersicht
      </a>
      <ol className="sidebar__list">
        {CHAPTERS.map((c) => (
          <li key={c.slug}>
            <a
              href={href.chapter(c.slug)}
              className={current === c.slug ? 'is-current' : ''}
              aria-current={current === c.slug ? 'page' : undefined}
              onClick={onNavigate}
            >
              <span className="mono sidebar__num">{pad(c.number)}</span>
              <span className="sidebar__title">{c.title}</span>
              <ProgressMark slug={c.slug} />
            </a>
          </li>
        ))}
      </ol>
      <div className="sidebar__extra">
        <a href={href.glossary()} onClick={onNavigate} className={route.name === 'glossary' ? 'is-current' : ''}>
          Glossar
        </a>
        <a href={href.credits()} onClick={onNavigate} className={route.name === 'credits' ? 'is-current' : ''}>
          Quellen &amp; Lizenzen
        </a>
      </div>
    </nav>
  );
}

function SettingsPanel() {
  const { theme, setTheme, math, setMath, motion, setMotion } = useSettings();
  return (
    <div className="settings">
      <Segmented<ThemePref>
        label="Modus"
        value={theme}
        options={[
          { value: 'system', label: 'Auto' },
          { value: 'light', label: 'Hell' },
          { value: 'dark', label: 'Dunkel' },
        ]}
        onChange={setTheme}
      />
      <Toggle label="Mathe-Ebene" checked={math} onChange={setMath} />
      <Segmented<MotionPref>
        label="Bewegung"
        value={motion}
        options={[
          { value: 'system', label: 'Auto' },
          { value: 'full', label: 'Voll' },
          { value: 'reduced', label: 'Reduziert' },
        ]}
        onChange={setMotion}
      />
    </div>
  );
}

export function Layout({ route, children }: { route: Route; children: ReactNode }) {
  const [navOpen, setNavOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  return (
    <div className={`shell ${navOpen ? 'nav-open' : ''}`}>
      <a className="skip-link" href="#main">
        Zum Inhalt springen
      </a>
      <header className="topbar">
        <button
          type="button"
          className="topbar__menu btn"
          aria-expanded={navOpen}
          aria-controls="sidebar"
          onClick={() => setNavOpen((o) => !o)}
        >
          Kapitel
        </button>
        <a className="topbar__brand" href={href.home()}>
          Inside GenAI
        </a>
        <div className="topbar__settings-inline">
          <SettingsPanel />
        </div>
        <button
          type="button"
          className="topbar__settings-btn btn"
          aria-expanded={settingsOpen}
          onClick={() => setSettingsOpen((o) => !o)}
        >
          Einstellungen
        </button>
      </header>
      {settingsOpen && (
        <div className="settings-drawer">
          <SettingsPanel />
        </div>
      )}
      <div className="shell__body">
        <div id="sidebar" className="shell__nav">
          <Sidebar route={route} onNavigate={() => setNavOpen(false)} />
        </div>
        <main id="main" className="shell__main" tabIndex={-1}>
          {children}
          <footer className="site-footer">
            <p>
              <strong>Privates Lernprojekt, nicht mit Anthropic verbunden.</strong>
            </p>
            <p className="muted small">
              Code unter MIT-Lizenz · <a href={href.credits()}>Quellen &amp; Lizenzen</a> ·{' '}
              <a href={href.diagnose()}>Diagnose</a> · <a href={href.styleguide()}>Styleguide</a>
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
}
