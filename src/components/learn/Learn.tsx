import { useId, type ReactNode } from 'react';
import { getTerm } from '../../content/glossary';
import { formatFact, getFact, MODEL_FACTS, type FactFormat } from '../../content/facts';
import { href } from '../../app/router';
import { Stamp, type StampKind } from '../primitives';

/* ── Glossar-Begriff mit Hover-Karte ──────────────────────────────── */

export function Term({ id, children }: { id: string; children?: ReactNode }) {
  const entry = getTerm(id);
  const tipId = useId();
  if (!entry) {
    // fehlende Einträge fallen im Test auf (test/content.test.ts); hier nicht crashen
    return <span className="term term--missing">{children ?? id}</span>;
  }
  return (
    <span className="term-wrap">
      <a className="term" href={href.glossary(entry.id)} aria-describedby={tipId}>
        {children ?? entry.term}
      </a>
      <span role="tooltip" id={tipId} className="term-card">
        <span className="term-card__head">
          <strong>{entry.term}</strong>
          {entry.alt && <span className="mono term-card__alt">{entry.alt}</span>}
        </span>
        <span className="term-card__body">{entry.short}</span>
      </span>
    </span>
  );
}

/* ── „Unter der Haube“: gehört zur Mathe-Ebene ───────────────────── */

export function DeepDive({ title = 'Unter der Haube', children }: { title?: string; children: ReactNode }) {
  return (
    <aside className="deepdive math-layer">
      <span className="meta">{title}</span>
      <div className="deepdive__body">{children}</div>
    </aside>
  );
}

/* ── Modellzahl aus der Config ───────────────────────────────────── */

export function Fact({ m, k, format = 'de' }: { m: string; k: string; format?: FactFormat }) {
  const v = getFact(m, k);
  if (v === undefined) {
    const label = MODEL_FACTS[m]?.label ?? m;
    return (
      <span className="fact fact--pending mono" title={`${label}: „${k}“ wird beim Build aus der Modell-Config gelesen (verfügbar ab P2).`}>
        n. v.
      </span>
    );
  }
  return <span className="fact mono">{formatFact(v, format)}</span>;
}

/* ── Kapitel-Bausteine ───────────────────────────────────────────── */

export function Takeaways({ children }: { children: ReactNode }) {
  return (
    <aside className="takeaways">
      <span className="meta">Was du jetzt weißt</span>
      {children}
    </aside>
  );
}

export function Placeholder({ phase, children }: { phase: string; children?: ReactNode }) {
  return (
    <div className="placeholder">
      <span className="meta">In Arbeit · folgt in Phase {phase}</span>
      {children && <div>{children}</div>}
    </div>
  );
}

export function StampNote({ kind, children }: { kind: StampKind; children: ReactNode }) {
  return (
    <p className="stamp-note">
      <Stamp kind={kind} /> <span className="muted">{children}</span>
    </p>
  );
}

/** Komponenten, die in jeder MDX-Datei ohne Import verfügbar sind. */
export const mdxComponents = { Term, DeepDive, Fact, Takeaways, Placeholder, StampNote, Stamp };
