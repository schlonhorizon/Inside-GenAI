import { CHAPTERS, pad } from '../content/chapters';
import { useProgress } from '../state/progress';
import { href } from '../app/router';

const PARTS: { title: string; question: string; from: number; to: number }[] = [
  { title: 'Fundament', question: 'Womit wird gerechnet?', from: 0, to: 1 },
  { title: 'Inferenz', question: 'Was passiert bei einer Anfrage?', from: 2, to: 8 },
  { title: 'Herkunft', question: 'Woher kommen die Gewichte?', from: 9, to: 10 },
  { title: 'Praxis', question: 'Was heißt das für mich?', from: 11, to: 14 },
  { title: 'Synthese', question: 'Alles zusammen', from: 15, to: 15 },
];

export function Home() {
  const progress = useProgress((s) => s.chapters);
  const lastSlug = useProgress((s) => s.lastSlug);
  const done = CHAPTERS.filter((c) => progress[c.slug]?.completed).length;
  const last = CHAPTERS.find((c) => c.slug === lastSlug);

  return (
    <article className="page home">
      <header className="home__hero">
        <span className="meta">Interaktives Lernkompendium</span>
        <h1>Wie aus einem Prompt eine Antwort wird</h1>
        <p className="lead">
          Schritt für Schritt durch das Innere eines Sprachmodells: vom Zerlegen des Textes in Tokens über Attention und
          Wahrscheinlichkeiten bis zum fertigen Satz. Mit Reglern, die du selbst verstellst, und einem echten kleinen
          Modell, das live im Browser rechnet.
        </p>
        <div className="home__actions">
          <a className="btn btn--primary" href={href.chapter(last?.slug ?? CHAPTERS[0].slug)}>
            {last ? `Weiter: ${last.title}` : 'Loslegen'}
          </a>
          <span className="meta">
            {done} / {CHAPTERS.length} Kapitel abgeschlossen
          </span>
        </div>
      </header>

      {PARTS.map((part) => (
        <section key={part.title} className="home__part">
          <header className="home__part-head">
            <h2>{part.title}</h2>
            <span className="muted">{part.question}</span>
          </header>
          <ol className="home__cards">
            {CHAPTERS.filter((c) => c.number >= part.from && c.number <= part.to).map((c) => {
              const p = progress[c.slug];
              return (
                <li key={c.slug}>
                  <a className="chapter-card" href={href.chapter(c.slug)}>
                    <span className="meta">
                      Kapitel {pad(c.number)} · ≈ {c.minutes} Min
                    </span>
                    <span className="chapter-card__title">{c.title}</span>
                    <span className="chapter-card__short muted">{c.short}</span>
                    <span className="chapter-card__status mono">
                      {p?.completed ? '✓ abgeschlossen' : p?.visited ? '◐ gelesen' : c.ready ? '○ neu' : `in Arbeit · ${c.phase}`}
                    </span>
                  </a>
                </li>
              );
            })}
          </ol>
        </section>
      ))}
    </article>
  );
}
