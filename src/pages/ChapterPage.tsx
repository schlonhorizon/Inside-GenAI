import { useEffect, useState, type ComponentType } from 'react';
import { getChapter, loadChapter, neighbours, pad } from '../content/chapters';
import { useProgress } from '../state/progress';
import { href } from '../app/router';
import { NotFound } from './NotFound';

export function ChapterPage({ slug }: { slug: string }) {
  const meta = getChapter(slug);
  const [Content, setContent] = useState<ComponentType | null>(null);
  const markVisited = useProgress((s) => s.markVisited);
  const setCompleted = useProgress((s) => s.setCompleted);
  const completed = useProgress((s) => s.chapters[slug]?.completed ?? false);

  useEffect(() => {
    if (!meta) return;
    markVisited(slug);
    let alive = true;
    setContent(null);
    loadChapter(slug)?.then((m) => alive && setContent(() => m.default));
    return () => {
      alive = false;
    };
  }, [slug, meta, markVisited]);

  if (!meta) return <NotFound path={`kapitel/${slug}`} />;
  const { prev, next } = neighbours(slug);

  return (
    <article className="page chapter">
      <header className="chapter__head">
        <span className="meta">
          Kapitel {pad(meta.number)} · {meta.title} · ≈ {meta.minutes} Min
        </span>
        <h1>{meta.title}</h1>
        <p className="lead muted">{meta.short}</p>
        <div className="goals">
          <span className="meta">Lernziele</span>
          <ul>
            {meta.goals.map((g) => (
              <li key={g}>{g}</li>
            ))}
          </ul>
        </div>
      </header>

      <div className="prose">{Content ? <Content /> : <p className="meta loading">Lädt …</p>}</div>

      <footer className="chapter__foot">
        <button
          type="button"
          className={`btn ${completed ? '' : 'btn--primary'}`}
          aria-pressed={completed}
          onClick={() => setCompleted(slug, !completed)}
        >
          {completed ? '✓ Als abgeschlossen markiert' : 'Kapitel abschließen'}
        </button>
        <nav className="pager" aria-label="Kapitel blättern">
          {prev ? (
            <a className="pager__link" href={href.chapter(prev.slug)}>
              <span className="meta">← Kapitel {pad(prev.number)}</span>
              <span>{prev.title}</span>
            </a>
          ) : (
            <span />
          )}
          {next && (
            <a className="pager__link pager__link--next" href={href.chapter(next.slug)}>
              <span className="meta">Kapitel {pad(next.number)} →</span>
              <span>{next.title}</span>
            </a>
          )}
        </nav>
      </footer>
    </article>
  );
}
