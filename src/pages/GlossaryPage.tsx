import { GLOSSARY, getTerm } from '../content/glossary';
import { getChapter } from '../content/chapters';
import { href } from '../app/router';

export function GlossaryPage({ activeId }: { activeId?: string }) {
  return (
    <article className="page glossary">
      <header className="page__head">
        <span className="meta">Nachschlagen</span>
        <h1>Glossar</h1>
        <p className="lead muted">
          Alle Fachbegriffe an einem Ort. Im Text sind sie unterstrichen: Fährst du mit der Maus darüber, erscheint die
          Kurzerklärung, ein Klick führt hierher.
        </p>
      </header>
      <dl className="glossary__list">
        {GLOSSARY.map((e) => {
          const ch = e.chapter ? getChapter(e.chapter) : undefined;
          return (
            <div key={e.id} id={e.id} className={`glossary__entry ${activeId === e.id ? 'is-active' : ''}`}>
              <dt>
                <span className="glossary__term">{e.term}</span>
                {e.alt && <span className="mono glossary__alt">{e.alt}</span>}
              </dt>
              <dd>
                <p>{e.short}</p>
                {e.long && <p className="muted">{e.long}</p>}
                <p className="glossary__links small">
                  {ch && (
                    <a href={href.chapter(ch.slug)}>
                      → Kapitel {ch.number}: {ch.title}
                    </a>
                  )}
                  {e.see?.map((s) => {
                    const t = getTerm(s);
                    return t ? (
                      <a key={s} href={href.glossary(s)}>
                        siehe {t.term}
                      </a>
                    ) : null;
                  })}
                </p>
              </dd>
            </div>
          );
        })}
      </dl>
    </article>
  );
}
