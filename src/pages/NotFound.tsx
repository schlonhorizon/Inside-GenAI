import { href } from '../app/router';

export function NotFound({ path }: { path: string }) {
  return (
    <article className="page">
      <span className="meta">Fehler 404</span>
      <h1>Diese Seite gibt es nicht</h1>
      <p className="lead muted">
        Die Adresse <code>#/{path.replace(/^\//, '')}</code> führt nirgendwohin.
      </p>
      <p>
        <a className="btn" href={href.home()}>
          Zur Übersicht
        </a>
      </p>
    </article>
  );
}
