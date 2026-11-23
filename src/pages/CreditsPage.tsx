import raw from '../../content/credits.yaml';
import npmLicenses from '../generated/npm-licenses.json';

interface CreditItem {
  name: string;
  what: string;
  by?: string;
  license: string;
  url?: string;
  licenseFile?: string;
}
interface CreditGroup {
  group: string;
  items: CreditItem[];
}

const GROUPS = raw as CreditGroup[];
const licenseUrl = (file: string) => `${import.meta.env.BASE_URL}licenses/${file}`;

export function CreditsPage() {
  return (
    <article className="page credits">
      <header className="page__head">
        <span className="meta">Transparenz</span>
        <h1>Quellen &amp; Lizenzen</h1>
        <p className="lead muted">
          Alles, was diese Seite nutzt, mit Lizenz und Lizenztext. Modelle und Datensätze kommen hinzu, sobald sie
          eingebunden sind.
        </p>
      </header>

      {GROUPS.map((g) => (
        <section key={g.group} className="credits__group">
          <h2>{g.group}</h2>
          <table className="table credits__table">
            <thead>
              <tr>
                <th>Komponente</th>
                <th>Verwendung</th>
                <th>Lizenz</th>
              </tr>
            </thead>
            <tbody>
              {g.items.map((it) => (
                <tr key={it.name}>
                  <td>
                    {it.url ? (
                      <a href={it.url} rel="noopener noreferrer" target="_blank">
                        {it.name}
                      </a>
                    ) : (
                      it.name
                    )}
                    {it.by && <span className="muted small"> · {it.by}</span>}
                  </td>
                  <td>{it.what}</td>
                  <td>
                    {it.licenseFile ? (
                      <a href={licenseUrl(it.licenseFile)} data-license-file={it.licenseFile}>
                        {it.license}
                      </a>
                    ) : (
                      it.license
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}

      <section className="credits__group">
        <h2>Direkte npm-Abhängigkeiten</h2>
        <p className="muted small">Automatisch beim Build aus den Paketangaben erzeugt.</p>
        <table className="table credits__table">
          <thead>
            <tr>
              <th>Paket</th>
              <th>Version</th>
              <th>Lizenz</th>
            </tr>
          </thead>
          <tbody>
            {npmLicenses.map((p) => (
              <tr key={p.name}>
                <td className="mono">{p.name}</td>
                <td className="mono">{p.version}</td>
                <td>{p.license}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </article>
  );
}
