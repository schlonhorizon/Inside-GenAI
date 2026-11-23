import { useEffect, useState } from 'react';
import { contrastRatio, deltaE2000, simulateCvd, type CvdType } from '../lib/color';
import { ALL_TOKENS, CHROMATIC_TOKENS, ROLE_GROUPS, TEXT_PAIRS } from '../lib/palette';
import { readTokens } from '../lib/theme';
import type { Theme } from '../state/settings';

const DE_MIN = 15;
const CVD_LABEL: Record<CvdType, string> = {
  protan: 'Protanopie',
  deutan: 'Deuteranopie',
  tritan: 'Tritanopie',
};

function Verdict({ ok }: { ok: boolean }) {
  return <span className={`verdict ${ok ? 'verdict--ok' : 'verdict--fail'}`}>{ok ? '✓' : '✗'}</span>;
}

export function PaletteLab({ theme }: { theme: Theme }) {
  const [v, setV] = useState<Record<string, string>>({});
  useEffect(() => {
    // nach dem Theme-Wechsel einen Frame warten, bis die Variablen greifen
    const id = requestAnimationFrame(() => setV(readTokens(ALL_TOKENS)));
    return () => cancelAnimationFrame(id);
  }, [theme]);

  if (!v['--bg']) return null;
  const bg = v['--bg'];

  // Prüfregel 1: Eindeutigkeit
  const seen = new Map<string, string[]>();
  for (const n of ALL_TOKENS) seen.set(v[n], [...(seen.get(v[n]) ?? []), n]);
  const dupes = [...seen.entries()].filter(([, names]) => names.length > 1);

  // Prüfregel 2: ΔE-Matrix
  const chroma = CHROMATIC_TOKENS;
  let minDe = Infinity;
  const matrix = chroma.map((a) =>
    chroma.map((b) => {
      if (a.name === b.name) return null;
      const d = deltaE2000(v[a.name], v[b.name]);
      minDe = Math.min(minDe, d);
      return d;
    }),
  );
  const failingPairs = matrix.flat().filter((d) => d !== null && d < DE_MIN).length / 2;

  return (
    <div className="palette">
      <div className="palette__rules">
        <div className="rule-card">
          <span className="meta">Regel 1 · Eindeutigkeit</span>
          <p>
            <Verdict ok={dupes.length === 0} />{' '}
            {dupes.length === 0
              ? `Alle ${ALL_TOKENS.length} Farbwerte sind genau einer Rolle zugeordnet.`
              : `Doppelt belegt: ${dupes.map(([hex, n]) => `${hex} (${n.join(', ')})`).join('; ')}`}
          </p>
        </div>
        <div className="rule-card">
          <span className="meta">Regel 2 · Unterscheidbarkeit</span>
          <p>
            <Verdict ok={failingPairs === 0} /> Kleinster Abstand zwischen bunten Rollenfarben: ΔE2000 ={' '}
            <span className="mono">{minDe.toFixed(1)}</span> (Ziel ≥ {DE_MIN})
            {failingPairs > 0 && ` · ${failingPairs} Paar(e) darunter`}
          </p>
        </div>
        <div className="rule-card">
          <span className="meta">Regel 3 · Kontrast</span>
          <p>
            <Verdict
              ok={
                TEXT_PAIRS.every(([f, b]) => contrastRatio(v[f], v[b]) >= 4.5) &&
                ROLE_GROUPS.every((g) => !g.minContrast || g.tokens.every((t) => contrastRatio(v[t.name], bg) >= g.minContrast!))
              }
            />{' '}
            Text ≥ 4,5 : 1, Rollenfarben gegen Hintergrund ≥ 3 : 1 (Details unten).
          </p>
        </div>
      </div>

      {ROLE_GROUPS.map((g) => (
        <div key={g.id} className="role">
          <div className="role__head">
            <h3>{g.title}</h3>
            <p className="muted">
              <strong className="role__meaning">{g.meaning}.</strong> {g.rule}
            </p>
          </div>
          <div className="role__swatches">
            {g.tokens.map((t) => {
              const c = contrastRatio(v[t.name], bg);
              return (
                <div key={t.name} className="swatch">
                  <span className="swatch__chip" style={{ background: `var(${t.name})` }} />
                  <span className="swatch__label">{t.label}</span>
                  <span className="mono swatch__meta">{t.name}</span>
                  <span className="mono swatch__meta">{v[t.name]}</span>
                  {g.minContrast && (
                    <span className="mono swatch__meta">
                      {c.toFixed(2)} : 1 <Verdict ok={c >= g.minContrast} />
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}

      <div className="palette__grid2">
        <div>
          <h3>Kontrast Text / Hintergrund</h3>
          <table className="table">
            <thead>
              <tr>
                <th>Paar</th>
                <th>Beispiel</th>
                <th>Kontrast</th>
              </tr>
            </thead>
            <tbody>
              {TEXT_PAIRS.map(([f, b]) => {
                const c = contrastRatio(v[f], v[b]);
                return (
                  <tr key={f + b}>
                    <td className="mono">
                      {f} / {b}
                    </td>
                    <td>
                      <span className="pair-sample" style={{ color: `var(${f})`, background: `var(${b})` }}>
                        Logit 3,14
                      </span>
                    </td>
                    <td className="mono">
                      {c.toFixed(2)} <Verdict ok={c >= 4.5} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div>
          <h3>ΔE2000 zwischen Rollenfarben</h3>
          <div className="matrix-wrap">
            <table className="table matrix">
              <thead>
                <tr>
                  <th />
                  {chroma.map((c) => (
                    <th key={c.name}>
                      <span className="matrix__dot" style={{ background: `var(${c.name})` }} title={c.name} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {chroma.map((a, i) => (
                  <tr key={a.name}>
                    <th className="mono matrix__rowhead">
                      <span className="matrix__dot" style={{ background: `var(${a.name})` }} /> {a.label}
                    </th>
                    {matrix[i].map((d, j) => (
                      <td key={j} className={`mono ${d !== null && d < DE_MIN ? 'matrix__low' : ''}`}>
                        {d === null ? '·' : d.toFixed(0)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">Gestrichelt umrandet = unter {DE_MIN}. Diese Paare dürfen nie im selben Panel auftreten.</p>
        </div>
      </div>

      <div>
        <h3>Simulation Farbfehlsichtigkeit</h3>
        <p className="muted small">Machado et al. (2009), volle Ausprägung. Darum trägt jede Farbrolle zusätzlich Form oder Symbol.</p>
        <div className="cvd">
          <div className="cvd__row">
            <span className="meta cvd__label">Normal</span>
            {chroma.map((c) => (
              <span key={c.name} className="cvd__chip" style={{ background: v[c.name] }} title={c.label} />
            ))}
          </div>
          {(Object.keys(CVD_LABEL) as CvdType[]).map((t) => {
            let m = Infinity;
            for (let i = 0; i < chroma.length; i++)
              for (let j = i + 1; j < chroma.length; j++)
                m = Math.min(m, deltaE2000(simulateCvd(v[chroma[i].name], t), simulateCvd(v[chroma[j].name], t)));
            return (
              <div key={t} className="cvd__row">
                <span className="meta cvd__label">{CVD_LABEL[t]}</span>
                {chroma.map((c) => (
                  <span key={c.name} className="cvd__chip" style={{ background: simulateCvd(v[c.name], t) }} title={c.label} />
                ))}
                <span className="mono small muted">min ΔE {m.toFixed(1)}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
