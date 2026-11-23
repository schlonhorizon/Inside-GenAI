/**
 * Die Farbrollen im Einsatz: A/B-Linien, divergierende Balken, kategoriale Heads,
 * sequenzielle Heatmap. Alle Werte illustrativ.
 */

const AB_A = [0.62, 0.48, 0.55, 0.31, 0.44, 0.27, 0.36];
const AB_B = [0.62, 0.48, 0.21, 0.4, 0.18, 0.33, 0.12];
const DIV: [string, number][] = [
  ['die', -0.04],
  ['bank', 0.31],
  ['erhöht', 0.12],
  ['den', -0.02],
  ['zins', 0.24],
  ['weil', -0.09],
];
const HEAT = [
  [1, 0, 0, 0, 0],
  [0.55, 0.45, 0, 0, 0],
  [0.2, 0.62, 0.18, 0, 0],
  [0.1, 0.14, 0.5, 0.26, 0],
  [0.08, 0.41, 0.12, 0.04, 0.35],
];
const HEAT_LABELS = ['die', 'bank', 'erhöht', 'den', 'zins'];

const seqVar = (v: number) => `var(--seq-${Math.min(5, Math.max(1, Math.ceil(v * 5)))})`;

export function RolesInUse() {
  const w = 240;
  const h = 110;
  const x = (i: number) => 16 + (i * (w - 32)) / (AB_A.length - 1);
  const y = (v: number) => h - 16 - v * (h - 32);
  const line = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'} ${x(i)} ${y(v)}`).join(' ');

  return (
    <div className="roles-in-use">
      <div className="riu">
        <span className="meta">A/B · KL pro Schritt</span>
        <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Zwei Läufe A und B im Vergleich">
          <path d={`M 16 ${h - 16} H ${w - 16}`} className="svg-line svg-line--faint" />
          <path d={line(AB_A)} fill="none" stroke="var(--ab-a)" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" />
          <path
            d={line(AB_B)}
            fill="none"
            stroke="var(--ab-b)"
            strokeWidth={1.75}
            strokeDasharray="5 4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path d={`M ${x(2)} 10 V ${h - 16}`} className="svg-line svg-line--faint" strokeDasharray="2 3" />
          <text x={x(2) + 4} y="16" className="svg-mono svg-muted">
            Abzweig
          </text>
        </svg>
        <div className="legend mono small">
          <span>
            <i className="legend__line" style={{ borderColor: 'var(--ab-a)' }} /> Lauf A
          </span>
          <span>
            <i className="legend__line legend__line--dashed" style={{ borderColor: 'var(--ab-b)' }} /> Lauf B
          </span>
        </div>
      </div>

      <div className="riu">
        <span className="meta">Divergierend · Einfluss auf „inflation“</span>
        <ul className="div-bars">
          {DIV.map(([t, d]) => (
            <li key={t}>
              <span className="mono">{t}</span>
              <span className="div-bars__track">
                <span className="div-bars__mid" />
                <span
                  className="div-bars__fill"
                  style={{
                    width: `${Math.abs(d) * 150}%`,
                    background: d >= 0 ? 'var(--div-pos)' : 'var(--div-neg)',
                    [d >= 0 ? 'left' : 'right']: '50%',
                  }}
                />
              </span>
              <span className="mono small">
                {d >= 0 ? '▲ +' : '▼ −'}
                {Math.abs(d).toFixed(2)}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div className="riu">
        <span className="meta">Sequenziell · Attention (kausal)</span>
        <div className="heat" role="img" aria-label="Attention-Heatmap, je dunkler desto höheres Gewicht">
          {HEAT.map((row, i) => (
            <div key={i} className="heat__row">
              <span className="mono small heat__lab">{HEAT_LABELS[i]}</span>
              {row.map((v, j) => (
                <span
                  key={j}
                  className={`heat__cell ${j > i ? 'heat__cell--masked' : ''}`}
                  style={j <= i ? { background: seqVar(v) } : undefined}
                  title={j <= i ? v.toFixed(2) : 'maskiert'}
                />
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="riu">
        <span className="meta">Kategorial · Heads</span>
        <ul className="heads">
          {[
            ['Head 1', 'vorheriges Token', 1],
            ['Head 2', 'erstes Token', 2],
            ['Head 3', 'Induction', 3],
          ].map(([name, desc, c]) => (
            <li key={name as string}>
              <svg width="56" height="22" aria-hidden="true">
                <path
                  d="M 4 16 C 18 2, 38 2, 52 16"
                  fill="none"
                  stroke={`var(--cat-${c})`}
                  strokeWidth={2}
                  strokeLinecap="round"
                />
              </svg>
              <span className="mono small">{name}</span>
              <span className="small muted">{desc}</span>
            </li>
          ))}
          <li>
            <svg width="56" height="22" aria-hidden="true">
              <path d="M 4 16 C 18 2, 38 2, 52 16" fill="none" stroke="var(--ink-faint)" strokeWidth={1.25} strokeLinecap="round" />
            </svg>
            <span className="mono small">weitere Heads</span>
            <span className="small muted">nicht hervorgehoben</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
