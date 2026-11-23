import { useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import katex from 'katex';
import { Slider } from './primitives';

/** Illustrative Logits für „The central bank raised interest rates because …“ */
const LOGITS: [string, number][] = [
  [' inflation', 4.1],
  [' prices', 3.4],
  [' the', 2.9],
  [' of', 2.5],
  [' demand', 2.2],
  [' it', 1.8],
  [' wages', 1.3],
  [' growth', 0.9],
];

function softmax(z: number[], T: number) {
  const s = z.map((v) => v / T);
  const m = Math.max(...s);
  const e = s.map((v) => Math.exp(v - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export function DistributionDemo() {
  const reduce = useReducedMotion();
  const [T, setT] = useState(0.8);
  const [topP, setTopP] = useState(0.9);
  const [sampled, setSampled] = useState<number | null>(null);

  const probs = useMemo(() => softmax(LOGITS.map((l) => l[1]), T), [T]);
  // Top-p: kleinste Menge (nach Wahrscheinlichkeit sortiert) mit Summe ≥ p
  const keep = useMemo(() => {
    const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0]);
    const set = new Set<number>();
    let acc = 0;
    for (const [p, i] of order) {
      set.add(i);
      acc += p;
      if (acc >= topP) break;
    }
    return set;
  }, [probs, topP]);

  const draw = () => {
    const kept = [...keep];
    const total = kept.reduce((a, i) => a + probs[i], 0);
    let r = Math.random() * total;
    for (const i of kept) {
      r -= probs[i];
      if (r <= 0) {
        setSampled(i);
        return;
      }
    }
    setSampled(kept[kept.length - 1]);
  };

  const formula = useMemo(
    () =>
      katex.renderToString(`p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}} \\qquad T = ${T.toFixed(2)}`, {
        displayMode: true,
        throwOnError: false,
      }),
    [T],
  );

  const max = Math.max(...probs);

  return (
    <div className="dist">
      <div className="dist__controls">
        <Slider label="Temperatur T" value={T} min={0.1} max={2} step={0.05} onChange={setT} />
        <Slider label="Top-p" value={topP} min={0.1} max={1} step={0.01} onChange={setTopP} />
        <button type="button" className="btn btn--primary" onClick={draw}>
          Würfeln
        </button>
      </div>
      <div className="dist__formula" dangerouslySetInnerHTML={{ __html: formula }} />
      <ol className="dist__bars" aria-label="Wahrscheinlichkeiten der nächsten Tokens">
        {LOGITS.map(([tok], i) => {
          const p = probs[i];
          const cut = !keep.has(i);
          return (
            <li key={tok} className={`bar ${cut ? 'bar--cut' : ''} ${sampled === i ? 'bar--sampled' : ''}`}>
              <span className="bar__tok mono">{tok.replace(' ', '·')}</span>
              <span className="bar__track">
                <motion.span
                  className="bar__fill"
                  initial={false}
                  animate={{ width: `${(p / max) * 100}%` }}
                  transition={reduce ? { duration: 0 } : { duration: 0.28, ease: [0.2, 0.7, 0.2, 1] }}
                />
              </span>
              <span className="bar__p mono">{(p * 100).toFixed(1)} %</span>
            </li>
          );
        })}
      </ol>
      <p className="dist__note muted">
        Schraffiert = durch Top-p abgeschnitten. Akzent = gezogenes Token
        {sampled !== null ? (
          <>
            : <span className="mono accent-text">{LOGITS[sampled][0].replace(' ', '·')}</span>
          </>
        ) : (
          ' (noch nicht gewürfelt)'
        )}
        .
      </p>
    </div>
  );
}
