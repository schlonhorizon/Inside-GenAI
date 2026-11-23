import { useMemo, useState } from 'react';
import katex from 'katex';
import { motion } from 'motion/react';
import { Figure } from '../../components/primitives';
import { EXAMPLES } from '../../content/examples';
import { distribution, mulberry32, rankOrder, sampleIndex, type SamplingParams } from '../../engine/sampling';
import type { ToyLanguageModel } from '../../engine/lm';
import { useReducedMotionPref } from '../../state/settings';
import { PromptInput } from '../shared/PromptInput';
import { pct, SamplingControls, showWhitespace, WithToyModel } from '../shared/Shared';

/* ── (a) Leitbeispiel: echte, vorberechnete Verteilung ───────────── */

export function LeitbeispielBars() {
  const lb = EXAMPLES.leitbeispiel;
  const rest = 1 - lb.top.reduce((s, t) => s + t.p, 0);
  const max = lb.top[0].p;
  return (
    <Figure
      num="7.1"
      caption={
        <>
          Nächstes Token nach „{lb.prompt}“: Top-10 von <strong>{lb.modelLabel}</strong>, {lb.precision}, Revision{' '}
          <span className="mono">{lb.revision.slice(0, 7)}</span>, vorberechnet im Engine-Spike.
        </>
      }
      stamp="precomputed"
    >
      <div className="widget">
        <p className="mono prompt-line">
          {lb.prompt}
          <span className="cursor">▮</span>
        </p>
        <ol className="dist__bars" aria-label="Wahrscheinlichkeiten der nächsten Tokens">
          {lb.top.map((t, i) => (
            <li key={t.token} className={`bar ${i === 1 ? 'bar--highlight' : ''}`}>
              <span className="bar__tok mono">{showWhitespace(t.token)}</span>
              <span className="bar__track">
                <span className="bar__fill" style={{ width: `${(t.p / max) * 100}%` }} />
              </span>
              <span className="bar__p mono">{pct(t.p)}</span>
            </li>
          ))}
          <li className="bar bar--rest">
            <span className="bar__tok muted">alle anderen</span>
            <span className="bar__track" />
            <span className="bar__p mono">{pct(rest)}</span>
          </li>
        </ol>
        <p className="small muted">
          Das richtige „ Paris“ (umrandet) ist nur zweitwahrscheinlichste Fortsetzung. Greedy würde das Modell schreiben: „{lb.greedy} …“
        </p>
      </div>
    </Figure>
  );
}

/* ── (b) Interaktive Verteilung am Spielzeugmodell ───────────────── */

const SHOW = 14;

function ToyDistributionInner({ model }: { model: ToyLanguageModel }) {
  const reduce = useReducedMotionPref();
  const [prompt, setPrompt] = useState(EXAMPLES.toyPrompts[0]);
  const [params, setParams] = useState<SamplingParams>({ temperature: 1, topK: 0, topP: 1, minP: 0 });
  const [seed, setSeed] = useState(42);
  const [draws, setDraws] = useState(0);
  const [sampled, setSampled] = useState<number | null>(null);
  const [hist, setHist] = useState<{ counts: Map<number, number>; n: number } | null>(null);

  const ids = useMemo(() => model.promptIds(prompt), [model, prompt]);
  const logits = useMemo(() => model.logitsFor(ids), [model, ids]);
  const d = useMemo(() => distribution(logits, params), [logits, params]);
  const order = useMemo(() => rankOrder(d.probs).slice(0, SHOW), [d]);
  const maxP = d.probs[order[0]];

  const resetDraws = () => {
    setSampled(null);
    setHist(null);
  };
  const draw = () => {
    const rng = mulberry32(seed * 1000 + draws);
    setSampled(sampleIndex(d.final, rng()));
    setDraws(draws + 1);
  };
  const drawMany = () => {
    const rng = mulberry32(seed);
    const counts = new Map<number, number>();
    const n = 1000;
    for (let i = 0; i < n; i++) {
      const k = sampleIndex(d.final, rng());
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    setHist({ counts, n });
  };
  const formula = useMemo(
    () =>
      katex.renderToString(
        String.raw`p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}\quad\text{mit } T = ${params.temperature === 0 ? '0 \\;(\\text{greedy})' : params.temperature.toFixed(2)}`,
        { displayMode: true, throwOnError: false },
      ),
    [params.temperature],
  );

  return (
    <div className="widget">
      <PromptInput
        model={model}
        value={prompt}
        onChange={(v) => {
          setPrompt(v);
          resetDraws();
        }}
        showVocab
      />
      <SamplingControls
        value={params}
        onChange={(p) => {
          setParams(p);
          resetDraws();
        }}
        vocabSize={model.vocabSize}
        seed={seed}
        onSeed={(s) => {
          setSeed(s);
          setDraws(0);
          resetDraws();
        }}
        showMinP
      />
      <div className="formula math-layer" dangerouslySetInnerHTML={{ __html: formula }} />
      <div className="row-gap">
        <button type="button" className="btn btn--primary" onClick={draw}>
          Würfeln
        </button>
        <button type="button" className="btn" onClick={drawMany}>
          1000× würfeln
        </button>
        {sampled !== null && (
          <span className="small">
            Gezogen: <span className="mono accent-text">{showWhitespace(model.token(sampled).text)}</span>
          </span>
        )}
      </div>
      <ol className="dist__bars" aria-label="Verteilung über die nächsten Tokens">
        {order.map((i) => {
          const cut = !d.kept[i];
          const h = hist ? (hist.counts.get(i) ?? 0) / hist.n : null;
          return (
            <li key={i} className={`bar ${cut ? 'bar--cut' : ''} ${sampled === i ? 'bar--sampled' : ''}`}>
              <span className="bar__tok mono">{showWhitespace(model.token(i).text)}</span>
              <span className="bar__track">
                <motion.span
                  className="bar__fill"
                  initial={false}
                  animate={{ width: `${(d.probs[i] / maxP) * 100}%` }}
                  transition={reduce ? { duration: 0 } : { duration: 0.25, ease: [0.2, 0.7, 0.2, 1] }}
                />
                {h !== null && <span className="bar__emp" style={{ width: `${(h / maxP) * 100}%` }} title={`gezogen: ${pct(h)}`} />}
              </span>
              <span className="bar__p mono">{cut ? '—' : pct(d.final[i])}</span>
              {h !== null && <span className="bar__h mono muted">{pct(h)}</span>}
            </li>
          );
        })}
      </ol>
      <p className="small muted">
        Balken: Wahrscheinlichkeit nach der Temperatur. Zahl rechts: Chance, gezogen zu werden (nach Top-k/Top-p, neu normiert).
        Schraffiert = abgeschnitten.{hist && ' Umrandet: Anteil bei 1000 Würfen.'}
      </p>
      <TemperatureCompare model={model} logits={logits} />
    </div>
  );
}

function TemperatureCompare({ model, logits }: { model: ToyLanguageModel; logits: Float32Array }) {
  return (
    <div className="tcompare">
      {[0, 1, 1.5].map((t) => {
        const d = distribution(logits, { temperature: t, topK: 0, topP: 1 });
        const order = rankOrder(d.final)
          .filter((i) => d.final[i] >= 0.005)
          .slice(0, 5);
        return (
          <div key={t} className="tcompare__col">
            <span className="meta">T = {t === 0 ? '0 (greedy)' : t.toLocaleString('de-DE')}</span>
            {order.map((i) => (
              <div key={i} className="tcompare__row">
                <span className="mono small">{showWhitespace(model.token(i).text)}</span>
                <span className="tcompare__bar" style={{ width: `${d.final[i] * 100}%` }} />
                <span className="mono small muted">{pct(d.final[i], 0)}</span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export function ToyDistribution() {
  return (
    <Figure num="7.2" caption="Die Verteilung des Spielzeugmodells live: Regler ändern, was gezogen werden kann." stamp="toy">
      <WithToyModel>{(m) => <ToyDistributionInner model={m} />}</WithToyModel>
    </Figure>
  );
}
