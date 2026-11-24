import { useEffect, useMemo, useState } from 'react';
import katex from 'katex';
import { motion } from 'motion/react';
import { Figure } from '../../components/primitives';
import { EXAMPLES } from '../../content/examples';
import { distribution, mulberry32, rankOrder, sampleIndex, type SamplingParams } from '../../engine/sampling';
import type { LanguageModel, ToyLanguageModel } from '../../engine/lm';
import { BASE_MODEL } from '../../engine/real';
import { useReducedMotionPref } from '../../state/settings';
import { useModelStore } from '../../state/model';
import { StampNote } from '../../components/learn/Learn';
import {
  CalcState,
  ModelBar,
  ModelFigure,
  ModelPromptInput,
  promptIdsOrNull,
  useLogits,
  usePromptPerKind,
  useWidgetModel,
} from '../shared/model';
import { pct, SamplingControls, showWhitespace, WithToyModel } from '../shared/Shared';

/* ── (a) Leitbeispiel: echte, vorberechnete Verteilung ───────────── */

export function LeitbeispielBars() {
  const lb = EXAMPLES.leitbeispiel;
  const base = useModelStore((s) => (s.model && s.model.edition.model === BASE_MODEL ? s.model : null));
  const [wantLive, setWantLive] = useState(false);
  const live = useLeitLive(wantLive ? base : null, lb.prompt, lb.top.map((t) => t.token));
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
              {live && <span className="bar__h mono" data-live-token={t.token}>{live.probs.has(t.token) ? pct(live.probs.get(t.token)!) : '…'}</span>}
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
        {base ? (
          !wantLive ? (
            <button type="button" className="btn" onClick={() => setWantLive(true)} data-leit-live>
              Live nachrechnen ({base.edition.edition})
            </button>
          ) : (
            <StampNote kind={base.stamp}>
              Rechte Spalte: eben live im Browser gerechnet mit {base.label}
              {live?.ready && live.maxDiff !== null && (
                <>
                  {' '}
                  – größte Abweichung von den vorberechneten fp16-Werten:{' '}
                  <span className="mono" data-live-maxdiff>
                    {(live.maxDiff * 100).toLocaleString('de-DE', { maximumFractionDigits: 2 })} Prozentpunkte
                  </span>
                  {base.edition.edition === 'int8' ? ' (int8 rechnet mit gerundeten Gewichten).' : '.'}
                </>
              )}
            </StampNote>
          )
        ) : (
          <p className="small muted">
            Mit dem echten Modell (Base), geladen in Abb. 7.2, kannst du diese Werte hier live nachrechnen.
          </p>
        )}
      </div>
    </Figure>
  );
}

/** Leitbeispiel live mit dem echten Modell: Wahrscheinlichkeiten der Top-10-Tokens */
function useLeitLive(model: LanguageModel | null, prompt: string, tokens: string[]) {
  const ids = useMemo(() => (model ? model.promptIds(prompt) : null), [model, prompt]);
  const st = useLogits(model ?? NO_MODEL, ids);
  return useMemo(() => {
    if (!model) return null;
    const probs = new Map<string, number>();
    if (!st.logits || !st.fresh) return { probs, ready: false, maxDiff: null as number | null };
    const p = distribution(st.logits, { temperature: 1, topK: 0, topP: 1 }).probs;
    for (let i = 0; i < p.length; i++) {
      const t = model.token(i).text;
      if (tokens.includes(t)) probs.set(t, Math.max(probs.get(t) ?? 0, p[i]));
    }
    let maxDiff = 0;
    for (const t of EXAMPLES.leitbeispiel.top) maxDiff = Math.max(maxDiff, Math.abs((probs.get(t.token) ?? 0) - t.p));
    return { probs, ready: true, maxDiff };
  }, [model, st.logits, st.fresh, tokens]);
}

/** Platzhalter, damit useLogits ohne Modell nichts anfragt */
const NO_MODEL = { id: 'none', kind: 'toy', logitsFor: () => new Promise<Float32Array>(() => {}) } as unknown as LanguageModel;

/* ── (b) Interaktive Verteilung: Spielzeugmodell oder echtes Modell ─ */

const SHOW = 14;

function DistributionInner({ toy }: { toy: ToyLanguageModel }) {
  const model = useWidgetModel(toy);
  const reduce = useReducedMotionPref();
  const { prompt, setPrompt, settled } = usePromptPerKind(model, EXAMPLES.toyPrompts[0]);
  const [params, setParams] = useState<SamplingParams>({ temperature: 1, topK: 0, topP: 1, minP: 0 });
  const [seed, setSeed] = useState(42);
  const [draws, setDraws] = useState(0);
  const [sampled, setSampled] = useState<number | null>(null);
  const [hist, setHist] = useState<{ counts: Map<number, number>; n: number } | null>(null);

  const ids = useMemo(() => promptIdsOrNull(model, settled), [model, settled]);
  const st = useLogits(model, ids);
  const logits = st.logits;
  const d = useMemo(() => (logits ? distribution(logits, params) : null), [logits, params]);
  const order = useMemo(() => (d ? rankOrder(d.probs).slice(0, SHOW) : []), [d]);
  const maxP = d ? d.probs[order[0]] : 1;

  const resetDraws = () => {
    setSampled(null);
    setHist(null);
  };
  useEffect(resetDraws, [model.id]);
  const draw = () => {
    if (!d) return;
    const rng = mulberry32(seed * 1000 + draws);
    setSampled(sampleIndex(d.final, rng()));
    setDraws(draws + 1);
  };
  const drawMany = () => {
    if (!d) return;
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
        String.raw`p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}\quad\text{mit } T = ${params.temperature === 0 ? '0 \;(\\text{greedy})' : params.temperature.toFixed(2)}`,
        { displayMode: true, throwOnError: false },
      ),
    [params.temperature],
  );

  return (
    <div className="widget">
      <ModelBar />
      <ModelPromptInput
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
        <button type="button" className="btn btn--primary" onClick={draw} disabled={!d}>
          Würfeln
        </button>
        <button type="button" className="btn" onClick={drawMany} disabled={!d}>
          1000× würfeln
        </button>
        {sampled !== null && (
          <span className="small">
            Gezogen: <span className="mono accent-text">{showWhitespace(model.token(sampled).text)}</span>
          </span>
        )}
      </div>
      <CalcState state={st} model={model} />
      {d ? (
        <ol className={`dist__bars ${st.fresh ? '' : 'is-pending'}`} aria-label="Verteilung über die nächsten Tokens" data-model={model.id}>
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
      ) : (
        !st.error && <p className="meta loading">rechnet …</p>
      )}
      <p className="small muted">
        Balken: Wahrscheinlichkeit nach der Temperatur. Zahl rechts: Chance, gezogen zu werden (nach Top-k/Top-p, neu normiert).
        Schraffiert = abgeschnitten.{hist && ' Umrandet: Anteil bei 1000 Würfen.'}
      </p>
      {logits && <TemperatureCompare model={model} logits={logits} />}
    </div>
  );
}

function TemperatureCompare({ model, logits }: { model: LanguageModel; logits: Float32Array }) {
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
    <ModelFigure num="7.2" caption={(m) => <>Die Verteilung {m.real ? <>von <strong>{m.label}</strong></> : 'des Spielzeugmodells'} live: Regler ändern, was gezogen werden kann.</>}>
      <WithToyModel>{(m) => <DistributionInner toy={m} />}</WithToyModel>
    </ModelFigure>
  );
}
