import { useMemo, useState } from 'react';
import { Figure, Slider } from '../../components/primitives';
import { EXAMPLES } from '../../content/examples';
import { distribution, mulberry32, sampleIndex, type SamplingParams } from '../../engine/sampling';
import type { ToyLanguageModel } from '../../engine/lm';
import { PromptInput } from './PromptInput';
import { pct, SamplingControls, WithToyModel } from './Shared';

interface GenToken {
  id: number;
  p: number;
}

function generate(model: ToyLanguageModel, prompt: string, params: SamplingParams, seed: number, n: number): GenToken[] {
  const ids = model.promptIds(prompt);
  const rng = mulberry32(seed);
  const out: GenToken[] = [];
  for (let i = 0; i < n && ids.length < model.maxContext; i++) {
    const d = distribution(model.logitsFor(ids), params);
    const tok = sampleIndex(d.final, rng());
    out.push({ id: tok, p: d.final[tok] });
    if (model.eosIds.includes(tok)) break;
    ids.push(tok);
  }
  return out;
}

function MiniPlaygroundInner({ model }: { model: ToyLanguageModel }) {
  const [prompt, setPrompt] = useState(EXAMPLES.toyPrompts[1]);
  const [params, setParams] = useState<SamplingParams>({ temperature: 0.8, topK: 0, topP: 0.95, minP: 0 });
  const [seed, setSeed] = useState(1);
  const [n, setN] = useState(10);
  const out = useMemo(() => generate(model, prompt, params, seed, n), [model, prompt, params, seed, n]);

  return (
    <div className="widget">
      <PromptInput model={model} value={prompt} onChange={setPrompt} />
      <SamplingControls value={params} onChange={setParams} vocabSize={model.vocabSize} seed={seed} onSeed={setSeed} showMinP />
      <div className="row-gap">
        <Slider label="Neue Tokens" value={n} min={1} max={20} step={1} onChange={setN} format={(v) => String(v)} ticks={19} />
        <button type="button" className="btn btn--primary" onClick={() => setSeed(seed + 1)}>
          Neu würfeln
        </button>
      </div>
      <p className="play-out" aria-live="polite">
        <span className="mono muted">{model.decode(model.promptIds(prompt))}</span>
        {out.map((t, i) => (
          <span key={i} className="play-tok" title={`Wahrscheinlichkeit beim Ziehen: ${pct(t.p)}`}>
            <span className="mono">{model.eosIds.includes(t.id) ? '⏎' : model.token(t.id).text}</span>
            <span className="play-tok__p mono">{pct(t.p, 0)}</span>
          </span>
        ))}
      </p>
      <p className="small muted">
        Gleicher Seed und gleiche Einstellungen ergeben immer denselben Text. „Neu würfeln“ wählt einen anderen Seed.
      </p>
    </div>
  );
}

export function MiniPlayground({ num = 'P.1' }: { num?: string }) {
  return (
    <Figure num={num} caption="Mini-Playground: eigener Prompt, alle Regler, Generierung mit dem Spielzeugmodell." stamp="toy">
      <WithToyModel>{(m) => <MiniPlaygroundInner model={m} />}</WithToyModel>
    </Figure>
  );
}
