import { useEffect, useId, useMemo, useState } from 'react';
import { Slider } from '../../components/primitives';
import { EXAMPLES } from '../../content/examples';
import { distribution, mulberry32, sampleIndex, type SamplingParams } from '../../engine/sampling';
import { StaleRequest, type LanguageModel, type ToyLanguageModel } from '../../engine/lm';
import { fitChat, renderChat, SMOL_SYSTEM, type ChatMessage } from '../../engine/chat';
import type { RealLanguageModel } from '../../engine/real';
import { ModelBar, ModelFigure, ModelPromptInput, promptIdsOrNull, usePromptPerKind, useWidgetModel } from './model';
import { pct, SamplingControls, showWhitespace, WithToyModel } from './Shared';

interface GenToken {
  id: number;
  p: number;
}

/**
 * Erzeugt bis zu n Tokens als abbrechbare Schleife und meldet jeden Zwischenstand.
 * Neue Eingaben brechen eine laufende Generierung ab (siehe `alive`).
 */
async function generate(
  model: LanguageModel,
  ids: number[],
  params: SamplingParams,
  seed: number,
  n: number,
  session: string,
  alive: () => boolean,
  onToken: (toks: GenToken[]) => void,
): Promise<GenToken[] | null> {
  const ctx = ids.slice();
  const rng = mulberry32(seed);
  const out: GenToken[] = [];
  for (let i = 0; i < n && ctx.length < model.maxContext; i++) {
    let logits: Float32Array;
    try {
      logits = await model.logitsFor(ctx, { session });
    } catch (e) {
      if (e instanceof StaleRequest) return null;
      throw e;
    }
    if (!alive()) return null;
    const d = distribution(logits, params);
    const tok = sampleIndex(d.final, rng());
    out.push({ id: tok, p: d.final[tok] });
    onToken(out.slice());
    if (model.eosIds.includes(tok)) break;
    ctx.push(tok);
  }
  return out;
}

/** Laufende Generierung als Zustand: Tokens, fertig, Fehler */
function useGeneration(model: LanguageModel, ids: number[] | null, params: SamplingParams, seed: number, n: number) {
  const session = useId();
  const key = ids ? `${model.id}|${ids.join(',')}|${JSON.stringify(params)}|${seed}|${n}` : null;
  const [state, setState] = useState<{ key: string | null; tokens: GenToken[]; done: boolean; error: string | null }>({
    key: null,
    tokens: [],
    done: false,
    error: null,
  });
  useEffect(() => {
    if (!ids || !key) return;
    let on = true;
    setState({ key, tokens: [], done: false, error: null });
    generate(model, ids, params, seed, n, session, () => on, (tokens) => on && setState({ key, tokens, done: false, error: null })).then(
      (tokens) => on && tokens && setState({ key, tokens, done: true, error: null }),
      (e: Error) => on && setState((s) => ({ ...s, done: true, error: e.message })),
    );
    return () => {
      on = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state.key === key ? state : { key, tokens: [], done: false, error: null };
}

function GeneratedText({ model, tokens }: { model: LanguageModel; tokens: GenToken[] }) {
  return (
    <>
      {tokens.map((t, i) => (
        <span key={i} className="play-tok" title={`Wahrscheinlichkeit beim Ziehen: ${pct(t.p)}`}>
          <span className="mono">{model.eosIds.includes(t.id) ? '⏎' : model.token(t.id).text}</span>
          <span className="play-tok__p mono">{pct(t.p, 0)}</span>
        </span>
      ))}
    </>
  );
}

/* ── Fortsetzen (Spielzeugmodell, Base) ───────────────────────────── */

function CompletionPlayground({ model, params, seed }: { model: LanguageModel; params: SamplingParams; seed: number }) {
  const { prompt, setPrompt, settled } = usePromptPerKind(model, EXAMPLES.toyPrompts[1]);
  const [n, setN] = useState(model.kind === 'real' ? 16 : 10);
  const ids = useMemo(() => promptIdsOrNull(model, settled), [model, settled]);
  const gen = useGeneration(model, ids, params, seed, n);
  return (
    <>
      <ModelPromptInput model={model} value={prompt} onChange={setPrompt} />
      <Slider label="Neue Tokens" value={n} min={1} max={model.kind === 'real' ? 48 : 20} step={1} onChange={setN} format={(v) => String(v)} ticks={model.kind === 'real' ? 12 : 19} />
      <p className="play-out" aria-live="polite" data-done={gen.done ? '1' : '0'}>
        <span className="mono muted">{model.decode(ids ?? [])}</span>
        <GeneratedText model={model} tokens={gen.tokens} />
        {!gen.done && ids && <span className="cursor">▮</span>}
      </p>
      {gen.error && <p className="widget-error small">{gen.error}</p>}
    </>
  );
}

/* ── Chat (Instruct) ─────────────────────────────────────────────── */

const CHAT_REPLY_TOKENS = 64;

function ChatPlayground({ model, params, seed }: { model: RealLanguageModel; params: SamplingParams; seed: number }) {
  const session = useId();
  const [system, setSystem] = useState(SMOL_SYSTEM);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState(EXAMPLES.chatPrompts[0]);
  const [reply, setReply] = useState<{ tokens: GenToken[]; running: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const [lastStop, setLastStop] = useState<'eos' | 'limit' | null>(null);
  const count = (t: string) => model.tokenizer.encodeIds(t).length;
  const fit = useMemo(() => fitChat(system, messages, count, model.promptLimit), [system, messages, model]);
  const running = !!reply?.running;

  // neue Nachricht abschicken: Kontext kürzen, dann Antwort Token für Token erzeugen
  const send = async () => {
    const text = input.trim();
    if (!text || running) return;
    const next: ChatMessage[] = [...messages, { role: 'user', content: text }];
    const f = fitChat(system, next, count, model.promptLimit);
    if (f.tooLong) {
      setError(`Deine Nachricht ist zu lang: ${f.tokens} Tokens samt System-Prompt, höchstens ${model.promptLimit} passen in den Kontext.`);
      return;
    }
    setError(null);
    setMessages(next);
    setInput('');
    const ids = model.tokenizer.encodeIds(renderChat([{ role: 'system', content: system }, ...f.kept]));
    setReply({ tokens: [], running: true });
    try {
      const toks = await generate(model, ids, params, seed + next.length, CHAT_REPLY_TOKENS, session, () => true, (t) => setReply({ tokens: t, running: true }));
      const final = (toks ?? []).filter((t) => !model.eosIds.includes(t.id));
      setLastStop(toks && toks.some((t) => model.eosIds.includes(t.id)) ? 'eos' : 'limit');
      setMessages([...next, { role: 'assistant', content: model.decode(final.map((t) => t.id)) }]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setReply(null);
    }
  };

  const shown = messages.map((m, i) => ({ ...m, dropped: i < fit.dropped }));
  return (
    <div className="chat" data-last-stop={lastStop ?? ''} data-running={running ? '1' : '0'}>
      <label className="meta" htmlFor={`sys-${session}`}>
        System-Prompt
      </label>
      <textarea id={`sys-${session}`} className="mono small" rows={2} value={system} onChange={(e) => setSystem(e.target.value)} disabled={running} />
      {fit.dropped > 0 && (
        <p className="small muted" role="status" data-chat-dropped={fit.dropped}>
          Ältere Nachrichten passen nicht mehr in den Kontext (höchstens {model.promptLimit} Tokens) – das Modell sieht sie nicht mehr. Warum
          das so ist, zeigt <a href="#/kapitel/kontext">Kapitel 11</a>.
        </p>
      )}
      {shown.map((m, i) => (
        <div key={i} className={`chat__msg chat__msg--${m.role} ${m.dropped ? 'chat__msg--dropped' : ''}`}>
          <span className="meta">{m.role === 'user' ? 'Du' : 'Modell'}</span>
          {'\n'}
          {m.content}
        </div>
      ))}
      {reply && (
        <div className="chat__msg chat__msg--assistant" aria-live="polite">
          <span className="meta">Modell</span>
          {'\n'}
          <GeneratedText model={model} tokens={reply.tokens} />
          <span className="cursor">▮</span>
        </div>
      )}
      <div className="real-prompt">
        <textarea
          aria-label="Nachricht"
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
      </div>
      <div className="row-gap">
        <button type="button" className="btn btn--primary" onClick={() => void send()} disabled={running || !input.trim()}>
          Senden
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => {
            setMessages([]);
            setError(null);
          }}
          disabled={running || !messages.length}
        >
          Neuer Chat
        </button>
        {EXAMPLES.chatPrompts.map((p) => (
          <button key={p} type="button" className="btn btn--small" onClick={() => setInput(p)} disabled={running}>
            {p}
          </button>
        ))}
        <span className="mono small muted">
          Kontext: {fit.tokens} / {model.promptLimit} Tokens
        </span>
      </div>
      {error && (
        <p className="widget-error small" role="alert">
          {error}
        </p>
      )}
      <button type="button" className="linklike small" onClick={() => setShowRaw(!showRaw)}>
        {showRaw ? 'Rohtext ausblenden' : 'Rohtext anzeigen: so sieht das Modell den Chat'}
      </button>
      {showRaw && <pre className="mono small chat__raw">{showWhitespace(renderChat([{ role: 'system', content: system }, ...fit.kept]))}</pre>}
    </div>
  );
}

/* ── Rahmen ───────────────────────────────────────────────────────── */

function MiniPlaygroundInner({ toy }: { toy: ToyLanguageModel }) {
  const model = useWidgetModel(toy);
  const [params, setParams] = useState<SamplingParams>({ temperature: 0.8, topK: 0, topP: 0.95, minP: 0 });
  const [seed, setSeed] = useState(1);
  const chat = model.kind === 'real' && (model as RealLanguageModel).isChat;

  return (
    <div className="widget">
      <ModelBar />
      {chat ? <ChatPlayground model={model as RealLanguageModel} params={params} seed={seed} /> : <CompletionPlayground model={model} params={params} seed={seed} />}
      <SamplingControls value={params} onChange={setParams} vocabSize={model.vocabSize} seed={seed} onSeed={setSeed} showMinP />
      <div className="row-gap">
        <button type="button" className="btn" onClick={() => setSeed(seed + 1)}>
          Neu würfeln
        </button>
      </div>
      <p className="small muted">
        Gleicher Seed und gleiche Einstellungen ergeben immer denselben Text. „Neu würfeln“ wählt einen anderen Seed.
        {chat && ' Im Chat-Modus rechnet das Instruct-Modell; es stoppt am Ende-Token seiner Antwort.'}
      </p>
    </div>
  );
}

export function MiniPlayground({ num = 'P.1' }: { num?: string }) {
  return (
    <ModelFigure num={num} caption={(m) => <>Mini-Playground: eigener Prompt, alle Regler, Generierung mit {m.real ? <strong>{m.label}</strong> : 'dem Spielzeugmodell'}.</>}>
      <WithToyModel>{(m) => <MiniPlaygroundInner toy={m} />}</WithToyModel>
    </ModelFigure>
  );
}
