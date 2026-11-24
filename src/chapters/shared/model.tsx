import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { Figure, Segmented } from '../../components/primitives';
import { TokenChips } from '../../components/TokenChips';
import { EXAMPLES } from '../../content/examples';
import { PromptInput } from './PromptInput';
import { StaleRequest, type LanguageModel, type ToyLanguageModel } from '../../engine/lm';
import { BASE_MODEL, EDITIONS, editionLabel, findEdition, manifestUrl, ramBytes, workerClient } from '../../engine/real';
import { hasRealModels, keyForSource, sourceAvailable, useActiveRealModel, useModelStore, type Source } from '../../state/model';

/* ── Welches Modell rechnet in diesem Widget? ─────────────────────── */

/** Das echte Modell, wenn es zur seitenweiten Auswahl passt und geladen ist, sonst das Spielzeugmodell. */
export function useWidgetModel(toy: ToyLanguageModel): LanguageModel {
  const real = useActiveRealModel();
  return real ?? toy;
}

/* ── Logits mit „neueste Anfrage gewinnt“ ─────────────────────────── */

export interface LogitsState {
  /** zuletzt erhaltene Logits für DIESES Modell (ggf. noch von der vorigen Eingabe) */
  logits: Float32Array | null;
  /** gehören die Logits zur aktuellen Eingabe? */
  fresh: boolean;
  error: string | null;
  /** Rechenzeit der letzten Antwort in ms (nur echtes Modell) */
  ms: number | null;
}

/**
 * Logits für eine Tokenfolge. Jede Widget-Instanz ist eine eigene Sitzung (eigener KV-Cache).
 * Antworten auf überholte Anfragen werden verworfen; bis die neue Antwort da ist, bleiben die
 * vorigen Logits sichtbar (aber nur, wenn sie vom selben Modell stammen).
 */
export function useLogits(model: LanguageModel, ids: number[] | null): LogitsState {
  const session = useId();
  const key = ids ? `${model.id}|${ids.join(',')}` : null;
  const [state, setState] = useState<{ key: string; model: string; logits: Float32Array; ms: number | null } | null>(null);
  const [error, setError] = useState<{ key: string; message: string } | null>(null);
  const latest = useRef(0);

  useEffect(() => {
    if (!ids || !key) return;
    const req = ++latest.current;
    const t0 = performance.now();
    model.logitsFor(ids, { session }).then(
      (logits) => {
        if (req !== latest.current) return; // überholt
        setState({ key, model: model.id, logits, ms: model.kind === 'real' ? performance.now() - t0 : null });
        setError(null);
      },
      (e: Error) => {
        if (req !== latest.current || e instanceof StaleRequest) return;
        setError({ key, message: e.message });
      },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, session]);

  const own = state && state.model === model.id ? state : null;
  return {
    logits: own?.logits ?? null,
    fresh: !!own && own.key === key,
    error: error && error.key === key ? error.message : null,
    ms: own?.ms ?? null,
  };
}

/** Wert erst nach einer kurzen Pause übernehmen (Texteingabe beim echten Modell). */
export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    if (ms <= 0) {
      setV(value);
      return;
    }
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return ms <= 0 ? value : v;
}

/** Verzögerung der Texteingabe beim echten Modell */
export const REAL_DEBOUNCE_MS = 250;

/* ── Modellwahl und Laden ─────────────────────────────────────────── */

export const mb = (b: number) => `${Math.round(b / 1048576).toLocaleString('de-DE')} MB`;

const SOURCE_OPTIONS: { value: Source; label: string }[] = [
  { value: 'toy', label: 'Spielzeug' },
  { value: 'base', label: 'Echt · Base' },
  { value: 'instruct', label: 'Echt · Instruct' },
];

/** Prüft (im Worker), ob eine Fassung schon im Cache Storage liegt – für „ohne Download“. */
function useCachedShare(key: string | null): number | null {
  const [share, setShare] = useState<number | null>(null);
  useEffect(() => {
    setShare(null);
    if (!key) return;
    let alive = true;
    workerClient()
      .cached(manifestUrl(key))
      .then((r) => alive && r && setShare(r.total ? r.cached / r.total : 0))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [key]);
  return share;
}

function LoadButton({ editionKey, primary, label }: { editionKey: string; primary: boolean; label: string }) {
  const e = findEdition(editionKey);
  const load = useModelStore((s) => s.load);
  const cached = useCachedShare(e ? editionKey : null);
  if (!e) return null;
  const inCache = cached !== null && cached >= 0.999;
  return (
    <div className="modelbar__offer">
      <button type="button" className={`btn ${primary ? 'btn--primary' : ''}`} onClick={() => load(editionKey)} data-load={editionKey}>
        {label}
      </button>
      <span className="small muted">
        {inCache ? 'bereits gespeichert, kein Download' : `≈ ${mb(e.downloadBytes)} Download`} · ≈ {mb(ramBytes(e))} Arbeitsspeicher
        {e.fake && ' · Zufallsgewichte (lokaler Test)'}
      </span>
    </div>
  );
}

/**
 * Leiste über einem Widget: Auswahl Spielzeug / Echt (Base) / Echt (Instruct), Laden mit
 * Fortschritt und Abbrechen, Hinweis beim Rückfall auf das Spielzeugmodell.
 * Die Auswahl gilt für die ganze Seite, weil höchstens ein echtes Modell geladen sein kann.
 */
export function ModelBar({ allowInstruct = true }: { allowInstruct?: boolean }) {
  const s = useModelStore();
  if (!hasRealModels) {
    return (
      <p className="modelbar modelbar--none small muted">
        Das echte Modell ist in dieser Fassung der Seite nicht enthalten. Es rechnet das Spielzeugmodell.
      </p>
    );
  }
  const options = SOURCE_OPTIONS.filter((o) => sourceAvailable(o.value) && (allowInstruct || o.value !== 'instruct'));
  const source = !allowInstruct && s.source === 'instruct' ? 'toy' : s.source;
  const wanted = keyForSource(source, s.baseEdition);
  const loadedHere = wanted && s.model?.id === wanted;
  const loadingHere = s.status === 'loading' && s.loadingKey !== null;
  const p = s.progress;

  return (
    <div className="modelbar" data-model-status={s.status}>
      <Segmented label="Modell" value={source} options={options} onChange={(v) => s.setSource(v)} />

      {loadingHere && (
        <div className="modelbar__loading" role="status">
          <span className="small">
            {editionLabel(findEdition(s.loadingKey!) ?? { model: BASE_MODEL, edition: 'int8' })} wird geladen
            {p && p.total > 0 && (
              <>
                {' '}
                · <span className="mono">{mb(p.loaded)} von {mb(p.total)}</span>
                {p.fromCache > 0 && ' · aus dem Speicher des Browsers'}
                {p.phase === 'prepare' && ' · Gewichte werden vorbereitet …'}
              </>
            )}
          </span>
          <progress className="modelbar__progress" max={p?.total || 1} value={p?.loaded ?? 0} />
          <button type="button" className="btn btn--small" onClick={s.cancel}>
            Abbrechen
          </button>
        </div>
      )}

      {!loadingHere && source !== 'toy' && !loadedHere && (
        <div className="modelbar__offers">
          <span className="small muted">Noch nicht geladen – bis dahin rechnet das Spielzeugmodell.</span>
          {source === 'base' ? (
            <>
              <LoadButton editionKey={`${BASE_MODEL}/int8`} primary label="Echtes Modell laden (int8)" />
              <LoadButton editionKey={`${BASE_MODEL}/fp16`} primary={false} label="Exakte Fassung laden (fp16)" />
            </>
          ) : (
            <LoadButton editionKey={wanted!} primary label="Echtes Modell laden (int8)" />
          )}
        </div>
      )}

      {!loadingHere && loadedHere && s.model && (
        <p className="small muted modelbar__ready">
          <span className="mono">{s.model.label}</span> geladen
          {s.info?.loadMs !== undefined && ` in ${(s.info.loadMs / 1000).toLocaleString('de-DE', { maximumFractionDigits: 1 })} s`}
          {s.info?.fromCache ? ' (aus dem Speicher des Browsers)' : ''}
          {s.model.edition.fake && ' · Zufallsgewichte (lokaler Test)'}
          {source === 'base' && s.model.edition.edition === 'int8' && findEdition(`${BASE_MODEL}/fp16`) && (
            <>
              {' · '}
              <button type="button" className="linklike" onClick={() => s.load(`${BASE_MODEL}/fp16`)}>
                exakte Fassung laden
              </button>
            </>
          )}
        </p>
      )}

      {s.status === 'error' && s.error && (
        <p className="modelbar__error small" role="alert">
          Das echte Modell konnte nicht geladen werden: {s.error.message} Die Seite rechnet mit dem Spielzeugmodell weiter.
        </p>
      )}
    </div>
  );
}

export { EDITIONS };

/* ── Prompt-Eingabe ───────────────────────────────────────────────── */

/**
 * Freie Texteingabe für das echte Modell: BPE-Chips mit IDs, Tokenzahl und Limit, Beispiel-Prompts.
 */
export function RealPromptInput({
  model,
  value,
  onChange,
  presets = EXAMPLES.realPrompts,
}: {
  model: LanguageModel;
  value: string;
  onChange: (v: string) => void;
  presets?: string[];
}) {
  const tokens = model.tokenize(value);
  const over = tokens.length > model.promptLimit;
  return (
    <div className="real-prompt">
      <label className="meta" htmlFor={`rp-${model.id}`}>
        Prompt (Englisch funktioniert am besten)
      </label>
      <textarea id={`rp-${model.id}`} value={value} spellCheck={false} rows={2} onChange={(e) => onChange(e.target.value)} />
      <div className="row-gap">
        {presets.map((p) => (
          <button key={p} type="button" className="btn btn--small" onClick={() => onChange(p)}>
            {p}
          </button>
        ))}
      </div>
      <TokenChips tokens={tokens} label="Tokens des Prompts" />
      <p className="real-prompt__meta small muted">
        <span className="mono">
          {tokens.length} / {model.promptLimit} Tokens
        </span>
        {over && <span role="alert">Zu lang: Das echte Modell nimmt im Browser höchstens {model.promptLimit} Tokens.</span>}
      </p>
    </div>
  );
}

/** Eingabe passend zum Modell: Wortliste beim Spielzeugmodell, freier Text beim echten. */
export function ModelPromptInput({
  model,
  value,
  onChange,
  showVocab,
}: {
  model: LanguageModel;
  value: string;
  onChange: (v: string) => void;
  showVocab?: boolean;
}) {
  if (model.kind === 'toy') return <PromptInput model={model as ToyLanguageModel} value={value} onChange={onChange} showVocab={showVocab} />;
  return <RealPromptInput model={model} value={value} onChange={onChange} />;
}

/** Prompt-Zustand getrennt nach Modellart (Deutsch fürs Spielzeug, Englisch fürs echte Modell). */
export function usePromptPerKind(model: LanguageModel, toyDefault: string, realDefault = EXAMPLES.realPrompts[0]) {
  const [prompts, setPrompts] = useState({ toy: toyDefault, real: realDefault });
  const prompt = prompts[model.kind];
  const setPrompt = (v: string) => setPrompts((p) => ({ ...p, [model.kind]: v }));
  // beim echten Modell erst nach einer kurzen Tipp-Pause rechnen
  const settled = useDebounced(prompt, model.kind === 'real' ? REAL_DEBOUNCE_MS : 0);
  return { prompt, setPrompt, settled };
}

/** Prompt-IDs mit Prüfung des Limits (null = zu lang oder leer) */
export function promptIdsOrNull(model: LanguageModel, text: string): number[] | null {
  const ids = model.promptIds(text);
  if (!ids.length || ids.length > model.promptLimit + (model.bosId !== null ? 1 : 0)) return null;
  return ids;
}

/** Kurzer Statushinweis „rechnet …“ / Rechenzeit / Fehler */
export function CalcState({ state, model }: { state: LogitsState; model: LanguageModel }) {
  if (state.error) return <p className="widget-error small">{state.error}</p>;
  if (model.kind !== 'real') return null;
  return (
    <p className="calc-state mono muted" aria-live="polite" data-fresh={state.fresh ? '1' : '0'}>
      {state.fresh ? (state.ms !== null ? `gerechnet in ${Math.round(state.ms).toLocaleString('de-DE')} ms` : '') : 'rechnet …'}
    </p>
  );
}

/* ── Abbildung mit automatischem Stempel ─────────────────────────── */

/** Figure, deren Stempel und Bildunterschrift dem gerade rechnenden Modell folgen. */
export function ModelFigure({
  num,
  caption,
  children,
}: {
  num: string;
  caption: (m: { label: string; real: boolean }) => ReactNode;
  children: ReactNode;
}) {
  const real = useActiveRealModel();
  return (
    <Figure num={num} stamp={real ? real.stamp : 'toy'} caption={caption(real ? { label: real.label, real: true } : { label: 'Spielzeugmodell', real: false })}>
      {children}
    </Figure>
  );
}
