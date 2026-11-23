import type { ReactNode } from 'react';
import { loadToyModel, type ToyLanguageModel } from '../../engine/lm';
import type { SamplingParams } from '../../engine/sampling';
import { useAsync } from '../../hooks/useAsync';
import { Slider } from '../../components/primitives';

/* ── Toy-Modell laden ─────────────────────────────────────────────── */

export function WithToyModel({ children }: { children: (m: ToyLanguageModel) => ReactNode }) {
  const state = useAsync(() => loadToyModel('toy'));
  if (state.status === 'loading') return <p className="meta loading">Spielzeugmodell lädt …</p>;
  if (state.status === 'error') return <p className="widget-error">Spielzeugmodell konnte nicht geladen werden: {state.error.message}</p>;
  return <>{children(state.data)}</>;
}

/* ── Sampling-Regler ──────────────────────────────────────────────── */

export function SamplingControls({
  value,
  onChange,
  vocabSize,
  seed,
  onSeed,
  showMinP = false,
}: {
  value: SamplingParams;
  onChange: (p: SamplingParams) => void;
  vocabSize: number;
  seed?: number;
  onSeed?: (s: number) => void;
  showMinP?: boolean;
}) {
  const set = (patch: Partial<SamplingParams>) => onChange({ ...value, ...patch });
  return (
    <div className="sampling-controls">
      <Slider label="Temperatur" value={value.temperature} min={0} max={2} step={0.05} onChange={(v) => set({ temperature: v })} format={(v) => (v === 0 ? '0 (greedy)' : v.toFixed(2))} />
      <Slider
        label="Top-k"
        value={value.topK}
        min={0}
        max={Math.min(40, vocabSize)}
        step={1}
        onChange={(v) => set({ topK: v })}
        format={(v) => (v === 0 ? 'aus' : String(v))}
      />
      <Slider label="Top-p" value={value.topP} min={0.05} max={1} step={0.01} onChange={(v) => set({ topP: v })} format={(v) => (v >= 1 ? 'aus' : v.toFixed(2))} />
      {showMinP && (
        <Slider label="Min-p" value={value.minP ?? 0} min={0} max={0.5} step={0.01} onChange={(v) => set({ minP: v })} format={(v) => (v === 0 ? 'aus' : v.toFixed(2))} />
      )}
      {onSeed && seed !== undefined && (
        <label className="seed-input">
          <span className="meta">Seed</span>
          <input type="number" className="mono" value={seed} min={0} max={99999} onChange={(e) => onSeed(Math.max(0, Number(e.target.value) || 0))} />
        </label>
      )}
    </div>
  );
}


/* ── Darstellung von Tokens ───────────────────────────────────────── */

/** Leerzeichen und Zeilenumbrüche in Tokens sichtbar machen. */
export function showWhitespace(t: string): string {
  return t.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥');
}

export function pct(p: number, digits = 1): string {
  return `${(p * 100).toLocaleString('de-DE', { minimumFractionDigits: digits, maximumFractionDigits: digits })} %`;
}
