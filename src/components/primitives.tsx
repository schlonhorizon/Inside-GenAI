import type { ReactNode } from 'react';

/* ── Ehrlichkeits-Stempel ─────────────────────────────────────────── */

export type StampKind = 'real' | 'toy' | 'precomputed' | 'illustrative';

const STAMP_LABEL: Record<StampKind, string> = {
  real: 'Echt',
  toy: 'Echt · Toy',
  precomputed: 'Vorberechnet',
  illustrative: 'Illustrativ',
};

const STAMP_HINT: Record<StampKind, string> = {
  real: 'Live im Browser mit dem echten Sprachmodell berechnet',
  toy: 'Live und exakt mit dem Spielzeugmodell berechnet',
  precomputed: 'Echte Werte, vorab berechnet',
  illustrative: 'Vereinfacht – zeigt das Prinzip, keine echten Werte',
};

export function Stamp({ kind }: { kind: StampKind }) {
  return (
    <span className={`stamp stamp--${kind}`} title={STAMP_HINT[kind]}>
      {STAMP_LABEL[kind]}
    </span>
  );
}

/* ── Abbildungsrahmen ─────────────────────────────────────────────── */

export function Figure({
  num,
  caption,
  stamp,
  children,
}: {
  num: string;
  caption: ReactNode;
  stamp?: StampKind;
  children: ReactNode;
}) {
  return (
    <figure className="figure">
      <div className="figure__body">{children}</div>
      <figcaption className="figure__caption">
        <span className="mono figure__num">Abb. {num}</span>
        <span>{caption}</span>
        {stamp && <Stamp kind={stamp} />}
      </figcaption>
    </figure>
  );
}

/* ── Segmentierte Auswahl ─────────────────────────────────────────── */

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      <span className="meta segmented__label">{label}</span>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className="segmented__opt"
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ── Schalter ─────────────────────────────────────────────────────── */

export function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button type="button" role="switch" aria-checked={checked} className="toggle" onClick={() => onChange(!checked)}>
      <span className="meta">{label}</span>
      <span className="toggle__track" aria-hidden="true">
        <span className="toggle__thumb" />
      </span>
    </button>
  );
}

/* ── Regler (Lineal-Optik) ────────────────────────────────────────── */

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = (v) => v.toFixed(2),
  ticks = 10,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  ticks?: number;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="slider">
      <span className="slider__head">
        <span className="meta">{label}</span>
        <span className="mono slider__value">{format(value)}</span>
      </span>
      <span className="slider__track-wrap">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ ['--pct' as string]: `${pct}%` }}
        />
        <span className="slider__ticks" aria-hidden="true">
          {Array.from({ length: ticks + 1 }, (_, i) => (
            <span key={i} className={i % 5 === 0 ? 'major' : ''} />
          ))}
        </span>
      </span>
    </label>
  );
}

/* ── Callout-Nummer (für Fließtext) ───────────────────────────────── */

export function CalloutNum({ n }: { n: number }) {
  return (
    <span className="callout-num mono" aria-label={`Hinweis ${n}`}>
      {n}
    </span>
  );
}
