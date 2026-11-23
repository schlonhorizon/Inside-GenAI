import { useEffect, useState } from 'react';
import { readStored, writeStored } from '../lib/theme';
import { Segmented, Slider } from './primitives';

export const DISPLAY_FONTS = {
  fraunces: { label: 'Fraunces', stack: "'Fraunces Variable', Georgia, serif", note: 'Variable · opsz, SOFT, WONK' },
  instrument: { label: 'Instrument Serif', stack: "'Instrument Serif', Georgia, serif", note: 'Regular + Italic, kondensiert' },
  young: { label: 'Young Serif', stack: "'Young Serif', Georgia, serif", note: 'ein Schnitt, freundlich-kräftig' },
} as const;
export const BODY_FONTS = {
  inter: { label: 'Inter', stack: "'Inter Variable', system-ui, sans-serif" },
  plex: { label: 'IBM Plex Sans', stack: "'IBM Plex Sans', system-ui, sans-serif" },
} as const;
export const MONO_FONTS = {
  plexmono: { label: 'IBM Plex Mono', stack: "'IBM Plex Mono', ui-monospace, monospace" },
  jetbrains: { label: 'JetBrains Mono', stack: "'JetBrains Mono', ui-monospace, monospace" },
} as const;

type DisplayKey = keyof typeof DISPLAY_FONTS;
type BodyKey = keyof typeof BODY_FONTS;
type MonoKey = keyof typeof MONO_FONTS;

/** Hält die aktuelle Schriftwahl und schreibt sie als CSS-Variablen auf <html>. */
export function useFontChoice() {
  const [display, setDisplay] = useState<DisplayKey>(() => readStored<DisplayKey>('igai.font.display', 'fraunces'));
  const [body, setBody] = useState<BodyKey>(() => readStored<BodyKey>('igai.font.body', 'inter'));
  const [mono, setMono] = useState<MonoKey>(() => readStored<MonoKey>('igai.font.mono', 'plexmono'));
  const [soft, setSoft] = useState(() => Number(readStored('igai.font.soft', '50')));
  const [wonk, setWonk] = useState(() => readStored('igai.font.wonk', '1') === '1');

  useEffect(() => {
    const r = document.documentElement.style;
    r.setProperty('--font-display', (DISPLAY_FONTS[display] ?? DISPLAY_FONTS.fraunces).stack);
    r.setProperty('--font-body', (BODY_FONTS[body] ?? BODY_FONTS.inter).stack);
    r.setProperty('--font-mono', (MONO_FONTS[mono] ?? MONO_FONTS.plexmono).stack);
    r.setProperty('--display-variation', `'SOFT' ${soft}, 'WONK' ${wonk ? 1 : 0}`);
    writeStored('igai.font.display', display);
    writeStored('igai.font.body', body);
    writeStored('igai.font.mono', mono);
    writeStored('igai.font.soft', String(soft));
    writeStored('igai.font.wonk', wonk ? '1' : '0');
  }, [display, body, mono, soft, wonk]);

  return { display, setDisplay, body, setBody, mono, setMono, soft, setSoft, wonk, setWonk };
}

type FontChoice = ReturnType<typeof useFontChoice>;

const HEAD = 'Wie aus einem Prompt eine Antwort wird';
const SUB = 'Kapitel 5 · Attention: Worauf achtet ein Token?';

export function FontLab({ f }: { f: FontChoice }) {
  return (
    <div className="fontlab">
      <div className="fontlab__display">
        {(Object.keys(DISPLAY_FONTS) as DisplayKey[]).map((k) => {
          const font = DISPLAY_FONTS[k];
          const variation = k === 'fraunces' ? `'SOFT' ${f.soft}, 'WONK' ${f.wonk ? 1 : 0}` : 'normal';
          return (
            <button
              key={k}
              type="button"
              className="font-card"
              aria-pressed={f.display === k}
              onClick={() => f.setDisplay(k)}
            >
              <span className="meta">
                {font.label} · {font.note}
              </span>
              <span className="font-card__head" style={{ fontFamily: font.stack, fontVariationSettings: variation }}>
                {HEAD}
              </span>
              <span className="font-card__sub" style={{ fontFamily: font.stack, fontVariationSettings: variation }}>
                {SUB}
              </span>
              <span className="font-card__sub font-card__italic" style={{ fontFamily: font.stack, fontVariationSettings: variation }}>
                <em>Next-Token-Prediction, Softmax &amp; Sampling</em>
              </span>
              <span className="font-card__pick mono">{f.display === k ? '● ausgewählt' : '○ auswählen'}</span>
            </button>
          );
        })}
      </div>

      <div className="fontlab__axes" aria-label="Achsen von Fraunces">
        <Slider
          label="Fraunces SOFT"
          value={f.soft}
          min={0}
          max={100}
          step={1}
          onChange={f.setSoft}
          format={(v) => v.toFixed(0)}
        />
        <Segmented
          label="Fraunces WONK"
          value={f.wonk ? 'on' : 'off'}
          options={[
            { value: 'off', label: 'aus' },
            { value: 'on', label: 'an' },
          ]}
          onChange={(v) => f.setWonk(v === 'on')}
        />
      </div>

      <div className="fontlab__pair">
        {(Object.keys(BODY_FONTS) as BodyKey[]).map((k) => (
          <button key={k} type="button" className="font-card font-card--body" aria-pressed={f.body === k} onClick={() => f.setBody(k)}>
            <span className="meta">Fließtext · {BODY_FONTS[k].label}</span>
            <span style={{ fontFamily: BODY_FONTS[k].stack }} className="font-card__para">
              Ein Sprachmodell gibt keine Antwort aus, sondern eine Wahrscheinlichkeitsverteilung über alle Tokens seines
              Vokabulars. Erst das Sampling macht daraus ein konkretes Wort – und genau dort greifen Temperatur, Top-k und
              Top-p. Zahlen im Text: 0,31 · 21,7 % · 1.024 Tokens.
            </span>
            <span className="font-card__pick mono">{f.body === k ? '● ausgewählt' : '○ auswählen'}</span>
          </button>
        ))}
      </div>

      <div className="fontlab__pair">
        {(Object.keys(MONO_FONTS) as MonoKey[]).map((k) => (
          <button key={k} type="button" className="font-card font-card--body" aria-pressed={f.mono === k} onClick={() => f.setMono(k)}>
            <span className="meta">Monospace · {MONO_FONTS[k].label}</span>
            <span style={{ fontFamily: MONO_FONTS[k].stack }} className="font-card__mono">
              {'<|im_start|>user↵ · ID 4218 · p=0.3121'}
              <br />
              {'[T × d_model] · 0123456789 · Il1| O0'}
            </span>
            <span className="font-card__pick mono">{f.mono === k ? '● ausgewählt' : '○ auswählen'}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
