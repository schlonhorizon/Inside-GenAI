import { useState } from 'react';

export interface ChipToken {
  text: string;
  id: number;
  special?: boolean;
  /** kein vollständiges Zeichen (z. B. ein Byte eines „ä“) */
  partial?: boolean;
  /** nicht im Vokabular (Toy-Modell) */
  unknown?: boolean;
}

/** Sichtbare Darstellung von Leerzeichen/Zeilenumbrüchen innerhalb eines Tokens. */
function show(text: string) {
  return text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥');
}

export function TokenChips({
  tokens,
  initialSelected = null,
  showIds = true,
  label,
}: {
  tokens: ChipToken[];
  initialSelected?: number | null;
  showIds?: boolean;
  label?: string;
}) {
  const [selected, setSelected] = useState<number | null>(initialSelected);
  return (
    <div className="chips" role="list" aria-label={label}>
      {tokens.map((t, i) => (
        <button
          key={i}
          type="button"
          role="listitem"
          className={[
            'chip',
            t.special ? 'chip--special' : i % 2 ? 'chip--b' : 'chip--a',
            t.partial ? 'chip--partial' : '',
            t.unknown ? 'chip--unknown' : '',
          ].join(' ')}
          aria-pressed={selected === i}
          title={t.partial ? 'Teil eines Zeichens (einzelne Bytes)' : t.unknown ? 'nicht im Vokabular → <unk>' : undefined}
          onClick={() => setSelected(selected === i ? null : i)}
        >
          <span className="chip__text mono">{show(t.text) || '∅'}</span>
          {showIds && <span className="chip__id mono">{t.id}</span>}
        </button>
      ))}
    </div>
  );
}
