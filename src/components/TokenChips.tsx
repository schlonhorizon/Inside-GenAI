import { useState } from 'react';

export interface ChipToken {
  text: string;
  id: number;
  special?: boolean;
}

/** Sichtbare Darstellung von Leerzeichen/Zeilenumbrüchen innerhalb eines Tokens. */
function show(text: string) {
  return text.replace(/ /g, '·').replace(/\n/g, '↵');
}

export function TokenChips({ tokens }: { tokens: ChipToken[] }) {
  const [selected, setSelected] = useState<number | null>(5);
  return (
    <div className="chips" role="list">
      {tokens.map((t, i) => (
        <button
          key={i}
          type="button"
          role="listitem"
          className={`chip ${t.special ? 'chip--special' : i % 2 ? 'chip--b' : 'chip--a'}`}
          aria-pressed={selected === i}
          onClick={() => setSelected(selected === i ? null : i)}
        >
          <span className="chip__text mono">{show(t.text)}</span>
          <span className="chip__id mono">{t.id}</span>
        </button>
      ))}
    </div>
  );
}
