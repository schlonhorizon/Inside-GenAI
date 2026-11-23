import { useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { ToyLanguageModel } from '../../engine/lm';
import { EXAMPLES } from '../../content/examples';
import { TokenChips } from '../../components/TokenChips';

/**
 * Prompt-Eingabe fürs Spielzeugmodell: Autovervollständigung aus dem Vokabular,
 * anklickbare Wortliste (gruppiert), unbekannte Wörter werden markiert.
 */
export function PromptInput({
  model,
  value,
  onChange,
  showVocab = true,
  presets = EXAMPLES.toyPrompts,
}: {
  model: ToyLanguageModel;
  value: string;
  onChange: (v: string) => void;
  showVocab?: boolean;
  presets?: string[];
}) {
  const [focus, setFocus] = useState(false);
  const [active, setActive] = useState(0);
  const [vocabOpen, setVocabOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const words = model.tokenizer.words;

  // Präfix = das gerade getippte (letzte) Wort
  const prefix = /([a-zäöüß]*)$/i.exec(value)?.[1]?.toLowerCase() ?? '';
  const suggestions = useMemo(
    () => (prefix ? words.filter((w) => w.startsWith(prefix) && w !== prefix).slice(0, 8) : []),
    [prefix, words],
  );
  const tokens = model.tokenize(value);
  const unknown = tokens.filter((t) => t.unknown);

  const accept = (w: string) => {
    const base = value.slice(0, value.length - prefix.length);
    onChange(base + w + ' ');
    setActive(0);
    inputRef.current?.focus();
  };
  const append = (w: string) => {
    const sep = value === '' || value.endsWith(' ') ? '' : ' ';
    onChange(value + sep + w + ' ');
    inputRef.current?.focus();
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (!suggestions.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((a) => (a + 1) % suggestions.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((a) => (a - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault();
      accept(suggestions[active] ?? suggestions[0]);
    } else if (e.key === 'Escape') setFocus(false);
  };

  const groups = useMemo(() => {
    const known = new Set(words);
    const used = new Set<string>();
    const gs = EXAMPLES.toyVocabGroups.map((g) => {
      const ws = g.words.filter((w) => known.has(w));
      ws.forEach((w) => used.add(w));
      return { name: g.name, words: ws };
    });
    const rest = words.filter((w) => !used.has(w)).sort((a, b) => a.localeCompare(b, 'de'));
    if (rest.length) gs.push({ name: 'Weitere', words: rest });
    return gs.filter((g) => g.words.length);
  }, [words]);

  return (
    <div className="prompt-input">
      <div className="prompt-input__row">
        <label className="field prompt-input__field">
          <span className="meta">Prompt (nur Wörter aus dem Vokabular des Spielzeugmodells)</span>
          <span className="combo">
            <input
              ref={inputRef}
              className="text-input mono"
              value={value}
              onChange={(e) => {
                onChange(e.target.value);
                setActive(0);
              }}
              onKeyDown={onKey}
              onFocus={() => setFocus(true)}
              onBlur={() => setTimeout(() => setFocus(false), 120)}
              role="combobox"
              aria-expanded={focus && suggestions.length > 0}
              aria-autocomplete="list"
              aria-controls="prompt-suggestions"
              spellCheck={false}
              autoComplete="off"
            />
            {focus && suggestions.length > 0 && (
              <ul className="combo__list" id="prompt-suggestions" role="listbox">
                {suggestions.map((w, i) => (
                  <li
                    key={w}
                    role="option"
                    aria-selected={i === active}
                    className="mono"
                    onMouseDown={(e) => {
                      e.preventDefault();
                      accept(w);
                    }}
                  >
                    {w}
                  </li>
                ))}
              </ul>
            )}
          </span>
        </label>
        {presets.length > 0 && (
          <label className="field">
            <span className="meta">Beispiele</span>
            <select className="select" value="" onChange={(e) => e.target.value && onChange(e.target.value)}>
              <option value="">Auswählen …</option>
              {presets.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      <TokenChips tokens={tokens.map((t) => ({ id: t.id, text: t.text, unknown: t.unknown }))} label="Prompt als Tokens" />
      {unknown.length > 0 && (
        <p className="small prompt-input__warn">
          Nicht im Vokabular: <span className="mono">{unknown.map((t) => t.text).join(', ')}</span> → das Modell sieht dort nur{' '}
          <code>&lt;unk&gt;</code>.
        </p>
      )}
      {showVocab && (
        <div className="vocab">
          <button type="button" className="btn btn--small" aria-expanded={vocabOpen} onClick={() => setVocabOpen(!vocabOpen)}>
            {vocabOpen ? 'Wortliste ausblenden' : `Wortliste zeigen (${words.length} Wörter)`}
          </button>
          {vocabOpen && (
            <div className="vocab__groups">
              {groups.map((g) => (
                <div key={g.name} className="vocab__group">
                  <span className="meta">{g.name}</span>
                  <div className="vocab__words">
                    {g.words.map((w) => (
                      <button key={w} type="button" className="vocab__word mono" onClick={() => append(w)}>
                        {w}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
