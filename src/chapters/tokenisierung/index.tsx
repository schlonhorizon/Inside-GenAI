import { useMemo, useState } from 'react';
import { Figure } from '../../components/primitives';
import { TokenChips } from '../../components/TokenChips';
import { EXAMPLES } from '../../content/examples';
import { loadTokenizer, TOKENIZER_LABEL, type TokenizerId } from '../../engine/tokenizer/load';
import type { BpeTokenizer } from '../../engine/tokenizer/bpe';
import { initBpe, stepBpe, totalTokens, vocabulary, type BpeState } from '../../engine/tokenizer/bpe-train';
import { useAsync } from '../../hooks/useAsync';

const IDS: TokenizerId[] = ['gpt2', 'smollm2'];
const fmt = (n: number, d = 1) => n.toLocaleString('de-DE', { maximumFractionDigits: d, minimumFractionDigits: d });

function useTokenizers() {
  return useAsync(() => Promise.all(IDS.map(loadTokenizer)));
}

function Loading({ state }: { state: ReturnType<typeof useTokenizers> }) {
  if (state.status === 'loading') return <p className="meta loading">Tokenizer laden (≈ 3 MB) …</p>;
  if (state.status === 'error') return <p className="widget-error">Tokenizer konnten nicht geladen werden: {state.error.message}</p>;
  return null;
}

/* ── (a) Tokenizer-Spielwiese ─────────────────────────────────────── */

function TokenRow({ tok, text, name }: { tok: BpeTokenizer; text: string; name: string }) {
  const tokens = useMemo(() => tok.encode(text), [tok, text]);
  const chars = Array.from(text).length;
  return (
    <div className="tok-row">
      <div className="tok-row__head">
        <strong>{name}</strong>
        <span className="mono small muted">
          {tokens.length} Tokens · {chars} Zeichen · {tokens.length ? fmt(chars / tokens.length) : '–'} Zeichen/Token · Vokabular{' '}
          {tok.vocabSize.toLocaleString('de-DE')}
        </span>
      </div>
      <TokenChips tokens={tokens.map((t) => ({ id: t.id, text: t.text, special: t.special, partial: t.partial }))} label={`Tokens ${name}`} />
    </div>
  );
}

export function TokenizerPlayground() {
  const state = useTokenizers();
  const [text, setText] = useState('Die Zentralbank erhöht den Leitzins.');
  return (
    <Figure num="2.1" caption="Dein Text, zerlegt von zwei echten Tokenizern. Die Zahl unter jedem Stück ist seine Token-ID." stamp="real">
      <div className="widget">
        <label className="field">
          <span className="meta">Text</span>
          <textarea className="text-input" rows={2} value={text} onChange={(e) => setText(e.target.value)} maxLength={600} />
        </label>
        <Loading state={state} />
        {state.status === 'ready' && IDS.map((id, i) => <TokenRow key={id} tok={state.data[i]} text={text} name={TOKENIZER_LABEL[id]} />)}
        <p className="small muted">
          · steht für ein Leerzeichen, ↵ für einen Zeilenumbruch. Gepunktete Chips sind nur Teile eines Zeichens (einzelne Bytes, z. B. bei
          Emojis).
        </p>
      </div>
    </Figure>
  );
}

/* ── (b) Deutsch vs. Englisch ─────────────────────────────────────── */

export function LanguageCompare() {
  const state = useTokenizers();
  return (
    <Figure num="2.2" caption="Derselbe Inhalt auf Deutsch und Englisch: Deutsch braucht meist mehr Tokens." stamp="real">
      <div className="widget">
        <Loading state={state} />
        {state.status === 'ready' && (
          <table className="table lang-table">
            <thead>
              <tr>
                <th>Satz</th>
                {IDS.map((id) => (
                  <th key={id}>{TOKENIZER_LABEL[id]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {EXAMPLES.languagePairs.map((pair) => (
                <tr key={pair.de}>
                  <td>
                    <div>
                      <span className="mono small muted">DE</span> {pair.de}
                    </div>
                    <div>
                      <span className="mono small muted">EN</span> {pair.en}
                    </div>
                  </td>
                  {IDS.map((id, i) => {
                    const de = state.data[i].encodeIds(pair.de).length;
                    const en = state.data[i].encodeIds(pair.en).length;
                    const max = Math.max(de, en);
                    return (
                      <td key={id} className="lang-bars">
                        <div className="lang-bar">
                          <span className="lang-bar__fill" style={{ width: `${(de / max) * 100}%` }} />
                          <span className="mono small">{de}</span>
                        </div>
                        <div className="lang-bar lang-bar--en">
                          <span className="lang-bar__fill" style={{ width: `${(en / max) * 100}%` }} />
                          <span className="mono small">{en}</span>
                        </div>
                        <span className="mono small muted">{en ? `× ${fmt(de / en, 2)}` : ''}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Figure>
  );
}

/* ── (c) BPE live ─────────────────────────────────────────────────── */

export function BpeLive() {
  const [corpus, setCorpus] = useState(EXAMPLES.bpeCorpus);
  const [history, setHistory] = useState<BpeState[]>(() => [initBpe(EXAMPLES.bpeCorpus)]);
  const state = history[history.length - 1];
  const reset = (text: string) => {
    setCorpus(text);
    setHistory([initBpe(text)]);
  };
  const step = (n = 1) => {
    let h = history.slice();
    for (let i = 0; i < n; i++) {
      const next = stepBpe(h[h.length - 1]);
      if (!next) break;
      h = [...h, next];
    }
    setHistory(h);
  };
  const last = state.merges[state.merges.length - 1];
  return (
    <Figure num="2.3" caption="BPE lernt sein Vokabular: Das häufigste benachbarte Paar wird zu einem neuen Token verschmolzen." stamp="toy">
      <div className="widget">
        <label className="field">
          <span className="meta">Mini-Korpus</span>
          <input className="text-input" value={corpus} onChange={(e) => reset(e.target.value)} />
        </label>
        <div className="row-gap">
          <button type="button" className="btn btn--primary" onClick={() => step(1)}>
            Nächster Merge
          </button>
          <button type="button" className="btn" onClick={() => step(5)}>
            5 Merges
          </button>
          <button type="button" className="btn" onClick={() => setHistory(history.slice(0, -1))} disabled={history.length < 2}>
            Zurück
          </button>
          <button type="button" className="btn" onClick={() => reset(corpus)}>
            Neu
          </button>
          <span className="mono small muted">
            Merges {state.merges.length} · Tokens im Text {totalTokens(state)} · Vokabular {vocabulary(state).length}
          </span>
        </div>
        {last && (
          <p className="bpe-last">
            Zuletzt verschmolzen:{' '}
            <span className="mono">
              {last.pair[0]} + {last.pair[1]} → <strong>{last.pair.join('')}</strong>
            </span>{' '}
            <span className="muted small">({last.count}× im Text)</span>
          </p>
        )}
        <div className="bpe-words">
          {state.words.map((w, i) => (
            <div key={i} className="bpe-word">
              <TokenChips tokens={w.symbols.map((s, j) => ({ id: j, text: s }))} showIds={false} />
              {w.count > 1 && <span className="mono small muted">× {w.count}</span>}
            </div>
          ))}
        </div>
      </div>
    </Figure>
  );
}

/* ── (d) Buchstaben zählen ────────────────────────────────────────── */

export function LetterCount() {
  const state = useTokenizers();
  const [reveal, setReveal] = useState(false);
  const { word, letter } = EXAMPLES.letterCount;
  const count = Array.from(word).filter((c) => c === letter).length;
  return (
    <Figure num="2.4" caption={`Wie viele „${letter}“ stecken in „${word}“? Das Modell sieht keine Buchstaben, sondern diese Stücke.`} stamp="real">
      <div className="widget">
        <Loading state={state} />
        {state.status === 'ready' &&
          IDS.map((id, i) => (
            <div key={id} className="tok-row">
              <div className="tok-row__head">
                <strong>{TOKENIZER_LABEL[id]}</strong>
              </div>
              {[word, ` ${word}`].map((w) => (
                <TokenChips key={w} tokens={state.data[i].encode(w).map((t) => ({ id: t.id, text: t.text }))} />
              ))}
            </div>
          ))}
        <button type="button" className="btn" onClick={() => setReveal(!reveal)}>
          {reveal ? 'Ausblenden' : 'Auflösung'}
        </button>
        {reveal && (
          <p>
            Es sind <strong>{count}</strong>. Für dich offensichtlich, für das Modell nicht: Es bekommt nur die Token-IDs und muss sich
            die Schreibweise jedes Stücks erst „gemerkt“ haben.
          </p>
        )}
      </div>
    </Figure>
  );
}
