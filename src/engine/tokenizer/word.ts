/**
 * Wort-Tokenizer des Spielzeugmodells: klein schreiben, Wörter und Satzzeichen trennen.
 * Muss zu scripts/py/build_assets.py (WORD_RE) passen.
 */
export const WORD_RE = /[a-zäöüß]+|[.,!?]/g;

export interface WordToken {
  id: number;
  text: string;
  unknown: boolean;
  special: boolean;
}

export class WordTokenizer {
  readonly vocab: string[];
  private index: Map<string, number>;
  readonly unkId: number;
  readonly bosId: number;
  readonly eosId: number;
  readonly padId: number;

  constructor(vocab: string[]) {
    this.vocab = vocab;
    this.index = new Map(vocab.map((w, i) => [w, i]));
    const req = (w: string) => {
      const id = this.index.get(w);
      if (id === undefined) throw new Error(`Spezialtoken fehlt: ${w}`);
      return id;
    };
    this.unkId = req('<unk>');
    this.bosId = req('<bos>');
    this.eosId = req('<eos>');
    this.padId = req('<pad>');
  }

  /** Zerlegt Text; unbekannte Wörter werden als <unk> markiert (Originaltext bleibt sichtbar). */
  encode(text: string): WordToken[] {
    const words = text.toLowerCase().match(WORD_RE) ?? [];
    return words.map((w) => {
      const id = this.index.get(w);
      return { id: id ?? this.unkId, text: w, unknown: id === undefined, special: false };
    });
  }

  has(word: string): boolean {
    return this.index.has(word.toLowerCase());
  }

  token(id: number): WordToken {
    const text = this.vocab[id] ?? '<unk>';
    return { id, text, unknown: false, special: text.startsWith('<') };
  }

  decode(ids: number[]): string {
    let out = '';
    for (const id of ids) {
      const t = this.vocab[id];
      if (t === undefined || t.startsWith('<')) continue;
      out += /^[.,!?]$/.test(t) ? t : (out ? ' ' : '') + t;
    }
    return out;
  }

  /** normale Wörter (ohne Spezialtoken), z. B. für die Wortliste im Playground */
  get words(): string[] {
    return this.vocab.filter((w) => !w.startsWith('<'));
  }
}
