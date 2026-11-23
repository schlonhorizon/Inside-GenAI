import { BpeTokenizer } from './bpe.ts';

export type TokenizerId = 'gpt2' | 'smollm2';

export const TOKENIZER_LABEL: Record<TokenizerId, string> = {
  gpt2: 'GPT-2',
  smollm2: 'SmolLM2',
};

const cache = new Map<TokenizerId, Promise<BpeTokenizer>>();

export function tokenizerUrl(id: TokenizerId) {
  return `${import.meta.env.BASE_URL}tokenizers/${id}.json`;
}

/** Lädt einen echten Tokenizer (ca. 2–3 MB) bei Bedarf, einmal pro Sitzung. */
export function loadTokenizer(id: TokenizerId): Promise<BpeTokenizer> {
  let p = cache.get(id);
  if (!p) {
    p = fetch(tokenizerUrl(id))
      .then((r) => {
        if (!r.ok) throw new Error(`Tokenizer ${id} nicht gefunden (${r.status})`);
        return r.json();
      })
      .then((json) => new BpeTokenizer(json));
    p.catch(() => cache.delete(id));
    cache.set(id, p);
  }
  return p;
}
