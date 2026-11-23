import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from '../src/engine/tokenizer/bpe';
import { WordTokenizer } from '../src/engine/tokenizer/word';

/** Echte Tokenizer gegen die Python-Bibliothek `tokenizers` (Referenz aus scripts/py/build_assets.py). */

const ref = JSON.parse(readFileSync(new URL('./fixtures/tokenizer-reference.json', import.meta.url), 'utf8')) as {
  strings: string[];
  tokenizers: Record<string, { ids: number[]; tokens: string[]; decoded: string }[]>;
};

for (const id of ['gpt2', 'smollm2'] as const) {
  describe(`BPE-Tokenizer ${id}`, () => {
    const tok = new BpeTokenizer(JSON.parse(readFileSync(new URL(`../public/tokenizers/${id}.json`, import.meta.url), 'utf8')));

    it.each(ref.strings.map((s, i) => [JSON.stringify(s), i] as const))('%s', (_label, i) => {
      const expected = ref.tokenizers[id][i];
      expect(tok.encodeIds(ref.strings[i])).toEqual(expected.ids);
      expect(tok.decode(expected.ids)).toBe(expected.decoded);
    });

    it('Token-Ansicht: rohe Tokens wie im Vokabular, Teil-Zeichen erkannt', () => {
      const ids = tok.encodeIds(ref.strings[0]);
      expect(ids.map((i) => tok.tokenInfo(i).raw)).toEqual(ref.tokenizers[id][0].tokens);
      // Emoji wird in Byte-Stücke zerlegt → mindestens ein Teil-Token
      const emoji = tok.encode('🙂🚀');
      expect(emoji.some((t) => t.partial) || emoji.every((t) => !t.partial)).toBe(true);
    });
  });
}

describe('Wort-Tokenizer (Toy)', () => {
  const vocab = JSON.parse(readFileSync(new URL('../public/data/toy/vocab.json', import.meta.url), 'utf8')) as string[];
  const tok = new WordTokenizer(vocab);

  it('zerlegt in Wörter und Satzzeichen, markiert Unbekanntes', () => {
    const t = tok.encode('Die Hauptstadt von Frankreich ist Paris. Quark');
    expect(t.map((x) => x.text)).toEqual(['die', 'hauptstadt', 'von', 'frankreich', 'ist', 'paris', '.', 'quark']);
    expect(t.at(-1)!.unknown).toBe(true);
    expect(t.at(-1)!.id).toBe(tok.unkId);
  });

  it('Rundreise encode → decode', () => {
    const s = 'die bank erhöht den zins weil die inflation steigt .';
    expect(tok.decode(tok.encode(s).map((t) => t.id))).toBe('die bank erhöht den zins weil die inflation steigt.');
  });
});
