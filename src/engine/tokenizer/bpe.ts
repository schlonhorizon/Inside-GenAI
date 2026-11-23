/**
 * Byte-Level-BPE-Tokenizer nach dem Format von Hugging Face `tokenizer.json`
 * (GPT-2, SmolLM2, Llama-3 u. a.). Unterstützt:
 *   - Added/Special Tokens (werden vor allem anderen exakt abgetrennt)
 *   - Pre-Tokenizer: ByteLevel (mit GPT-2-Regex), Split (Regex/String), Digits, Sequence
 *   - BPE mit Merge-Rängen, Cache pro Wortstück
 * Ergebnis wird in test/tokenizer.test.ts gegen die Python-Bibliothek `tokenizers` geprüft.
 */

export interface BpeToken {
  id: number;
  /** Token im Byte-Level-Alphabet, wie im Vokabular (z. B. "Ġworld") */
  raw: string;
  /** lesbare Darstellung (Leerzeichen als ␣ nicht ersetzt – das macht die Anzeige) */
  text: string;
  special: boolean;
  /** true, wenn das Token kein vollständiges UTF-8-Zeichen ergibt (z. B. halbes „ä“) */
  partial: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

const GPT2_REGEX = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** GPT-2-Abbildung Byte → druckbares Unicode-Zeichen (und zurück). */
function byteTables() {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const enc: string[] = new Array(256);
  const dec = new Map<string, number>();
  bs.forEach((b, i) => {
    const ch = String.fromCharCode(cs[i]);
    enc[b] = ch;
    dec.set(ch, b);
  });
  return { enc, dec };
}
const BYTES = byteTables();
const utf8 = new TextEncoder();
const utf8dec = new TextDecoder('utf-8', { fatal: false });
const utf8strict = new TextDecoder('utf-8', { fatal: true });

function escapeRegex(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Teilt Text an einem Regex; behavior wie in HF (Isolated, Removed, MergedWithPrevious, MergedWithNext). */
function splitBy(text: string, re: RegExp, behavior: string, invert = false): string[] {
  const out: string[] = [];
  let last = 0;
  const flags = re.flags.includes('g') ? re.flags : re.flags + 'g';
  const g = new RegExp(re.source, flags);
  const parts: { s: string; match: boolean }[] = [];
  for (const m of text.matchAll(g)) {
    if (m[0].length === 0) continue;
    if (m.index! > last) parts.push({ s: text.slice(last, m.index), match: false });
    parts.push({ s: m[0], match: true });
    last = m.index! + m[0].length;
  }
  if (last < text.length) parts.push({ s: text.slice(last), match: false });
  if (invert) parts.forEach((p) => (p.match = !p.match));
  for (const p of parts) {
    if (behavior === 'Removed' && p.match) continue;
    if (behavior === 'MergedWithPrevious' && p.match && out.length) {
      out[out.length - 1] += p.s;
      continue;
    }
    if (behavior === 'MergedWithNext' && p.match) {
      out.push(p.s);
      continue;
    }
    if (behavior === 'MergedWithNext' && out.length && parts[parts.indexOf(p) - 1]?.match) {
      out[out.length - 1] += p.s;
      continue;
    }
    out.push(p.s);
  }
  return out;
}

export class BpeTokenizer {
  readonly vocab: Map<string, number>;
  readonly idToToken: string[];
  readonly vocabSize: number;
  private ranks = new Map<string, number>();
  private cache = new Map<string, number[]>();
  private added: { content: string; id: number; special: boolean }[];
  private addedRe: RegExp | null;
  private preTok: Json;
  private usesByteLevel = false;
  private addPrefixSpace = false;
  private specialIds = new Set<number>();

  constructor(json: Json) {
    const model = json.model;
    // ältere Dateien (z. B. GPT-2) lassen "type" weg; entscheidend sind vocab + merges
    if ((model?.type && model.type !== 'BPE') || !model?.merges) throw new Error(`Nur BPE wird unterstützt, gefunden: ${model?.type}`);
    this.vocab = new Map(Object.entries(model.vocab as Record<string, number>));
    this.added = (json.added_tokens ?? []).map((t: Json) => ({ content: t.content, id: t.id, special: !!t.special }));
    for (const t of this.added) {
      this.vocab.set(t.content, t.id);
      if (t.special) this.specialIds.add(t.id);
    }
    let max = 0;
    for (const id of this.vocab.values()) max = Math.max(max, id);
    this.idToToken = new Array(max + 1);
    for (const [tok, id] of this.vocab) this.idToToken[id] = tok;
    this.vocabSize = max + 1;
    (model.merges as (string | [string, string])[]).forEach((m, i) => {
      const key = Array.isArray(m) ? `${m[0]} ${m[1]}` : m;
      this.ranks.set(key, i);
    });
    const sorted = this.added.map((t) => t.content).sort((a, b) => b.length - a.length);
    this.addedRe = sorted.length ? new RegExp(sorted.map(escapeRegex).join('|'), 'g') : null;
    this.preTok = json.pre_tokenizer;
    const scan = (p: Json) => {
      if (!p) return;
      if (p.type === 'ByteLevel') {
        this.usesByteLevel = true;
        if (p.add_prefix_space) this.addPrefixSpace = true;
      }
      if (p.type === 'Sequence') p.pretokenizers.forEach(scan);
    };
    scan(this.preTok);
    if (json.decoder?.type === 'ByteLevel') this.usesByteLevel = true;
  }

  private applyPre(p: Json, pieces: string[]): string[] {
    if (!p) return pieces;
    switch (p.type) {
      case 'Sequence':
        return p.pretokenizers.reduce((acc: string[], q: Json) => this.applyPre(q, acc), pieces);
      case 'ByteLevel':
        return pieces.flatMap((s) => {
          let t = s;
          if (p.add_prefix_space && !t.startsWith(' ')) t = ' ' + t;
          if (p.use_regex === false) return [t];
          return t.match(GPT2_REGEX) ?? [];
        });
      case 'Digits':
        return pieces.flatMap((s) => splitBy(s, p.individual_digits ? /\p{N}/u : /\p{N}+/u, 'Isolated'));
      case 'Split': {
        const re = p.pattern.Regex !== undefined ? new RegExp(p.pattern.Regex, 'u') : new RegExp(escapeRegex(p.pattern.String));
        return pieces.flatMap((s) => splitBy(s, re, p.behavior ?? 'Isolated', !!p.invert));
      }
      case 'Whitespace':
        return pieces.flatMap((s) => s.match(/\w+|[^\w\s]+/gu) ?? []);
      default:
        throw new Error(`Pre-Tokenizer ${p.type} wird nicht unterstützt`);
    }
  }

  private toByteLevel(s: string): string {
    if (!this.usesByteLevel) return s;
    let out = '';
    for (const b of utf8.encode(s)) out += BYTES.enc[b];
    return out;
  }

  private bpe(piece: string): number[] {
    const hit = this.cache.get(piece);
    if (hit) return hit;
    let syms = Array.from(piece);
    while (syms.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < syms.length - 1; i++) {
        const r = this.ranks.get(`${syms[i]} ${syms[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const a = syms[best];
      const b = syms[best + 1];
      const merged: string[] = [];
      for (let i = 0; i < syms.length; i++) {
        if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) {
          merged.push(a + b);
          i++;
        } else merged.push(syms[i]);
      }
      syms = merged;
    }
    const ids = syms.map((s) => {
      const id = this.vocab.get(s);
      if (id === undefined) throw new Error(`Token nicht im Vokabular: ${JSON.stringify(s)}`);
      return id;
    });
    if (this.cache.size < 50000) this.cache.set(piece, ids);
    return ids;
  }

  /** Text → Token-IDs (ohne automatisch angefügte Special Tokens). */
  encodeIds(text: string): number[] {
    const out: number[] = [];
    const segments: { s: string; added?: number }[] = [];
    if (this.addedRe) {
      let last = 0;
      for (const m of text.matchAll(this.addedRe)) {
        if (m.index! > last) segments.push({ s: text.slice(last, m.index) });
        segments.push({ s: m[0], added: this.vocab.get(m[0]) });
        last = m.index! + m[0].length;
      }
      if (last < text.length) segments.push({ s: text.slice(last) });
    } else segments.push({ s: text });
    for (const seg of segments) {
      if (seg.added !== undefined) {
        out.push(seg.added);
        continue;
      }
      if (!seg.s) continue;
      for (const piece of this.applyPre(this.preTok, [seg.s])) {
        if (piece) out.push(...this.bpe(this.toByteLevel(piece)));
      }
    }
    return out;
  }

  /** Bytes eines Tokens (Byte-Level-Zeichen → Bytes). */
  tokenBytes(id: number): Uint8Array {
    const raw = this.idToToken[id] ?? '';
    if (this.specialIds.has(id) || !this.usesByteLevel) return utf8.encode(raw);
    const bytes: number[] = [];
    for (const ch of raw) {
      const b = BYTES.dec.get(ch);
      if (b !== undefined) bytes.push(b);
      else bytes.push(...utf8.encode(ch));
    }
    return new Uint8Array(bytes);
  }

  tokenInfo(id: number): BpeToken {
    const raw = this.idToToken[id] ?? '';
    const special = this.specialIds.has(id);
    const bytes = this.tokenBytes(id);
    let text: string;
    let partial = false;
    try {
      text = utf8strict.decode(bytes);
    } catch {
      partial = true;
      text = Array.from(bytes, (b) => (b < 0x80 ? String.fromCharCode(b) : `<${b.toString(16).toUpperCase()}>`)).join('');
    }
    return { id, raw, text, special, partial };
  }

  encode(text: string): BpeToken[] {
    return this.encodeIds(text).map((id) => this.tokenInfo(id));
  }

  decode(ids: number[]): string {
    const chunks: number[] = [];
    for (const id of ids) chunks.push(...this.tokenBytes(id));
    return utf8dec.decode(new Uint8Array(chunks));
  }

  /** Für die Erklärung: hat der Tokenizer ein Präfix-Leerzeichen? */
  get prefixSpace(): boolean {
    return this.addPrefixSpace;
  }
}
