/**
 * Einheitliche Schnittstelle für Sprachmodelle in den Widgets.
 * P1b: Spielzeugmodell (Wort-Tokenizer + winziges Llama). P2: echtes SmolLM2 über dieselbe Schnittstelle.
 */
import type { StampKind } from '../components/primitives';
import { LlamaModel, configFromHf, loadLlamaWeights, type LlamaConfig } from './model/llama.ts';
import { parseSafeTensors } from './model/safetensors.ts';
import { WordTokenizer } from './tokenizer/word.ts';

export interface TokenView {
  id: number;
  text: string;
  unknown?: boolean;
  special?: boolean;
}

export interface LanguageModel {
  id: string;
  label: string;
  stamp: StampKind;
  vocabSize: number;
  bosId: number | null;
  eosIds: number[];
  /** Prompt-Text → Tokens (ohne BOS) */
  tokenize(text: string): TokenView[];
  token(id: number): TokenView;
  decode(ids: number[]): string;
  /** Logits für das nächste Token nach der kompletten Folge (inkl. BOS, falls verwendet); Kopie */
  logitsFor(ids: number[]): Float32Array;
}

export class ToyLanguageModel implements LanguageModel {
  readonly id: string;
  readonly label: string;
  readonly stamp: StampKind = 'toy';
  readonly tokenizer: WordTokenizer;
  readonly cfg: LlamaConfig;
  readonly config: Record<string, unknown>;
  private model: LlamaModel;
  private cache = new Map<string, Float32Array>();

  constructor(id: string, label: string, config: Record<string, unknown>, vocab: string[], weights: ArrayBuffer) {
    this.id = id;
    this.label = label;
    this.config = config;
    this.cfg = configFromHf(config);
    this.tokenizer = new WordTokenizer(vocab);
    this.model = new LlamaModel(this.cfg, loadLlamaWeights(parseSafeTensors(weights), this.cfg, 'f32'), this.cfg.maxPos);
  }

  get vocabSize() {
    return this.cfg.vocab;
  }
  get bosId() {
    return this.tokenizer.bosId;
  }
  get eosIds() {
    return [this.tokenizer.eosId];
  }
  get maxContext() {
    return this.cfg.maxPos;
  }

  tokenize(text: string): TokenView[] {
    return this.tokenizer.encode(text);
  }
  token(id: number): TokenView {
    return this.tokenizer.token(id);
  }
  decode(ids: number[]): string {
    return this.tokenizer.decode(ids);
  }

  logitsFor(ids: number[]): Float32Array {
    const key = ids.join(',');
    const hit = this.cache.get(key);
    if (hit) return hit;
    if (ids.length > this.cfg.maxPos) throw new Error(`Kontext zu lang (max. ${this.cfg.maxPos} Tokens)`);
    this.model.reset();
    const out = Float32Array.from(this.model.prefill(ids));
    if (this.cache.size > 2000) this.cache.clear();
    this.cache.set(key, out);
    return out;
  }

  /** Prompt-Text → IDs mit vorangestelltem BOS (wie im Training) */
  promptIds(text: string): number[] {
    return [this.tokenizer.bosId, ...this.tokenize(text).map((t) => t.id)];
  }
}

/* ── Laden im Browser ─────────────────────────────────────────────── */

export type ToyVariant = 'toy' | 'toy-nopos';
const loaded = new Map<ToyVariant, Promise<ToyLanguageModel>>();

export function toyAssetUrls(variant: ToyVariant = 'toy') {
  const base = `${import.meta.env.BASE_URL}data/${variant}/`;
  return { config: base + 'config.json', vocab: base + 'vocab.json', weights: base + 'model.safetensors' };
}

export function loadToyModel(variant: ToyVariant = 'toy'): Promise<ToyLanguageModel> {
  let p = loaded.get(variant);
  if (!p) {
    const urls = toyAssetUrls(variant);
    p = Promise.all([
      fetch(urls.config).then((r) => r.json()),
      fetch(urls.vocab).then((r) => r.json()),
      fetch(urls.weights).then((r) => {
        if (!r.ok) throw new Error(`Toy-Modell nicht gefunden (${r.status})`);
        return r.arrayBuffer();
      }),
    ]).then(
      ([config, vocab, weights]) =>
        new ToyLanguageModel(variant, variant === 'toy' ? 'Spielzeugmodell' : 'Spielzeugmodell ohne Position', config, vocab, weights),
    );
    p.catch(() => loaded.delete(variant));
    loaded.set(variant, p);
  }
  return p;
}
