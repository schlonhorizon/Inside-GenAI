/**
 * Einheitliche Schnittstelle für Sprachmodelle in den Widgets: Spielzeugmodell (im Hauptthread)
 * und echtes SmolLM2 (im Worker, src/engine/real.ts). Logits kommen immer asynchron.
 */
import type { StampKind } from '../components/primitives';
import { LlamaModel, configFromHf, loadLlamaWeights, type LlamaConfig } from './model/llama.ts';
import { parseSafeTensors } from './model/safetensors.ts';
import { KvSessions } from './model/sessions.ts';
import { WordTokenizer } from './tokenizer/word.ts';

export interface TokenView {
  id: number;
  text: string;
  unknown?: boolean;
  special?: boolean;
  partial?: boolean;
}

export interface LogitsOptions {
  /** eine Sitzung pro Widget: eigener KV-Cache, „neueste Anfrage gewinnt“ */
  session?: string;
}

/** Ergebnis einer überholten Anfrage: wird still verworfen. */
export class StaleRequest extends Error {
  constructor() {
    super('überholt');
    this.name = 'StaleRequest';
  }
}

export interface LanguageModel {
  id: string;
  label: string;
  stamp: StampKind;
  kind: 'toy' | 'real';
  vocabSize: number;
  bosId: number | null;
  /** Tokens, bei denen die Generierung stoppt */
  eosIds: number[];
  /** Höchstzahl Tokens im Kontext (Prompt + Erzeugtes) */
  maxContext: number;
  /** Höchstzahl Tokens im Prompt */
  promptLimit: number;
  /** Prompt-Text → Tokens (ohne BOS) */
  tokenize(text: string): TokenView[];
  token(id: number): TokenView;
  decode(ids: number[]): string;
  /** Prompt-Text → IDs, wie das Modell sie im Training gesehen hat (ggf. mit BOS) */
  promptIds(text: string): number[];
  /** Logits für das nächste Token nach der kompletten Folge; Kopie */
  logitsFor(ids: number[], opts?: LogitsOptions): Promise<Float32Array>;
}

export class ToyLanguageModel implements LanguageModel {
  readonly id: string;
  readonly label: string;
  readonly stamp: StampKind = 'toy';
  readonly kind = 'toy' as const;
  readonly tokenizer: WordTokenizer;
  readonly cfg: LlamaConfig;
  readonly config: Record<string, unknown>;
  private sessions: KvSessions;

  constructor(id: string, label: string, config: Record<string, unknown>, vocab: string[], weights: ArrayBuffer) {
    this.id = id;
    this.label = label;
    this.config = config;
    this.cfg = configFromHf(config);
    this.tokenizer = new WordTokenizer(vocab);
    const model = new LlamaModel(this.cfg, loadLlamaWeights(parseSafeTensors(weights), this.cfg, 'f32'), this.cfg.maxPos);
    this.sessions = new KvSessions(model, 16);
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
  get promptLimit() {
    return this.cfg.maxPos - 1;
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

  /** Synchron (das Spielzeugmodell rechnet in Mikrosekunden); für Tests und Auswertungen. */
  logitsSync(ids: number[], session = 'sync'): Float32Array {
    if (ids.length > this.cfg.maxPos) throw new Error(`Kontext zu lang (max. ${this.cfg.maxPos} Tokens)`);
    return this.sessions.logits(session, ids);
  }

  async logitsFor(ids: number[], opts: LogitsOptions = {}): Promise<Float32Array> {
    return this.logitsSync(ids, opts.session ?? 'default');
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
