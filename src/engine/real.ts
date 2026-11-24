/**
 * Echtes Modell (SmolLM2) im Browser: Verbindung zum Worker und Anbindung an `LanguageModel`.
 * Die Tokenisierung läuft im Hauptthread (der Tokenizer ist ohnehin geladen), gerechnet wird im Worker.
 */
import * as Comlink from 'comlink';
import editionsJson from '../generated/model-editions.json';
import type { StampKind } from '../components/primitives';
import { StaleRequest, type LanguageModel, type LogitsOptions, type TokenView } from './lm.ts';
import { REAL_MAX_SEQ, REAL_PROMPT_LIMIT, kvBytes } from './limits.ts';
import type { BpeTokenizer } from './tokenizer/bpe.ts';
import type { LoadProgress } from './model/shards.ts';
import type { ModelWorkerApi, WorkerInfo } from './worker/model.worker.ts';

export interface EditionInfo {
  model: string;
  edition: 'int8' | 'fp16';
  repo: string;
  revision: string;
  version: string;
  spec: string;
  downloadBytes: number;
  weightBytes: number;
  layers: number;
  kvDim: number;
  eos: number | number[] | null;
  fake: boolean;
}

/** Fassungen, die dieser Build ausliefert (leer, wenn ohne Modell-Shards gebaut) */
export const EDITIONS = editionsJson as EditionInfo[];

export type EditionKey = `${string}/${'int8' | 'fp16'}`;
export const editionKey = (e: Pick<EditionInfo, 'model' | 'edition'>) => `${e.model}/${e.edition}` as EditionKey;
export const findEdition = (key: string) => EDITIONS.find((e) => editionKey(e) === key);

export const BASE_MODEL = 'smollm2-135m';
export const INSTRUCT_MODEL = 'smollm2-135m-instruct';

/** Arbeitsspeicher nach dem Laden: Gewichte + KV-Caches aller Sitzungen */
export const ramBytes = (e: EditionInfo) => e.weightBytes + kvBytes(e.layers, e.kvDim);

export function editionLabel(e: Pick<EditionInfo, 'model' | 'edition'>, short = false) {
  const kind = e.model === INSTRUCT_MODEL ? 'Instruct' : 'Base';
  if (short) return `${kind} · ${e.edition}`;
  return `SmolLM2-135M (${kind}, ${e.edition === 'fp16' ? 'fp16, exakt' : 'int8'})`;
}

export const manifestUrl = (key: string) => new URL(`${import.meta.env.BASE_URL}models/${key}/manifest.json`, location.href).href;

/* ── Worker ───────────────────────────────────────────────────────── */

let client: Comlink.Remote<ModelWorkerApi> | null = null;

export function workerClient(): Comlink.Remote<ModelWorkerApi> {
  if (!client) {
    const w = new Worker(new URL('./worker/model.worker.ts', import.meta.url), { type: 'module', name: 'igai-model' });
    client = Comlink.wrap<ModelWorkerApi>(w);
  }
  return client;
}

export async function workerInfo(): Promise<WorkerInfo | null> {
  return client ? client.info() : null;
}

/* ── Modell ───────────────────────────────────────────────────────── */

let requestCounter = 0;

export class RealLanguageModel implements LanguageModel {
  readonly kind = 'real' as const;
  readonly id: string;
  readonly label: string;
  readonly stamp: StampKind;
  readonly bosId = null;
  readonly eosIds: number[];
  readonly maxContext = REAL_MAX_SEQ;
  readonly promptLimit = REAL_PROMPT_LIMIT;
  readonly isChat: boolean;
  readonly edition: EditionInfo;
  readonly tokenizer: BpeTokenizer;

  constructor(edition: EditionInfo, tokenizer: BpeTokenizer) {
    this.edition = edition;
    this.tokenizer = tokenizer;
    this.id = editionKey(edition);
    this.label = editionLabel(edition);
    this.stamp = edition.edition === 'fp16' ? 'real' : 'real-q8';
    this.isChat = edition.model === INSTRUCT_MODEL;
    // Stopp-Tokens: aus generation_config.json plus die Endmarken des Chat-Formats
    const eos = new Set<number>(Array.isArray(edition.eos) ? edition.eos : edition.eos !== null ? [edition.eos] : []);
    for (const t of ['<|im_end|>', '<|endoftext|>']) {
      const id = tokenizer.vocab.get(t);
      if (id !== undefined) eos.add(id);
    }
    this.eosIds = [...eos];
  }

  get vocabSize() {
    return this.tokenizer.vocabSize;
  }

  tokenize(text: string): TokenView[] {
    return this.tokenizer.encode(text);
  }
  token(id: number): TokenView {
    return this.tokenizer.tokenInfo(id);
  }
  decode(ids: number[]): string {
    return this.tokenizer.decode(ids);
  }
  promptIds(text: string): number[] {
    return this.tokenizer.encodeIds(text);
  }

  async logitsFor(ids: number[], opts: LogitsOptions = {}): Promise<Float32Array> {
    const req = ++requestCounter;
    const r = await workerClient().logits(this.id, opts.session ?? 'default', req, ids);
    if (r.stale || !r.logits) throw new StaleRequest();
    return r.logits;
  }
}

export type { LoadProgress, WorkerInfo };
