/// <reference lib="webworker" />
/**
 * Worker für das echte Modell. Hier laufen Download, Aufbau der Gewichte und jeder Forward-Pass,
 * damit die Seite nie blockiert. Höchstens EIN echtes Modell ist geladen: Beim Wechsel werden
 * Gewichte und KV-Caches des alten verworfen (die Shards bleiben im Cache Storage).
 *
 * „Neueste Anfrage gewinnt“: Jede Anfrage trägt eine fortlaufende Nummer pro Sitzung. Bevor
 * gerechnet wird, gibt der Worker der Warteschlange einmal Gelegenheit, neuere Anfragen
 * anzunehmen. Ist inzwischen eine neuere für dieselbe Sitzung da, wird die ältere übersprungen.
 */
import * as Comlink from 'comlink';
import { LlamaModel } from '../model/llama.ts';
import { configFromHf } from '../model/weights.ts';
import { KvSessions } from '../model/sessions.ts';
import { unpackWeights, type ModelManifest } from '../model/pack.ts';
import { cachedBytes, fetchManifest, fetchShards, ModelLoadError, type LoadProgress } from '../model/shards.ts';
import { KV_SESSIONS, REAL_MAX_SEQ } from '../limits.ts';

interface Loaded {
  key: string;
  manifest: ModelManifest;
  model: LlamaModel;
  sessions: KvSessions;
  loadMs: number;
  fromCache: number;
}

let current: Loaded | null = null;
let loading: AbortController | null = null;
const latest = new Map<string, number>();

/** Einmal die Ereignisschleife drehen, damit wartende Nachrichten angenommen werden. */
const yieldToQueue = () => new Promise<void>((r) => setTimeout(r, 0));

export interface WorkerInfo {
  key: string | null;
  model?: string;
  edition?: string;
  revision?: string;
  fake?: boolean;
  weightBytes: number;
  kvBytes: number;
  sessions: number;
  loadMs?: number;
  fromCache?: number;
}

export interface LogitsReply {
  req: number;
  stale?: boolean;
  logits?: Float32Array;
  ms?: number;
  computed?: number;
  reused?: number;
}

const api = {
  /** Wie viele Bytes dieses Modells liegen schon im Cache Storage? (für „ohne Download“) */
  async cached(manifestUrl: string): Promise<{ cached: number; total: number } | null> {
    try {
      const m = await fetchManifest(manifestUrl);
      return { cached: await cachedBytes(m, manifestUrl), total: m.totalBytes };
    } catch {
      return null;
    }
  },

  async load(key: string, manifestUrl: string, onProgress: (p: LoadProgress) => void): Promise<WorkerInfo> {
    loading?.abort();
    const ctrl = new AbortController();
    loading = ctrl;
    // höchstens ein Modell: das alte zuerst freigeben
    current = null;
    latest.clear();
    const t0 = performance.now();
    try {
      onProgress({ loaded: 0, total: 0, fromCache: 0, phase: 'manifest' });
      const manifest = await fetchManifest(manifestUrl, ctrl.signal);
      let fromCache = 0;
      const buf = await fetchShards(
        manifest,
        manifestUrl,
        (p) => {
          fromCache = p.fromCache;
          onProgress(p);
        },
        ctrl.signal,
      );
      if (ctrl.signal.aborted) throw new ModelLoadError('Abgebrochen', 'aborted');
      const cfg = configFromHf(manifest.config);
      const weights = unpackWeights(manifest.tensors, buf, cfg.layers);
      const model = new LlamaModel(cfg, weights, REAL_MAX_SEQ);
      current = { key, manifest, model, sessions: new KvSessions(model, KV_SESSIONS), loadMs: performance.now() - t0, fromCache };
      return api.info();
    } finally {
      if (loading === ctrl) loading = null;
    }
  },

  cancel() {
    loading?.abort();
  },

  unload() {
    current = null;
    latest.clear();
  },

  info(): WorkerInfo {
    if (!current) return { key: null, weightBytes: 0, kvBytes: 0, sessions: 0 };
    const m = current.manifest;
    return {
      key: current.key,
      model: m.model,
      edition: m.edition,
      revision: m.revision,
      fake: m.fake,
      weightBytes: m.weightBytes,
      kvBytes: current.sessions.bytes,
      sessions: current.sessions.size,
      loadMs: current.loadMs,
      fromCache: current.fromCache,
    };
  },

  async logits(key: string, session: string, req: number, ids: number[]): Promise<LogitsReply> {
    latest.set(session, Math.max(latest.get(session) ?? 0, req));
    await yieldToQueue();
    if (req < (latest.get(session) ?? 0)) return { req, stale: true };
    if (!current || current.key !== key) throw new Error('Modell nicht geladen');
    const t0 = performance.now();
    const logits = current.sessions.logits(session, ids);
    const { computed, reused } = current.sessions.last;
    return Comlink.transfer({ req, logits, ms: performance.now() - t0, computed, reused }, [logits.buffer]);
  },

  /** Benchmark für #/diagnose: Prefill Token für Token vs. Batch auf dem geladenen Modell */
  bench(n = 32): { perToken: number; batch: number } | null {
    if (!current) return null;
    const ids = Array.from({ length: n }, (_, i) => 100 + i);
    const c = current.model.newCache(n + 1);
    current.model.useCache(c);
    let t = performance.now();
    current.model.prefill(ids);
    const perToken = performance.now() - t;
    current.model.reset();
    t = performance.now();
    current.model.prefillBatch(ids);
    const batch = performance.now() - t;
    return { perToken, batch };
  },
};

export type ModelWorkerApi = typeof api;
Comlink.expose(api);
