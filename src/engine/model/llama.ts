import { applyRope, linear, linearBatch, linearRow, rmsNorm, ropeTables, silu, softmaxInPlace } from '../core/ops.ts';
import type { LlamaConfig, LlamaWeights } from './weights.ts';

export {
  configFromHf,
  downloadBytes,
  loadLlamaWeights,
  parseFormat,
  weightBytes,
  type Fmt,
  type LlamaConfig,
  type LlamaWeights,
  type Precision,
  type WeightFormat,
} from './weights.ts';

/**
 * Decoder-Transformer im Llama-Stil (SmolLM2, Llama, Mistral …):
 * RMSNorm · RoPE · Grouped-Query-Attention · SwiGLU-MLP · optional gebundene Embeddings.
 *
 * Der KV-Cache ist ein eigenes Objekt (`KvCache`). Ein Modell kann mehrere davon haben
 * (eine Sitzung pro Widget) und wechselt mit `useCache` zwischen ihnen, ohne Gewichte zu kopieren.
 */

/** Keys und Values aller Layer für eine Tokenfolge; `length` = Anzahl gespeicherter Positionen. */
export class KvCache {
  readonly k: Float32Array[];
  readonly v: Float32Array[];
  readonly maxSeq: number;
  length = 0;
  constructor(cfg: LlamaConfig, maxSeq: number) {
    const kvDim = cfg.kvHeads * cfg.headDim;
    this.maxSeq = maxSeq;
    this.k = Array.from({ length: cfg.layers }, () => new Float32Array(maxSeq * kvDim));
    this.v = Array.from({ length: cfg.layers }, () => new Float32Array(maxSeq * kvDim));
  }
  get bytes() {
    return this.k.reduce((s, a) => s + a.byteLength, 0) * 2;
  }
}

/** Batches beim Prefill höchstens so groß (begrenzt den Arbeitsspeicher pro Aufruf). */
const PREFILL_CHUNK = 64;

export class LlamaModel {
  readonly cfg: LlamaConfig;
  readonly w: LlamaWeights;
  readonly maxSeq: number;
  private kv: KvCache;
  private rope: ReturnType<typeof ropeTables>;
  // Arbeitsspeicher für Einzelschritte, einmal angelegt
  private x: Float32Array;
  private xn: Float32Array;
  private q: Float32Array;
  private k: Float32Array;
  private v: Float32Array;
  private att: Float32Array;
  private attOut: Float32Array;
  private tmp: Float32Array;
  private g: Float32Array;
  private u: Float32Array;
  readonly logits: Float32Array;

  constructor(cfg: LlamaConfig, w: LlamaWeights, maxSeq = 512) {
    this.cfg = cfg;
    this.w = w;
    this.maxSeq = maxSeq;
    const kvDim = cfg.kvHeads * cfg.headDim;
    this.kv = new KvCache(cfg, maxSeq);
    this.rope = ropeTables(cfg.headDim, maxSeq, cfg.ropeTheta);
    this.x = new Float32Array(cfg.dModel);
    this.xn = new Float32Array(cfg.dModel);
    this.q = new Float32Array(cfg.heads * cfg.headDim);
    this.k = new Float32Array(kvDim);
    this.v = new Float32Array(kvDim);
    this.att = new Float32Array(maxSeq);
    this.attOut = new Float32Array(cfg.heads * cfg.headDim);
    this.tmp = new Float32Array(cfg.dModel);
    this.g = new Float32Array(cfg.dFF);
    this.u = new Float32Array(cfg.dFF);
    this.logits = new Float32Array(cfg.vocab);
  }

  get position() {
    return this.kv.length;
  }

  /** Neuer, leerer KV-Cache passend zu diesem Modell. */
  newCache(maxSeq = this.maxSeq): KvCache {
    return new KvCache(this.cfg, Math.min(maxSeq, this.maxSeq));
  }

  /** Arbeitet ab jetzt mit diesem KV-Cache (z. B. einer anderen Sitzung). */
  useCache(c: KvCache) {
    this.kv = c;
  }

  get cache(): KvCache {
    return this.kv;
  }

  reset() {
    this.kv.length = 0;
  }

  /** Verwirft alle Positionen ab n; die ersten n bleiben gültig (gemeinsamer Anfang). */
  truncate(n: number) {
    this.kv.length = Math.max(0, Math.min(this.kv.length, n));
  }

  /** Verarbeitet ein Token an der nächsten Position; berechnet Logits nur auf Wunsch. */
  step(token: number, computeLogits = true): Float32Array | null {
    const { cfg, w } = this;
    const { heads, kvHeads, headDim, eps } = cfg;
    const pos = this.kv.length;
    if (pos >= this.kv.maxSeq) throw new Error(`Kontext voll (${this.kv.maxSeq} Tokens)`);
    const kvDim = kvHeads * headDim;
    const group = heads / kvHeads;
    const scale = 1 / Math.sqrt(headDim);
    const x = this.x;

    linearRow(w.embed, token, x);

    for (let li = 0; li < w.layers.length; li++) {
      const L = w.layers[li];
      // ── Attention ──
      rmsNorm(x, L.inNorm, eps, this.xn);
      linear(L.q, this.xn, this.q);
      linear(L.k, this.xn, this.k);
      linear(L.v, this.xn, this.v);
      if (cfg.rope) {
        for (let h = 0; h < heads; h++) applyRope(this.q, h * headDim, pos, this.rope);
        for (let h = 0; h < kvHeads; h++) applyRope(this.k, h * headDim, pos, this.rope);
      }
      const kc = this.kv.k[li];
      const vc = this.kv.v[li];
      kc.set(this.k, pos * kvDim);
      vc.set(this.v, pos * kvDim);

      for (let h = 0; h < heads; h++) {
        const kvh = Math.floor(h / group);
        const qo = h * headDim;
        for (let t = 0; t <= pos; t++) {
          const ko = t * kvDim + kvh * headDim;
          let dot = 0;
          for (let i = 0; i < headDim; i++) dot += this.q[qo + i] * kc[ko + i];
          this.att[t] = dot * scale;
        }
        softmaxInPlace(this.att, pos + 1);
        for (let i = 0; i < headDim; i++) this.attOut[qo + i] = 0;
        for (let t = 0; t <= pos; t++) {
          const a = this.att[t];
          const vo = t * kvDim + kvh * headDim;
          for (let i = 0; i < headDim; i++) this.attOut[qo + i] += a * vc[vo + i];
        }
      }
      linear(L.o, this.attOut, this.tmp);
      for (let i = 0; i < x.length; i++) x[i] += this.tmp[i];

      // ── MLP (SwiGLU) ──
      rmsNorm(x, L.postNorm, eps, this.xn);
      linear(L.gate, this.xn, this.g);
      linear(L.up, this.xn, this.u);
      for (let i = 0; i < this.g.length; i++) this.g[i] = silu(this.g[i]) * this.u[i];
      linear(L.down, this.g, this.tmp);
      for (let i = 0; i < x.length; i++) x[i] += this.tmp[i];
    }

    this.kv.length = pos + 1;
    if (!computeLogits) return null;
    rmsNorm(x, w.finalNorm, eps, this.xn);
    linear(w.lmHead, this.xn, this.logits);
    return this.logits;
  }

  /** Prompt Token für Token einlesen (Referenzweg); liefert die Logits nach dem letzten Token. */
  prefill(tokens: number[]): Float32Array {
    let out: Float32Array | null = null;
    for (let i = 0; i < tokens.length; i++) out = this.step(tokens[i], i === tokens.length - 1);
    return out!;
  }

  /**
   * Batch-Prefill: alle Tokens einer Folge gemeinsam durch jede Schicht (Matrix × Matrix).
   * Jede Gewichtszeile wird pro Schicht nur einmal gelesen statt einmal pro Token.
   * Setzt am aktuellen Ende des KV-Caches an (Verlängerung nach `truncate`).
   * Ergebnis ist bitgleich mit `prefill`, nur schneller.
   */
  prefillBatch(tokens: number[]): Float32Array {
    if (!tokens.length) throw new Error('prefillBatch: keine Tokens');
    for (let s = 0; s < tokens.length; s += PREFILL_CHUNK) this.batchChunk(tokens.slice(s, s + PREFILL_CHUNK));
    return this.logits;
  }

  private batchChunk(tokens: number[]) {
    const { cfg, w } = this;
    const { heads, kvHeads, headDim, eps, dModel: d, dFF } = cfg;
    const n = tokens.length;
    const start = this.kv.length;
    if (start + n > this.kv.maxSeq) throw new Error(`Kontext voll (${this.kv.maxSeq} Tokens)`);
    const kvDim = kvHeads * headDim;
    const qDim = heads * headDim;
    const group = heads / kvHeads;
    const scale = 1 / Math.sqrt(headDim);

    const X = new Float32Array(n * d);
    const XN = new Float32Array(n * d);
    const Q = new Float32Array(n * qDim);
    const K = new Float32Array(n * kvDim);
    const V = new Float32Array(n * kvDim);
    const A = new Float32Array(n * qDim);
    const T = new Float32Array(n * d);
    const G = new Float32Array(n * dFF);
    const U = new Float32Array(n * dFF);
    const att = this.att;
    const row = (buf: Float32Array, t: number, len: number) => buf.subarray(t * len, (t + 1) * len);

    for (let t = 0; t < n; t++) linearRow(w.embed, tokens[t], row(X, t, d));

    for (let li = 0; li < w.layers.length; li++) {
      const L = w.layers[li];
      for (let t = 0; t < n; t++) rmsNorm(row(X, t, d), L.inNorm, eps, row(XN, t, d));
      linearBatch(L.q, XN, n, Q);
      linearBatch(L.k, XN, n, K);
      linearBatch(L.v, XN, n, V);
      const kc = this.kv.k[li];
      const vc = this.kv.v[li];
      for (let t = 0; t < n; t++) {
        const pos = start + t;
        if (cfg.rope) {
          for (let h = 0; h < heads; h++) applyRope(Q, t * qDim + h * headDim, pos, this.rope);
          for (let h = 0; h < kvHeads; h++) applyRope(K, t * kvDim + h * headDim, pos, this.rope);
        }
        kc.set(row(K, t, kvDim), pos * kvDim);
        vc.set(row(V, t, kvDim), pos * kvDim);
      }
      // kausale Attention: Token t sieht alle Positionen bis einschließlich start + t
      for (let t = 0; t < n; t++) {
        const pos = start + t;
        for (let h = 0; h < heads; h++) {
          const kvh = Math.floor(h / group);
          const qo = t * qDim + h * headDim;
          for (let s = 0; s <= pos; s++) {
            const ko = s * kvDim + kvh * headDim;
            let dot = 0;
            for (let i = 0; i < headDim; i++) dot += Q[qo + i] * kc[ko + i];
            att[s] = dot * scale;
          }
          softmaxInPlace(att, pos + 1);
          for (let i = 0; i < headDim; i++) A[qo + i] = 0;
          for (let s = 0; s <= pos; s++) {
            const a = att[s];
            const vo = s * kvDim + kvh * headDim;
            for (let i = 0; i < headDim; i++) A[qo + i] += a * vc[vo + i];
          }
        }
      }
      linearBatch(L.o, A, n, T);
      for (let i = 0; i < X.length; i++) X[i] += T[i];

      for (let t = 0; t < n; t++) rmsNorm(row(X, t, d), L.postNorm, eps, row(XN, t, d));
      linearBatch(L.gate, XN, n, G);
      linearBatch(L.up, XN, n, U);
      for (let i = 0; i < G.length; i++) G[i] = silu(G[i]) * U[i];
      linearBatch(L.down, G, n, T);
      for (let i = 0; i < X.length; i++) X[i] += T[i];
    }

    this.kv.length = start + n;
    // Logits nur für das letzte Token
    rmsNorm(row(X, n - 1, d), w.finalNorm, eps, this.xn);
    linear(w.lmHead, this.xn, this.logits);
  }
}

export function argmax(x: Float32Array): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}
