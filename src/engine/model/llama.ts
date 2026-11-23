import { applyRope, linear, linearRow, rmsNorm, ropeTables, silu, softmaxInPlace, type Linear } from '../core/ops.ts';
import { linearBytes, linearF32, quantizeRows } from './quantize.ts';
import type { SafeTensors } from './safetensors.ts';

/**
 * Decoder-Transformer im Llama-Stil (SmolLM2, Llama, Mistral …):
 * RMSNorm · RoPE · Grouped-Query-Attention · SwiGLU-MLP · optional gebundene Embeddings.
 * Ein Token pro Aufruf mit KV-Cache – einfach nachvollziehbar, Hooks folgen in P2.
 */

export interface LlamaConfig {
  vocab: number;
  dModel: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  dFF: number;
  eps: number;
  ropeTheta: number;
  maxPos: number;
  tied: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function configFromHf(c: any): LlamaConfig {
  return {
    vocab: c.vocab_size,
    dModel: c.hidden_size,
    layers: c.num_hidden_layers,
    heads: c.num_attention_heads,
    kvHeads: c.num_key_value_heads ?? c.num_attention_heads,
    headDim: c.head_dim ?? c.hidden_size / c.num_attention_heads,
    dFF: c.intermediate_size,
    eps: c.rms_norm_eps ?? 1e-5,
    ropeTheta: c.rope_theta ?? 10000,
    maxPos: c.max_position_embeddings ?? 2048,
    tied: c.tie_word_embeddings ?? false,
  };
}

interface Layer {
  inNorm: Float32Array;
  q: Linear;
  k: Linear;
  v: Linear;
  o: Linear;
  postNorm: Float32Array;
  gate: Linear;
  up: Linear;
  down: Linear;
}

export interface LlamaWeights {
  embed: Linear;
  lmHead: Linear;
  layers: Layer[];
  finalNorm: Float32Array;
}

export type Precision = 'f32' | 'q8';

/** Lädt Gewichte aus einer HF-safetensors-Datei; optional als int8 pro Zeile. */
export function loadLlamaWeights(st: SafeTensors, cfg: LlamaConfig, precision: Precision): LlamaWeights {
  const lin = (name: string, rows: number, cols: number): Linear => {
    const w = st.get(name).data;
    return precision === 'q8' ? quantizeRows(w, rows, cols) : linearF32(w, rows, cols);
  };
  const vec = (name: string) => st.get(name).data;
  const { dModel: d, heads, kvHeads, headDim, dFF, vocab } = cfg;
  const embed = lin('model.embed_tokens.weight', vocab, d);
  const layers: Layer[] = [];
  for (let i = 0; i < cfg.layers; i++) {
    const p = `model.layers.${i}.`;
    layers.push({
      inNorm: vec(p + 'input_layernorm.weight'),
      q: lin(p + 'self_attn.q_proj.weight', heads * headDim, d),
      k: lin(p + 'self_attn.k_proj.weight', kvHeads * headDim, d),
      v: lin(p + 'self_attn.v_proj.weight', kvHeads * headDim, d),
      o: lin(p + 'self_attn.o_proj.weight', d, heads * headDim),
      postNorm: vec(p + 'post_attention_layernorm.weight'),
      gate: lin(p + 'mlp.gate_proj.weight', dFF, d),
      up: lin(p + 'mlp.up_proj.weight', dFF, d),
      down: lin(p + 'mlp.down_proj.weight', d, dFF),
    });
  }
  const lmHead = cfg.tied || !st.header['lm_head.weight'] ? embed : lin('lm_head.weight', vocab, d);
  return { embed, lmHead, layers, finalNorm: vec('model.norm.weight') };
}

export function weightBytes(w: LlamaWeights): number {
  let n = linearBytes(w.embed) + (w.lmHead === w.embed ? 0 : linearBytes(w.lmHead)) + w.finalNorm.byteLength;
  for (const l of w.layers) {
    n += l.inNorm.byteLength + l.postNorm.byteLength;
    for (const x of [l.q, l.k, l.v, l.o, l.gate, l.up, l.down]) n += linearBytes(x);
  }
  return n;
}

export class LlamaModel {
  readonly cfg: LlamaConfig;
  readonly w: LlamaWeights;
  readonly maxSeq: number;
  private kCache: Float32Array[];
  private vCache: Float32Array[];
  private rope: ReturnType<typeof ropeTables>;
  private pos = 0;
  // Arbeitsspeicher, einmal angelegt
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
    this.kCache = w.layers.map(() => new Float32Array(maxSeq * kvDim));
    this.vCache = w.layers.map(() => new Float32Array(maxSeq * kvDim));
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
    return this.pos;
  }

  reset() {
    this.pos = 0;
  }

  /** Verarbeitet ein Token an der nächsten Position; berechnet Logits nur auf Wunsch. */
  step(token: number, computeLogits = true): Float32Array | null {
    const { cfg, w } = this;
    const { heads, kvHeads, headDim, eps } = cfg;
    const pos = this.pos;
    if (pos >= this.maxSeq) throw new Error(`Kontext voll (${this.maxSeq} Tokens)`);
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
      for (let h = 0; h < heads; h++) applyRope(this.q, h * headDim, pos, this.rope);
      for (let h = 0; h < kvHeads; h++) applyRope(this.k, h * headDim, pos, this.rope);
      const kc = this.kCache[li];
      const vc = this.vCache[li];
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

    this.pos++;
    if (!computeLogits) return null;
    rmsNorm(x, w.finalNorm, eps, this.xn);
    linear(w.lmHead, this.xn, this.logits);
    return this.logits;
  }

  /** Prompt einlesen; liefert die Logits nach dem letzten Token. */
  prefill(tokens: number[]): Float32Array {
    let out: Float32Array | null = null;
    for (let i = 0; i < tokens.length; i++) out = this.step(tokens[i], i === tokens.length - 1);
    return out!;
  }
}

export function argmax(x: Float32Array): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}
