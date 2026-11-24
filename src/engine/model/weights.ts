import type { Linear } from '../core/ops.ts';
import { linearBytes, linearDownloadBytes, linearF32, quantizeGroups, quantizeRows, toF16Precision } from './quantize.ts';
import type { SafeTensors } from './safetensors.ts';

/**
 * Gewichte eines Llama-Modells: Aufbau, Formate je Modellteil, Laden aus safetensors.
 * Eigene Datei, damit der CI-Cache der Modell-Shards nur von der Gewichtsaufbereitung
 * abhängt und nicht von jeder Änderung am Forward-Pass (llama.ts).
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
  /** false = kein Positionssignal (Toy-Variante für Kapitel 4) */
  rope: boolean;
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
    rope: c.rope !== false,
  };
}

export interface Layer {
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

/**
 * Gewichtsformat je Teil des Modells:
 *   f32      volle Genauigkeit
 *   f16      fp16-Genauigkeit (so würde es ausgeliefert), im Speicher als f32
 *   q8       int8, ein Skalierungsfaktor pro Zeile
 *   q8g<N>   int8, ein Skalierungsfaktor je N Werte einer Zeile (z. B. q8g64)
 *
 * Teile: attn (q/k/v/o), mlp (gate/up), down (MLP-down), emb (Embedding = LM-Head).
 * Als String: "q8g64" (alle Teile) oder "attn:q8,mlp:q8,down:f16,emb:f16";
 * Zusatz "+s16" rundet die int8-Skalen auf fp16 (wie bei der Auslieferung).
 */
export type Fmt = 'f32' | 'f16' | 'q8' | `q8g${number}`;
export interface WeightFormat {
  attn: Fmt;
  mlp: Fmt;
  down: Fmt;
  emb: Fmt;
  scales16: boolean;
}
export type Precision = string;

const PARTS = ['attn', 'mlp', 'down', 'emb'] as const;

export function parseFormat(spec: string | WeightFormat): WeightFormat {
  if (typeof spec !== 'string') return spec;
  const [body, ...flags] = spec.split('+');
  const scales16 = flags.includes('s16');
  const check = (f: string): Fmt => {
    if (f === 'f32' || f === 'f16' || f === 'q8' || /^q8g\d+$/.test(f)) return f as Fmt;
    throw new Error(`Unbekanntes Format: ${f}`);
  };
  if (!body.includes(':')) {
    const f = check(body);
    return { attn: f, mlp: f, down: f, emb: f, scales16 };
  }
  const out: WeightFormat = { attn: 'f32', mlp: 'f32', down: 'f32', emb: 'f32', scales16 };
  for (const part of body.split(',')) {
    const [k, v] = part.split(':');
    if (!(PARTS as readonly string[]).includes(k)) throw new Error(`Unbekannter Teil: ${k}`);
    out[k as (typeof PARTS)[number]] = check(v);
  }
  return out;
}

function makeLinear(w: Float32Array, rows: number, cols: number, fmt: Fmt, scales16: boolean): Linear {
  if (fmt === 'f32') return linearF32(w, rows, cols);
  if (fmt === 'f16') return linearF32(toF16Precision(w), rows, cols);
  if (fmt === 'q8') return quantizeRows(w, rows, cols, scales16);
  return quantizeGroups(w, rows, cols, Number(fmt.slice(3)), scales16);
}

/** Lädt Gewichte aus einer HF-safetensors-Datei im gewünschten Format. */
export function loadLlamaWeights(st: SafeTensors, cfg: LlamaConfig, precision: Precision | WeightFormat): LlamaWeights {
  const f = parseFormat(precision);
  const lin = (name: string, rows: number, cols: number, part: (typeof PARTS)[number]): Linear =>
    makeLinear(st.get(name).data, rows, cols, f[part], f.scales16);
  const vec = (name: string) => st.get(name).data;
  const { dModel: d, heads, kvHeads, headDim, dFF, vocab } = cfg;
  const embed = lin('model.embed_tokens.weight', vocab, d, 'emb');
  const layers: Layer[] = [];
  for (let i = 0; i < cfg.layers; i++) {
    const p = `model.layers.${i}.`;
    layers.push({
      inNorm: vec(p + 'input_layernorm.weight'),
      q: lin(p + 'self_attn.q_proj.weight', heads * headDim, d, 'attn'),
      k: lin(p + 'self_attn.k_proj.weight', kvHeads * headDim, d, 'attn'),
      v: lin(p + 'self_attn.v_proj.weight', kvHeads * headDim, d, 'attn'),
      o: lin(p + 'self_attn.o_proj.weight', d, heads * headDim, 'attn'),
      postNorm: vec(p + 'post_attention_layernorm.weight'),
      gate: lin(p + 'mlp.gate_proj.weight', dFF, d, 'mlp'),
      up: lin(p + 'mlp.up_proj.weight', dFF, d, 'mlp'),
      down: lin(p + 'mlp.down_proj.weight', d, dFF, 'down'),
    });
  }
  const lmHead = cfg.tied || !st.header['lm_head.weight'] ? embed : lin('lm_head.weight', vocab, d, 'emb');
  return { embed, lmHead, layers, finalNorm: vec('model.norm.weight') };
}

function sumBytes(w: LlamaWeights, f: (l: Linear) => number, vecBytes: number): number {
  let n = f(w.embed) + (w.lmHead === w.embed ? 0 : f(w.lmHead)) + w.finalNorm.length * vecBytes;
  for (const l of w.layers) {
    n += (l.inNorm.length + l.postNorm.length) * vecBytes;
    for (const x of [l.q, l.k, l.v, l.o, l.gate, l.up, l.down]) n += f(x);
  }
  return n;
}

/** Arbeitsspeicher der Gewichte im jeweiligen Format. */
export const weightBytes = (w: LlamaWeights) => sumBytes(w, linearBytes, 4);

/** Geschätzte Downloadgröße (f32-Teile als fp16 ausgeliefert, Skalen als fp16). */
export const downloadBytes = (w: LlamaWeights) => sumBytes(w, linearDownloadBytes, 2);
