import type { Linear } from '../core/ops.ts';
import { f16BitsToF32, f32ToF16Bits } from './quantize.ts';
import type { Layer, LlamaWeights } from './weights.ts';

/**
 * Auslieferungsformat der Modellgewichte (Shards).
 *
 * Alle Tensoren liegen hintereinander in einem Datenstrom (16-Byte-ausgerichtet). Der Strom wird
 * für den Download in Shards zu höchstens 32 MB geschnitten; ein Tensor darf über eine
 * Shard-Grenze reichen. Beim Laden werden die Shards wieder zu einem Puffer zusammengesetzt.
 *
 *   f16   Gewichtsmatrix als fp16-Bitmuster (exakt, weil vorher auf fp16 gerundet)
 *   f32   Gewichtsmatrix als f32
 *   q8g   int8-Werte + Skalen je Gruppe als fp16
 *   q8    int8-Werte + Skala je Zeile als fp16
 *   vec   Vektor (RMSNorm-Gewichte) als f32
 *
 * CI (scripts/prepare-models.ts) schreibt mit `packWeights`, Browser und Golden-Tests lesen mit
 * `unpackWeights`, also mit genau demselben Code.
 */

export interface PackedTensor {
  name: string;
  kind: 'f16' | 'f32' | 'q8g' | 'q8' | 'vec';
  rows: number;
  cols: number;
  group?: number;
  offset: number;
  bytes: number;
  scaleOffset?: number;
  scaleBytes?: number;
}

export interface ShardInfo {
  file: string;
  offset: number;
  bytes: number;
  sha256: string;
}

export interface ModelManifest {
  format: 1;
  /** z. B. "smollm2-135m" */
  model: string;
  /** "int8" oder "fp16" */
  edition: string;
  repo: string;
  revision: string;
  /** Gewichtsformat je Modellteil, z. B. "attn:f16,mlp:q8g32,down:q8g32,emb:q8g32+s16" */
  spec: string;
  /** Kurzer Hash über alle Shards: ändert sich, sobald sich die Daten ändern */
  version: string;
  config: Record<string, unknown>;
  generation: Record<string, unknown>;
  tensors: PackedTensor[];
  totalBytes: number;
  shards: ShardInfo[];
  /** Arbeitsspeicher der Gewichte nach dem Laden */
  weightBytes: number;
  /** nur lokal: Zufallsgewichte zum Testen der Oberfläche */
  fake?: boolean;
}

export const SHARD_BYTES = 32 * 1024 * 1024;
const ALIGN = 16;

const layerParts: (keyof Layer)[] = ['inNorm', 'q', 'k', 'v', 'o', 'postNorm', 'gate', 'up', 'down'];

function entries(w: LlamaWeights): [string, Linear | Float32Array][] {
  const out: [string, Linear | Float32Array][] = [['embed', w.embed]];
  w.layers.forEach((L, i) => layerParts.forEach((p) => out.push([`layers.${i}.${p}`, L[p]])));
  out.push(['finalNorm', w.finalNorm]);
  if (w.lmHead !== w.embed) out.push(['lmHead', w.lmHead]);
  return out;
}

const isF16Exact = (a: Float32Array) => {
  for (let i = 0; i < a.length; i++) if (f16BitsToF32(f32ToF16Bits(a[i])) !== a[i]) return false;
  return true;
};

/** Gewichte → Tensorliste + ein zusammenhängender Datenstrom */
export function packWeights(w: LlamaWeights): { tensors: PackedTensor[]; data: Uint8Array } {
  const tensors: PackedTensor[] = [];
  const chunks: { offset: number; bytes: Uint8Array }[] = [];
  let pos = 0;
  const put = (bytes: Uint8Array) => {
    pos = Math.ceil(pos / ALIGN) * ALIGN;
    const offset = pos;
    chunks.push({ offset, bytes });
    pos += bytes.byteLength;
    return offset;
  };
  const f16Bytes = (a: Float32Array) => {
    const u = new Uint16Array(a.length);
    for (let i = 0; i < a.length; i++) u[i] = f32ToF16Bits(a[i]);
    return new Uint8Array(u.buffer);
  };
  const f32Bytes = (a: Float32Array) => new Uint8Array(a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength));

  for (const [name, t] of entries(w)) {
    if (t instanceof Float32Array) {
      const bytes = f32Bytes(t);
      tensors.push({ name, kind: 'vec', rows: 1, cols: t.length, offset: put(bytes), bytes: bytes.byteLength });
    } else if (t.kind === 'f32') {
      const exact = isF16Exact(t.w);
      const bytes = exact ? f16Bytes(t.w) : f32Bytes(t.w);
      tensors.push({ name, kind: exact ? 'f16' : 'f32', rows: t.rows, cols: t.cols, offset: put(bytes), bytes: bytes.byteLength });
    } else {
      if (!isF16Exact(t.scale)) throw new Error(`${name}: Skalen nicht fp16-exakt – Format mit "+s16" verwenden`);
      const q = new Uint8Array(t.q.buffer.slice(t.q.byteOffset, t.q.byteOffset + t.q.byteLength));
      const offset = put(q);
      const s = f16Bytes(t.scale);
      const scaleOffset = put(s);
      tensors.push({
        name,
        kind: t.kind,
        rows: t.rows,
        cols: t.cols,
        group: t.kind === 'q8g' ? t.group : undefined,
        offset,
        bytes: q.byteLength,
        scaleOffset,
        scaleBytes: s.byteLength,
      });
    }
  }
  const data = new Uint8Array(Math.ceil(pos / ALIGN) * ALIGN);
  for (const c of chunks) data.set(c.bytes, c.offset);
  return { tensors, data };
}

/** Schneidet den Datenstrom in Shards (Byte-Bereiche) */
export function shardRanges(total: number, size = SHARD_BYTES): { offset: number; bytes: number }[] {
  const out = [];
  for (let o = 0; o < total; o += size) out.push({ offset: o, bytes: Math.min(size, total - o) });
  return out;
}

/**
 * Datenstrom → Gewichte. int8-Werte und f32-Matrizen sind Sichten auf den Puffer (keine Kopie),
 * fp16-Matrizen und Skalen werden in f32 umgewandelt, Vektoren kopiert.
 */
export function unpackWeights(tensors: PackedTensor[], buf: ArrayBuffer, layers: number): LlamaWeights {
  const byName = new Map(tensors.map((t) => [t.name, t]));
  const f16 = (offset: number, n: number) => {
    const u = new Uint16Array(buf, offset, n);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = f16BitsToF32(u[i]);
    return out;
  };
  const vec = (name: string): Float32Array => {
    const t = byName.get(name);
    if (!t || t.kind !== 'vec') throw new Error(`Vektor fehlt: ${name}`);
    // Kopie: Vektoren sind klein, und so hält bei der fp16-Fassung nichts den Download-Puffer fest
    return new Float32Array(buf, t.offset, t.cols).slice();
  };
  const lin = (name: string): Linear => {
    const t = byName.get(name);
    if (!t) throw new Error(`Tensor fehlt: ${name}`);
    const { rows, cols } = t;
    switch (t.kind) {
      case 'f16':
        return { kind: 'f32', rows, cols, w: f16(t.offset, rows * cols) };
      case 'f32':
        return { kind: 'f32', rows, cols, w: new Float32Array(buf, t.offset, rows * cols) };
      case 'q8':
        return { kind: 'q8', rows, cols, q: new Int8Array(buf, t.offset, rows * cols), scale: f16(t.scaleOffset!, rows) };
      case 'q8g':
        return {
          kind: 'q8g',
          rows,
          cols,
          group: t.group!,
          q: new Int8Array(buf, t.offset, rows * cols),
          scale: f16(t.scaleOffset!, rows * (cols / t.group!)),
        };
      default:
        throw new Error(`${name}: unerwartete Art ${t.kind}`);
    }
  };
  const embed = lin('embed');
  const out: LlamaWeights = {
    embed,
    lmHead: byName.has('lmHead') ? lin('lmHead') : embed,
    finalNorm: vec('finalNorm'),
    layers: [],
  };
  for (let i = 0; i < layers; i++) {
    const p = `layers.${i}.`;
    out.layers.push({
      inNorm: vec(p + 'inNorm'),
      q: lin(p + 'q'),
      k: lin(p + 'k'),
      v: lin(p + 'v'),
      o: lin(p + 'o'),
      postNorm: vec(p + 'postNorm'),
      gate: lin(p + 'gate'),
      up: lin(p + 'up'),
      down: lin(p + 'down'),
    });
  }
  return out;
}

/**
 * Arbeitsspeicher nach dem Laden: umgewandelte f32-Daten plus der Download-Puffer, solange
 * int8-Sichten auf ihn zeigen (bei der int8-Fassung immer, bei der fp16-Fassung nie).
 */
export function residentBytes(tensors: PackedTensor[], totalBytes: number): number {
  let converted = 0;
  let views = false;
  for (const t of tensors) {
    if (t.kind === 'f16') converted += t.rows * t.cols * 4;
    else if (t.kind === 'vec') converted += t.cols * 4;
    else views = true;
    if (t.scaleBytes) converted += t.scaleBytes * 2;
  }
  return converted + (views ? totalBytes : 0);
}

/** SHA-256 als Hex (WebCrypto, im Browser-Worker und in Node verfügbar) */
export async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const view = data instanceof Uint8Array ? data : new Uint8Array(data);
  const digest = await crypto.subtle.digest('SHA-256', view as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}
