import type { Linear } from '../core/ops.ts';

/**
 * Symmetrische int8-Quantisierung pro Zeile: q = round(w / s), s = max|w_row| / 127.
 * Jede Zeile (= ein Ausgangsneuron) bekommt ihren eigenen Skalierungsfaktor.
 */
export function quantizeRows(w: Float32Array, rows: number, cols: number, scales16 = false): Linear {
  const q = new Int8Array(rows * cols);
  const scale = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let max = 0;
    for (let c = 0; c < cols; c++) max = Math.max(max, Math.abs(w[off + c]));
    const s = scales16 ? roundToF16(max / 127 || 1) : max / 127 || 1;
    scale[r] = s;
    for (let c = 0; c < cols; c++) q[off + c] = Math.max(-127, Math.min(127, Math.round(w[off + c] / s)));
  }
  return { kind: 'q8', rows, cols, q, scale };
}

/**
 * int8 in Gruppen: Jede Zeile wird in Blöcke zu `group` Werten geteilt, jeder Block hat
 * seinen eigenen Skalierungsfaktor. Ausreißer verderben so nur ihren Block, nicht die ganze Zeile.
 */
export function quantizeGroups(w: Float32Array, rows: number, cols: number, group: number, scales16 = false): Linear {
  if (cols % group !== 0) return quantizeRows(w, rows, cols, scales16);
  const groups = cols / group;
  const q = new Int8Array(rows * cols);
  const scale = new Float32Array(rows * groups);
  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < groups; g++) {
      const off = r * cols + g * group;
      let max = 0;
      for (let c = 0; c < group; c++) max = Math.max(max, Math.abs(w[off + c]));
      const s = scales16 ? roundToF16(max / 127 || 1) : max / 127 || 1;
      scale[r * groups + g] = s;
      for (let c = 0; c < group; c++) q[off + c] = Math.max(-127, Math.min(127, Math.round(w[off + c] / s)));
    }
  }
  return { kind: 'q8g', rows, cols, group, q, scale };
}

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);

/** Rundet einen f32-Wert auf fp16-Genauigkeit (round to nearest even) und zurück. */
export function roundToF16(v: number): number {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  let h: number;
  if (exp >= 0x1f) h = sign | 0x7c00;
  else if (exp <= 0) {
    if (exp < -10) h = sign;
    else {
      mant |= 0x800000;
      const shift = 14 - exp;
      let m = mant >>> shift;
      const rem = mant & ((1 << shift) - 1);
      const half = 1 << (shift - 1);
      if (rem > half || (rem === half && m & 1)) m++;
      h = sign | m;
    }
  } else {
    let m = mant >>> 13;
    const rem = mant & 0x1fff;
    let e = exp;
    if (rem > 0x1000 || (rem === 0x1000 && m & 1)) {
      m++;
      if (m === 0x400) {
        m = 0;
        e++;
      }
    }
    h = e >= 0x1f ? sign | 0x7c00 : sign | (e << 10) | m;
  }
  const s = h & 0x8000 ? -1 : 1;
  const e2 = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e2 === 0) return s * 2 ** -14 * (f / 1024);
  if (e2 === 0x1f) return s * Infinity;
  return s * 2 ** (e2 - 15) * (1 + f / 1024);
}

export function toF16Precision(w: Float32Array): Float32Array {
  const out = new Float32Array(w.length);
  for (let i = 0; i < w.length; i++) out[i] = roundToF16(w[i]);
  return out;
}

export function linearF32(w: Float32Array, rows: number, cols: number): Linear {
  if (w.length !== rows * cols) throw new Error(`Form passt nicht: ${w.length} ≠ ${rows}×${cols}`);
  return { kind: 'f32', rows, cols, w };
}

export function linearBytes(l: Linear): number {
  return l.kind === 'f32' ? l.w.byteLength : l.q.byteLength + l.scale.byteLength;
}

/** Bytes als Download: f32-Gewichte zählen als fp16 (so würden sie ausgeliefert), int8-Skalen als fp16. */
export function linearDownloadBytes(l: Linear): number {
  return l.kind === 'f32' ? l.w.length * 2 : l.q.byteLength + l.scale.length * 2;
}
