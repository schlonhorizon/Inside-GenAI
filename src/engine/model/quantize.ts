import type { Linear } from '../core/ops.ts';

/**
 * Symmetrische int8-Quantisierung pro Zeile: q = round(w / s), s = max|w_row| / 127.
 * Jede Zeile (= ein Ausgangsneuron) bekommt ihren eigenen Skalierungsfaktor.
 */
export function quantizeRows(w: Float32Array, rows: number, cols: number): Linear {
  const q = new Int8Array(rows * cols);
  const scale = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let max = 0;
    for (let c = 0; c < cols; c++) max = Math.max(max, Math.abs(w[off + c]));
    const s = max / 127 || 1;
    scale[r] = s;
    for (let c = 0; c < cols; c++) q[off + c] = Math.max(-127, Math.min(127, Math.round(w[off + c] / s)));
  }
  return { kind: 'q8', rows, cols, q, scale };
}

export function linearF32(w: Float32Array, rows: number, cols: number): Linear {
  if (w.length !== rows * cols) throw new Error(`Form passt nicht: ${w.length} ≠ ${rows}×${cols}`);
  return { kind: 'f32', rows, cols, w };
}

export function linearBytes(l: Linear): number {
  return l.kind === 'f32' ? l.w.byteLength : l.q.byteLength + l.scale.byteLength;
}
