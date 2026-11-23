/**
 * Rechenbausteine der Engine. Alles auf Float32Array/Int8Array, ohne Abhängigkeiten,
 * damit derselbe Code im Browser (Worker) und in Node (Tests, Spike) läuft.
 */

/** Lineare Schicht y = W·x, W zeilenweise [rows × cols] (HF-Layout: [out, in]). */
export type Linear =
  | { kind: 'f32'; rows: number; cols: number; w: Float32Array }
  | { kind: 'q8'; rows: number; cols: number; q: Int8Array; scale: Float32Array };

export function linear(l: Linear, x: Float32Array, out: Float32Array): void {
  const { rows, cols } = l;
  if (l.kind === 'f32') {
    const w = l.w;
    for (let r = 0; r < rows; r++) {
      let acc = 0;
      const off = r * cols;
      for (let c = 0; c < cols; c++) acc += w[off + c] * x[c];
      out[r] = acc;
    }
  } else {
    const q = l.q;
    const s = l.scale;
    for (let r = 0; r < rows; r++) {
      let acc = 0;
      const off = r * cols;
      for (let c = 0; c < cols; c++) acc += q[off + c] * x[c];
      out[r] = acc * s[r];
    }
  }
}

/** Eine Zeile einer linearen Schicht als f32 (für Embedding-Lookups). */
export function linearRow(l: Linear, row: number, out: Float32Array): void {
  const off = row * l.cols;
  if (l.kind === 'f32') {
    out.set(l.w.subarray(off, off + l.cols));
  } else {
    const s = l.scale[row];
    for (let c = 0; c < l.cols; c++) out[c] = l.q[off + c] * s;
  }
}

/** RMSNorm wie in Llama: x / sqrt(mean(x²) + eps) · weight */
export function rmsNorm(x: Float32Array, weight: Float32Array, eps: number, out: Float32Array): void {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * weight[i];
}

/** Numerisch stabile Softmax in-place über die ersten n Werte. */
export function softmaxInPlace(x: Float32Array, n = x.length): void {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[i] - max);
    x[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) x[i] /= sum;
}

export const silu = (v: number) => v / (1 + Math.exp(-v));

/**
 * RoPE-Tabellen wie in Hugging Face Llama: Paare (i, i + d/2) werden um pos·θ^(−2i/d) gedreht.
 */
export function ropeTables(headDim: number, maxPos: number, theta: number) {
  const half = headDim / 2;
  const cos = new Float32Array(maxPos * half);
  const sin = new Float32Array(maxPos * half);
  for (let p = 0; p < maxPos; p++) {
    for (let i = 0; i < half; i++) {
      const invFreq = 1 / Math.pow(theta, (2 * i) / headDim);
      const a = p * invFreq;
      cos[p * half + i] = Math.cos(a);
      sin[p * half + i] = Math.sin(a);
    }
  }
  return { cos, sin, half };
}

/** Wendet RoPE in-place auf einen Kopf-Vektor (Länge headDim, ab Offset off) an. */
export function applyRope(v: Float32Array, off: number, pos: number, t: { cos: Float32Array; sin: Float32Array; half: number }) {
  const { cos, sin, half } = t;
  const base = pos * half;
  for (let i = 0; i < half; i++) {
    const a = v[off + i];
    const b = v[off + i + half];
    const c = cos[base + i];
    const s = sin[base + i];
    v[off + i] = a * c - b * s;
    v[off + i + half] = b * c + a * s;
  }
}
