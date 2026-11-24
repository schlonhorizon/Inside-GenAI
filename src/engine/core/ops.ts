/**
 * Rechenbausteine der Engine. Alles auf Float32Array/Int8Array, ohne Abhängigkeiten,
 * damit derselbe Code im Browser (Worker) und in Node (Tests, Spike) läuft.
 */

/** Lineare Schicht y = W·x, W zeilenweise [rows × cols] (HF-Layout: [out, in]). */
export type Linear =
  | { kind: 'f32'; rows: number; cols: number; w: Float32Array }
  | { kind: 'q8'; rows: number; cols: number; q: Int8Array; scale: Float32Array }
  /** int8 mit eigenem Skalierungsfaktor je Gruppe von `group` Werten innerhalb einer Zeile */
  | { kind: 'q8g'; rows: number; cols: number; group: number; q: Int8Array; scale: Float32Array };

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
  } else if (l.kind === 'q8') {
    const q = l.q;
    const s = l.scale;
    for (let r = 0; r < rows; r++) {
      let acc = 0;
      const off = r * cols;
      for (let c = 0; c < cols; c++) acc += q[off + c] * x[c];
      out[r] = acc * s[r];
    }
  } else {
    const { q, scale, group } = l;
    const groups = cols / group;
    for (let r = 0; r < rows; r++) {
      let acc = 0;
      const off = r * cols;
      for (let g = 0; g < groups; g++) {
        let part = 0;
        const go = off + g * group;
        const xo = g * group;
        for (let c = 0; c < group; c++) part += q[go + c] * x[xo + c];
        acc += part * scale[r * groups + g];
      }
      out[r] = acc;
    }
  }
}

/** Eine Zeile einer linearen Schicht als f32 (für Embedding-Lookups). */
export function linearRow(l: Linear, row: number, out: Float32Array): void {
  const off = row * l.cols;
  if (l.kind === 'f32') {
    out.set(l.w.subarray(off, off + l.cols));
  } else if (l.kind === 'q8') {
    const s = l.scale[row];
    for (let c = 0; c < l.cols; c++) out[c] = l.q[off + c] * s;
  } else {
    const groups = l.cols / l.group;
    for (let c = 0; c < l.cols; c++) out[c] = l.q[off + c] * l.scale[row * groups + Math.floor(c / l.group)];
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

/** Arbeitspuffer für verschränkte 8er-Blöcke, als f64: jeder Eingabewert wird einmal pro Schicht umgewandelt statt einmal pro Gewichtszeile. */
let packed = new Float64Array(0);

/**
 * Legt je 8 aufeinanderfolgende Vektoren verschränkt ab: P[(b·cols + c)·8 + j] = X[(8b + j)·cols + c].
 * Die innere Schleife liest dann 8 benachbarte Werte statt 8 weit auseinanderliegender.
 */
function pack8(X: Float32Array, blocks: number, cols: number): Float64Array {
  const need = blocks * cols * 8;
  if (packed.length < need) packed = new Float64Array(need);
  const P = packed;
  for (let b = 0; b < blocks; b++) {
    const pb = b * cols * 8;
    for (let j = 0; j < 8; j++) {
      const xo = (b * 8 + j) * cols;
      for (let c = 0; c < cols; c++) P[pb + c * 8 + j] = X[xo + c];
    }
  }
  return P;
}

/**
 * Dieselbe lineare Schicht für n Vektoren auf einmal (Batch-Prefill):
 * X ist [n × cols] zeilenweise, Y wird [n × rows]. Jede Gewichtszeile wird einmal geladen und
 * in Blöcken zu 8 Vektoren angewendet (ein geladenes Gewicht → 8 Multiplikationen).
 * Die Summationsreihenfolge je Ausgabewert ist identisch mit `linear`, die Ergebnisse sind
 * daher bitgleich mit dem Token-für-Token-Weg.
 */
export function linearBatch(l: Linear, X: Float32Array, n: number, Y: Float32Array): void {
  const { rows, cols } = l;
  const blocks = n >> 3;
  const n8 = blocks * 8;
  const P = blocks ? pack8(X, blocks, cols) : X;
  if (l.kind === 'f32' || l.kind === 'q8') {
    const w = l.kind === 'f32' ? l.w : l.q;
    const scale = l.kind === 'q8' ? l.scale : null;
    for (let r = 0; r < rows; r++) {
      const off = r * cols;
      const s = scale ? scale[r] : 1;
      for (let b = 0; b < blocks; b++) {
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        let pi = b * cols * 8;
        for (let c = 0; c < cols; c++, pi += 8) {
          const wv = w[off + c];
          a0 += wv * P[pi];
          a1 += wv * P[pi + 1];
          a2 += wv * P[pi + 2];
          a3 += wv * P[pi + 3];
          a4 += wv * P[pi + 4];
          a5 += wv * P[pi + 5];
          a6 += wv * P[pi + 6];
          a7 += wv * P[pi + 7];
        }
        if (scale) {
          a0 *= s;
          a1 *= s;
          a2 *= s;
          a3 *= s;
          a4 *= s;
          a5 *= s;
          a6 *= s;
          a7 *= s;
        }
        const t = b * 8;
        Y[t * rows + r] = a0;
        Y[(t + 1) * rows + r] = a1;
        Y[(t + 2) * rows + r] = a2;
        Y[(t + 3) * rows + r] = a3;
        Y[(t + 4) * rows + r] = a4;
        Y[(t + 5) * rows + r] = a5;
        Y[(t + 6) * rows + r] = a6;
        Y[(t + 7) * rows + r] = a7;
      }
      for (let t = n8; t < n; t++) {
        const xo = t * cols;
        let acc = 0;
        for (let c = 0; c < cols; c++) acc += w[off + c] * X[xo + c];
        Y[t * rows + r] = scale ? acc * s : acc;
      }
    }
  } else {
    const { q, scale, group } = l;
    const groups = cols / group;
    for (let r = 0; r < rows; r++) {
      const off = r * cols;
      const so = r * groups;
      for (let b = 0; b < blocks; b++) {
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        let pi = b * cols * 8;
        for (let g = 0; g < groups; g++) {
          let p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0;
          const go = off + g * group;
          for (let c = 0; c < group; c++, pi += 8) {
            const wv = q[go + c];
            p0 += wv * P[pi];
            p1 += wv * P[pi + 1];
            p2 += wv * P[pi + 2];
            p3 += wv * P[pi + 3];
            p4 += wv * P[pi + 4];
            p5 += wv * P[pi + 5];
            p6 += wv * P[pi + 6];
            p7 += wv * P[pi + 7];
          }
          const s = scale[so + g];
          a0 += p0 * s;
          a1 += p1 * s;
          a2 += p2 * s;
          a3 += p3 * s;
          a4 += p4 * s;
          a5 += p5 * s;
          a6 += p6 * s;
          a7 += p7 * s;
        }
        const t = b * 8;
        Y[t * rows + r] = a0;
        Y[(t + 1) * rows + r] = a1;
        Y[(t + 2) * rows + r] = a2;
        Y[(t + 3) * rows + r] = a3;
        Y[(t + 4) * rows + r] = a4;
        Y[(t + 5) * rows + r] = a5;
        Y[(t + 6) * rows + r] = a6;
        Y[(t + 7) * rows + r] = a7;
      }
      for (let t = n8; t < n; t++) {
        const xb = t * cols;
        let acc = 0;
        for (let g = 0; g < groups; g++) {
          let part = 0;
          const go = off + g * group;
          const xo = xb + g * group;
          for (let c = 0; c < group; c++) part += q[go + c] * X[xo + c];
          acc += part * scale[so + g];
        }
        Y[t * rows + r] = acc;
      }
    }
  }
}
