/// <reference lib="webworker" />
/**
 * Rechen-Benchmark: int8-Matrix × f32-Vektor mit Skalierung pro Zeile – dieselbe Art
 * Rechnung, die die Engine später pro Token sehr oft ausführt.
 */

const ROWS = 1536;
const COLS = 576;

function setup() {
  const w = new Int8Array(ROWS * COLS);
  for (let i = 0; i < w.length; i++) w[i] = ((i * 2654435761) >>> 24) - 128;
  const scale = new Float32Array(ROWS).fill(0.01);
  const x = new Float32Array(COLS);
  for (let i = 0; i < COLS; i++) x[i] = Math.sin(i);
  return { w, scale, x, y: new Float32Array(ROWS) };
}

function matvec(w: Int8Array, scale: Float32Array, x: Float32Array, y: Float32Array) {
  for (let r = 0; r < ROWS; r++) {
    let acc = 0;
    const off = r * COLS;
    for (let c = 0; c < COLS; c++) acc += w[off + c] * x[c];
    y[r] = acc * scale[r];
  }
}

self.onmessage = (e: MessageEvent<{ durationMs: number }>) => {
  const { w, scale, x, y } = setup();
  // Aufwärmen (JIT)
  for (let i = 0; i < 20; i++) matvec(w, scale, x, y);
  const t0 = performance.now();
  let n = 0;
  while (performance.now() - t0 < e.data.durationMs) {
    matvec(w, scale, x, y);
    n++;
  }
  const ms = performance.now() - t0;
  const macs = n * ROWS * COLS;
  (self as unknown as Worker).postMessage({ gmacPerSec: macs / (ms / 1000) / 1e9, iterations: n, ms, checksum: y[0] });
};
