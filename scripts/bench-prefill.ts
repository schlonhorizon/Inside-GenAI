/**
 * Benchmark Batch-Prefill vs. Token für Token am echten Modell (DoD P2a: mindestens 2× schneller).
 *
 *   node --experimental-strip-types scripts/bench-prefill.ts [tokens=64]
 *
 * Schreibt das Ergebnis ins Log und – im CI – in die Job-Zusammenfassung. Unter 2× gibt es eine
 * Warnung (keine harte Grenze: Zeitmessungen auf geteilten Runnern schwanken).
 */
import { appendFileSync } from 'node:fs';
import { hasLocalModel, loadLocalModel } from './local-model.ts';

const n = Number(process.argv[2] ?? 64);
const MODEL = 'smollm2-135m';
const EDITION = 'int8';
if (!hasLocalModel(MODEL, EDITION)) {
  console.log('[bench] keine echten Shards in .cache/models – übersprungen');
  process.exit(0);
}
const { model } = loadLocalModel(MODEL, EDITION, 256);
const ids = Array.from({ length: n }, (_, i) => 1000 + ((i * 37) % 3000));
const time = (f: () => void) => {
  const t = performance.now();
  f();
  return performance.now() - t;
};
// Aufwärmen (JIT), dann je zweimal messen und den besseren Wert nehmen
model.prefillBatch(ids.slice(0, 16));
model.reset();
let perToken = Infinity;
let batch = Infinity;
for (let r = 0; r < 2; r++) {
  model.reset();
  perToken = Math.min(perToken, time(() => model.prefill(ids)));
  model.reset();
  batch = Math.min(batch, time(() => model.prefillBatch(ids)));
}
const factor = perToken / batch;
const line = `Prefill ${n} Tokens (SmolLM2-135M, int8): Token für Token ${(perToken / 1000).toFixed(2)} s · Batch ${(batch / 1000).toFixed(2)} s · Faktor ${factor.toFixed(2)}×`;
console.log(`[bench] ${line}`);
if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, `### Batch-Prefill\n\n${line}\n`);
if (factor < 2) console.log(`::warning::Batch-Prefill nur ${factor.toFixed(2)}× schneller (Ziel ≥ 2×)`);
