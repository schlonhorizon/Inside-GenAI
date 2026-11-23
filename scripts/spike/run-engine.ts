/**
 * Spike P1a, Schritt 2: dieselben Prompts mit der eigenen TS-Engine (fp32 und int8),
 * dazu Lade-, Prefill- und Decode-Zeiten sowie Speicherbedarf.
 *
 *   node --experimental-strip-types scripts/spike/run-engine.ts [spike-out]
 *
 * Liest spike-out/reference.json (aus reference.py), schreibt spike-out/engine.json und
 * spike-out/ts-<precision>-<prompt>.f32.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseSafeTensors } from '../../src/engine/model/safetensors.ts';
import { argmax, configFromHf, LlamaModel, loadLlamaWeights, weightBytes, type Precision } from '../../src/engine/model/llama.ts';

interface RefPrompt {
  id: string;
  model: string;
  kind: string;
  ids: number[];
  greedy_ids: number[];
}
interface Ref {
  models: Record<string, { repo: string; path: string; config: unknown; safetensors: string[] }>;
  prompts: RefPrompt[];
}

const outDir = process.argv[2] ?? 'spike-out';
const ref = JSON.parse(readFileSync(join(outDir, 'reference.json'), 'utf8')) as Ref;
const MB = 1024 * 1024;
const now = () => performance.now();
const mem = () => {
  const m = process.memoryUsage();
  return { rssMB: Math.round(m.rss / MB), arrayBuffersMB: Math.round(m.arrayBuffers / MB) };
};

const result: Record<string, unknown> = { node: process.version, cpu: `${process.arch}`, runs: [] as unknown[] };
const runs = result.runs as unknown[];

for (const [mkey, minfo] of Object.entries(ref.models)) {
  const cfg = configFromHf(minfo.config);
  if (minfo.safetensors.length !== 1) throw new Error(`Erwartet genau eine safetensors-Datei, gefunden: ${minfo.safetensors}`);
  const file = readFileSync(join(minfo.path, minfo.safetensors[0]));
  const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const precisions: Precision[] = mkey === 'base' ? ['f32', 'q8'] : ['f32', 'q8'];

  for (const precision of precisions) {
    globalThis.gc?.();
    const t0 = now();
    const st = parseSafeTensors(buf);
    const weights = loadLlamaWeights(st, cfg, precision);
    const loadMs = now() - t0;
    const model = new LlamaModel(cfg, weights, 256);
    const run: Record<string, unknown> = {
      model: mkey,
      precision,
      loadMs: Math.round(loadMs),
      weightMB: Math.round(weightBytes(weights) / MB),
      prompts: [] as unknown[],
    };
    console.log(`[engine] ${mkey}/${precision}: geladen in ${(loadMs / 1000).toFixed(1)} s, Gewichte ${run.weightMB} MB`);

    for (const p of ref.prompts.filter((x) => x.model === mkey)) {
      model.reset();
      const tp = now();
      const logits = model.prefill(p.ids);
      const prefillMs = now() - tp;
      writeFileSync(join(outDir, `ts-${precision}-${p.id}.f32`), Buffer.from(logits.buffer.slice(0)));
      const greedy: number[] = [];
      let next = argmax(logits);
      const td = now();
      for (let i = 0; i < p.greedy_ids.length; i++) {
        greedy.push(next);
        next = argmax(model.step(next)!);
      }
      const decodeMs = (now() - td) / p.greedy_ids.length;
      (run.prompts as unknown[]).push({ id: p.id, greedy_ids: greedy, prefillMs: Math.round(prefillMs), decodeMsPerToken: Math.round(decodeMs) });
      console.log(`[engine]   ${p.id}: prefill ${p.ids.length} Tokens ${Math.round(prefillMs)} ms, decode ${Math.round(decodeMs)} ms/Token`);
    }

    // Leistung nur für das Base-Modell (Instruct ist architekturgleich)
    if (mkey === 'base') {
      const seed = ref.prompts[0].ids;
      const prefill: Record<string, number> = {};
      for (const n of [16, 64, 128]) {
        const toks = Array.from({ length: n }, (_, i) => seed[i % seed.length]);
        model.reset();
        const t = now();
        model.prefill(toks);
        prefill[n] = Math.round(now() - t);
      }
      model.reset();
      model.prefill(seed);
      const steps = 32;
      let tok = seed[seed.length - 1];
      const t = now();
      for (let i = 0; i < steps; i++) tok = argmax(model.step(tok)!);
      run.perf = { prefillMs: prefill, decodeMsPerToken: Math.round((now() - t) / steps), contextAtDecodeStart: seed.length };
      console.log(`[engine]   perf: prefill ${JSON.stringify(prefill)} ms, decode ${(run.perf as { decodeMsPerToken: number }).decodeMsPerToken} ms/Token`);
    }
    run.memory = mem();
    runs.push(run);
  }
}

writeFileSync(join(outDir, 'engine.json'), JSON.stringify(result, null, 2) + '\n');
console.log('[engine] fertig');
