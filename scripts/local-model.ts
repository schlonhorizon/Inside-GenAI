/**
 * Lädt eine Modell-Fassung aus .cache/models in Node (Golden-Tests, Benchmark).
 * Setzt die Shards genauso zusammen wie der Browser (src/engine/model/shards.ts) und baut die
 * Gewichte mit derselben Funktion (`unpackWeights`). Prüft dabei Länge und SHA-256 jedes Shards.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { LlamaModel } from '../src/engine/model/llama.ts';
import { configFromHf } from '../src/engine/model/weights.ts';
import { unpackWeights, type ModelManifest } from '../src/engine/model/pack.ts';

export const MODELS_DIR = '.cache/models';

/** Liegt eine echte Fassung vor? (Zufallsgewichte aus `--fake` zählen nicht) */
export function hasLocalModel(model: string, edition: string, dir = MODELS_DIR): boolean {
  const p = join(dir, model, edition, 'manifest.json');
  return existsSync(p) && !JSON.parse(readFileSync(p, 'utf8')).fake;
}

export function loadLocalModel(model: string, edition: string, maxSeq = 256, dir = MODELS_DIR) {
  const base = join(dir, model, edition);
  const manifest = JSON.parse(readFileSync(join(base, 'manifest.json'), 'utf8')) as ModelManifest;
  const buf = new ArrayBuffer(manifest.totalBytes);
  const all = new Uint8Array(buf);
  for (const s of manifest.shards) {
    const data = readFileSync(join(base, s.file));
    if (data.byteLength !== s.bytes) throw new Error(`${s.file}: ${data.byteLength} statt ${s.bytes} Bytes`);
    const sha = createHash('sha256').update(data).digest('hex');
    if (sha !== s.sha256) throw new Error(`${s.file}: Prüfsumme falsch`);
    all.set(data, s.offset);
  }
  const cfg = configFromHf(manifest.config);
  const weights = unpackWeights(manifest.tensors, buf, cfg.layers);
  return { manifest, cfg, model: new LlamaModel(cfg, weights, maxSeq) };
}
