#!/usr/bin/env node
/**
 * Kopiert die Modell-Shards aus .cache/models (von scripts/prepare-models.ts) nach dist/models,
 * damit GitHub Pages sie same-origin ausliefert. Ohne Cache (lokal) passiert nichts – die Seite
 * läuft dann nur mit dem Spielzeugmodell.
 */
import { cpSync, existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const src = join(root, '.cache/models');
const dst = join(root, 'dist/models');
if (!existsSync(src)) {
  console.log('[models] keine Shards in .cache/models – Build ohne echtes Modell');
  process.exit(0);
}
let n = 0;
for (const model of readdirSync(src)) {
  for (const edition of readdirSync(join(src, model))) {
    const dir = join(src, model, edition);
    const manifestPath = join(dir, 'manifest.json');
    if (!existsSync(manifestPath)) continue;
    const m = JSON.parse(readFileSync(manifestPath, 'utf8'));
    cpSync(dir, join(dst, model, edition), { recursive: true });
    n++;
    console.log(`[models] ${model}/${edition}: ${(m.totalBytes / 1048576).toFixed(1)} MB${m.fake ? ' (Zufallsgewichte, nur lokal)' : ''}`);
  }
}
console.log(`[models] ${n} Fassungen nach dist/models kopiert`);
