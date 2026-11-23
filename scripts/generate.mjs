#!/usr/bin/env node
/**
 * Erzeugt Build-Daten in src/generated/ (nicht eingecheckt):
 *   - model-facts.json   Modellzahlen aus den Configs (PLAN.md 5.5)
 *   - npm-licenses.json  Lizenzen der direkten npm-Abhängigkeiten (für #/credits)
 *
 * Modell-Configs werden gesucht unter .cache/models/<id>/config.json (ab P2 vom
 * Deploy-Workflow befüllt) bzw. public/data/toy/config.json (Toy-Modell).
 * Fehlt eine Config, wird das Modell als "available: false" eingetragen.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { llamaFacts } from './model-facts-lib.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'src/generated');
mkdirSync(out, { recursive: true });

const MODELS = [
  { id: 'smollm2', label: 'SmolLM2-135M', configs: ['.cache/models/smollm2-135m/config.json'] },
  { id: 'smollm2-instruct', label: 'SmolLM2-135M-Instruct', configs: ['.cache/models/smollm2-135m-instruct/config.json'] },
  { id: 'toy', label: 'Spielzeugmodell', configs: ['public/data/toy/config.json'] },
];

const facts = {};
for (const m of MODELS) {
  const found = m.configs.map((c) => join(root, c)).find((p) => existsSync(p));
  if (!found) {
    facts[m.id] = { available: false, label: m.label, values: {} };
    continue;
  }
  const config = JSON.parse(readFileSync(found, 'utf8'));
  facts[m.id] = { available: true, label: m.label, source: found.replace(root + '/', ''), values: llamaFacts(config) };
}
writeFileSync(join(out, 'model-facts.json'), JSON.stringify(facts, null, 2) + '\n');

// npm-Lizenzen der direkten Abhängigkeiten
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
const licenses = Object.keys(pkg.dependencies ?? {})
  .sort()
  .map((name) => {
    const p = join(root, 'node_modules', name, 'package.json');
    if (!existsSync(p)) return { name, version: '?', license: 'unbekannt' };
    const d = JSON.parse(readFileSync(p, 'utf8'));
    const repo = typeof d.repository === 'string' ? d.repository : d.repository?.url;
    return { name, version: d.version, license: d.license ?? 'unbekannt', homepage: d.homepage ?? repo ?? null };
  });
writeFileSync(join(out, 'npm-licenses.json'), JSON.stringify(licenses, null, 2) + '\n');

const avail = Object.entries(facts)
  .map(([k, v]) => `${k}:${v.available ? 'ok' : '—'}`)
  .join(' ');
console.log(`[generate] model-facts (${avail}), npm-licenses (${licenses.length})`);
