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
// Toy-Kennzahlen aus der Auswertung (public/data/toy/eval.json) – für <Fact m="toy" k="…" />
const evalPath = join(root, 'public/data/toy/eval.json');
if (facts.toy.available && existsSync(evalPath)) {
  const ev = JSON.parse(readFileSync(evalPath, 'utf8'));
  const top = Object.fromEntries(ev.leitbeispiel.ist.top5);
  const soll = Object.fromEntries(ev.leitbeispiel.soll.top5);
  Object.assign(facts.toy.values, {
    leit_paris_p: top.paris ?? 0,
    leit_schoen_p: top['schön'] ?? 0,
    leit_soll_paris_p: soll.paris ?? 0,
    eval_positions: ev.after.model.positions,
    eval_below80_share: ev.after.model.below80_share,
    eval_entropy_bits: ev.after.model.entropy_bits,
    eval_kl_bits: ev.after.kl_bits,
    soll_below80_share: ev.after.soll.below80_share,
    soll_entropy_bits: ev.after.soll.entropy_bits,
    before_below80_share: ev.before?.model.below80_share ?? 0,
    before_entropy_bits: ev.before?.model.entropy_bits ?? 0,
  });
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
