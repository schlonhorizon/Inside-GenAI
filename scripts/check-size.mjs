#!/usr/bin/env node
/**
 * Größenprüfung vor dem Deploy: GitHub Pages erlaubt höchstens 1 GB pro Seite.
 * Wir bleiben mit Abstand darunter (PLAN.md / docs/spike.md: Grenze 800 MB).
 */
import { readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const LIMIT_MB = 800;
const dir = resolve(import.meta.dirname, '..', process.argv[2] ?? 'dist');
const size = (p) => (statSync(p).isDirectory() ? readdirSync(p).reduce((s, f) => s + size(join(p, f)), 0) : statSync(p).size);
const mb = (b) => (b / 1048576).toFixed(1);
const total = size(dir);
const parts = readdirSync(dir)
  .map((f) => [f, size(join(dir, f))])
  .sort((a, b) => b[1] - a[1])
  .slice(0, 6)
  .map(([f, b]) => `${f} ${mb(b)} MB`)
  .join(', ');
console.log(`[size] ${dir}: ${mb(total)} MB (Grenze ${LIMIT_MB} MB) – größte Teile: ${parts}`);
if (total > LIMIT_MB * 1048576) {
  console.error(`[size] Zu groß für GitHub Pages: ${mb(total)} MB > ${LIMIT_MB} MB`);
  process.exit(1);
}
