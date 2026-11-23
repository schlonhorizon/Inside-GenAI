#!/usr/bin/env node
/**
 * Screenshots aller Routen in Hell und Dunkel + Smoke-Test.
 *
 *   node scripts/screenshots.mjs [baseUrl] [outDir]
 *   (Standard: http://localhost:4173/  screenshots/)
 *
 * Erwartet einen laufenden Server mit der gebauten Seite (vite preview).
 * Exit-Code 1, wenn eine Route Konsolenfehler wirft, eine Ressource 4xx/5xx liefert
 * oder eine Anfrage an eine fremde Domain geht.
 */
import { mkdirSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const base = (process.argv[2] ?? 'http://localhost:4173/').replace(/\/?$/, '/');
const outDir = resolve(process.argv[3] ?? join(root, 'screenshots'));
mkdirSync(outDir, { recursive: true });

const slugs = readdirSync(join(root, 'content/chapters'))
  .filter((f) => f.endsWith('.mdx'))
  .sort()
  .map((f) => /\nslug:\s*([\w-]+)/.exec(readFileSync(join(root, 'content/chapters', f), 'utf8'))[1]);

const routes = [
  ['00-start', '#/'],
  ...slugs.map((s, i) => [`kapitel-${String(i).padStart(2, '0')}-${s}`, `#/kapitel/${s}`]),
  ['glossar', '#/glossar'],
  ['credits', '#/credits'],
  ['diagnose', '#/diagnose'],
  ['styleguide', '#/styleguide'],
  ['404', '#/gibt-es-nicht'],
];

const launchOpts = process.env.PW_CHROMIUM_PATH ? { executablePath: process.env.PW_CHROMIUM_PATH } : {};
const browser = await chromium.launch(launchOpts);
const baseHost = new URL(base).host;
const problems = [];

for (const scheme of ['light', 'dark']) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 }, colorScheme: scheme, reducedMotion: 'reduce' });
  for (const [name, hash] of routes) {
    const page = await ctx.newPage();
    const errs = [];
    page.on('console', (m) => m.type() === 'error' && errs.push(`console: ${m.text()}`));
    page.on('pageerror', (e) => errs.push(`pageerror: ${e.message}`));
    page.on('response', (r) => r.status() >= 400 && errs.push(`HTTP ${r.status()} ${r.url()}`));
    page.on('request', (r) => {
      const u = new URL(r.url());
      if (u.protocol.startsWith('http') && u.host !== baseHost) errs.push(`extern: ${r.url()}`);
    });
    await page.goto(base + hash, { waitUntil: 'networkidle' });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(300);
    await page.screenshot({ path: join(outDir, `${name}--${scheme}.png`), fullPage: true });
    if (errs.length) problems.push(`${name} (${scheme}):\n  ${errs.join('\n  ')}`);
    await page.close();
  }
  await ctx.close();
}
await browser.close();

console.log(`[screenshots] ${routes.length * 2} Bilder → ${outDir}`);
if (problems.length) {
  console.error(`[smoke] Probleme:\n${problems.join('\n')}`);
  process.exit(1);
}
console.log('[smoke] alle Routen ohne Fehler');
