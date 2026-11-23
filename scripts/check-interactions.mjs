#!/usr/bin/env node
/**
 * Interaktionsprüfung der MVP-Kapitel (P1b-DoD) gegen die gebaute Seite:
 *   node scripts/check-interactions.mjs [baseUrl]
 * Prüft: Stempel an jeder Abbildung, Regler wirken, Seed reproduzierbar, Token erzwingen →
 * neuer Ast, Autovervollständigung, Navigation ohne Netz nach dem ersten Laden (ohne Reload).
 */
import { chromium } from 'playwright';

const base = (process.argv[2] ?? 'http://localhost:4173/').replace(/\/?$/, '/');
const launchOpts = process.env.PW_CHROMIUM_PATH ? { executablePath: process.env.PW_CHROMIUM_PATH } : {};
const browser = await chromium.launch(launchOpts);
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

const results = [];
async function check(name, fn) {
  try {
    await fn();
    results.push(['✓', name]);
  } catch (e) {
    results.push(['✗', `${name}: ${e.message.split('\n')[0]}`]);
  }
}
const assert = (cond, msg) => {
  if (!cond) throw new Error(msg);
};
async function go(hash, waitFor) {
  await page.evaluate((h) => (location.hash = h), hash);
  if (waitFor) await page.waitForSelector(waitFor, { timeout: 10000 });
}
async function setRange(locator, value) {
  await locator.evaluate((el, v) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    setter.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
}

await page.goto(base + '#/', { waitUntil: 'networkidle' });

const MVP = [
  ['karte', '.map'],
  ['tokenisierung', '.tok-row .chip'],
  ['next-token', '.dist__bars .bar'],
  ['generierung', '.gen-line'],
];

await check('jede Abbildung der MVP-Kapitel trägt einen Stempel', async () => {
  for (const [slug, sel] of MVP) {
    await go(`#/kapitel/${slug}`, sel);
    await page.waitForTimeout(300);
    const missing = await page.$$eval('figure.figure', (fs) => fs.filter((f) => !f.querySelector('.figure__caption .stamp')).length);
    assert(missing === 0, `${slug}: ${missing} Abbildung(en) ohne Stempel`);
  }
});

await check('Kap. 7: Temperatur verändert die Verteilung sofort', async () => {
  await go('#/kapitel/next-token', '.dist__bars .bar');
  const fig = page.locator('figure', { hasText: 'Abb. 7.2' });
  const top = () => fig.locator('.dist__bars .bar').first().locator('.bar__p').innerText();
  const before = await top();
  await setRange(fig.locator('input[type=range]').first(), 0.3);
  await page.waitForTimeout(100);
  const after = await top();
  assert(before !== after, `Top-Wahrscheinlichkeit unverändert (${before})`);
  await setRange(fig.locator('input[type=range]').nth(2), 0.5);
  await page.waitForTimeout(100);
  const cut = await fig.locator('.bar--cut').count();
  assert(cut > 0, 'Top-p 0,5 schneidet nichts ab');
});

await check('Playground: gleicher Seed → gleicher Text, anderer Seed → anderer Text', async () => {
  await go('#/playground', '.play-out .play-tok');
  const out = () => page.locator('.play-out').innerText();
  const seed = page.locator('.seed-input input');
  await seed.fill('11');
  const a = await out();
  await seed.fill('12');
  const b = await out();
  await seed.fill('11');
  const c = await out();
  assert(a === c, 'Seed 11 liefert unterschiedliche Texte');
  assert(a !== b, 'Seed 11 und 12 liefern denselben Text');
});

await check('Playground: Autovervollständigung und Markierung unbekannter Wörter', async () => {
  const input = page.locator('.prompt-input input.text-input').first();
  await input.fill('die haupt');
  await input.focus();
  await page.waitForSelector('#prompt-suggestions li', { timeout: 3000 });
  await input.press('Enter');
  const v = await input.inputValue();
  assert(v === 'die hauptstadt ', `Vervollständigung ergab „${v}“`);
  await input.fill('die quark');
  await page.waitForSelector('.prompt-input__warn', { timeout: 3000 });
  assert((await page.locator('.chip--unknown').count()) > 0, 'unbekanntes Wort nicht markiert');
});

await check('Kap. 8: Schritte erzeugen Tokens, Erzwingen erzeugt einen neuen Ast', async () => {
  await go('#/kapitel/generierung', '.gen-line');
  const fig = page.locator('figure', { hasText: 'Abb. 8.1' });
  const stepBtn = fig.getByRole('button', { name: 'Ein Schritt' });
  // schrittweise, bis der Satz endet (EOS) oder 4 Tokens erzeugt sind
  for (let i = 0; i < 4 && (await stepBtn.isEnabled()); i++) await stepBtn.click();
  const toks = await fig.locator('.gen-tok').count();
  assert(toks >= 1, 'keine Tokens erzeugt');
  const nodesBefore = await fig.locator('.tree__node').count();
  await fig.locator('.gen-tok').first().click();
  await fig.locator('.alt:not(.alt--chosen)').first().click();
  const nodesAfter = await fig.locator('.tree__node').count();
  const forced = await fig.locator('.tree__box.is-forced').count();
  assert(nodesAfter > nodesBefore, 'kein neuer Knoten im Baum');
  assert(forced === 1, `erwartet 1 erzwungenen Knoten, gefunden ${forced}`);
});

await check('ohne Netz: Navigation zwischen allen MVP-Routen nach dem ersten Laden (ohne Reload)', async () => {
  await page.goto(base + '#/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(5000); // Vorladen im Leerlauf abwarten
  await ctx.setOffline(true);
  for (const [slug, sel] of [...MVP, ['playground', '.play-out .play-tok']]) {
    await go(slug === 'playground' ? '#/playground' : `#/kapitel/${slug}`, sel);
  }
  await ctx.setOffline(false);
});

await browser.close();
for (const [s, n] of results) console.log(`${s} ${n}`);
const failed = results.filter(([s]) => s === '✗').length;
const relevantErrors = errors.filter((e) => !/net::ERR_INTERNET_DISCONNECTED|Failed to fetch/.test(e));
if (relevantErrors.length) console.log(`Konsolenfehler:\n  ${relevantErrors.join('\n  ')}`);
process.exit(failed || relevantErrors.length ? 1 : 0);
