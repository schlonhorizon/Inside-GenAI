#!/usr/bin/env node
/**
 * Interaktionsprüfung der MVP-Kapitel (P1b-DoD) gegen die gebaute Seite:
 *   node scripts/check-interactions.mjs [baseUrl] [--models]
 * Prüft: Stempel an jeder Abbildung, Regler wirken, Seed reproduzierbar, Token erzwingen →
 * neuer Ast, Autovervollständigung, Navigation ohne Netz nach dem ersten Laden (ohne Reload).
 *
 * Mit --models zusätzlich das echte Modell (P2a-DoD): Laden mit Fortschritt, Kapitel 7/8 und
 * Playground rechnen echt, Stempel, „neueste Anfrage gewinnt“, Instruct-Chat stoppt am End-Token,
 * zweiter Besuch ohne Shard-Download, blockierter Download → Spielzeugmodell mit Hinweis.
 */
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const withModels = args.includes('--models');
const base = (args.find((a) => !a.startsWith('--')) ?? 'http://localhost:4173/').replace(/\/?$/, '/');
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
/** Abbildung über ihre Nummer in der Bildunterschrift (nicht über beliebigen Text darin) */
const figure = (num, pg = page) => pg.locator('figure.figure').filter({ has: pg.locator('.figure__num', { hasText: new RegExp(`^Abb\\. ${num.replace('.', '\\.')}$`) }) });
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
  const fig = figure('7.2');
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
  const out = async () => {
    await page.waitForTimeout(50);
    await page.waitForSelector('.play-out[data-done="1"]', { timeout: 10000 });
    return page.locator('.play-out').innerText();
  };
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
  const fig = figure('8.1');
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

/* ── Echtes Modell (nur mit --models) ─────────────────────────────── */

const LOAD_TIMEOUT = 180000;
async function selectAndLoad(fig, radioName, key) {
  await fig.getByRole('radio', { name: radioName }).click();
  await fig.locator(`[data-load="${key}"]`).click();
  await page.waitForSelector(`.modelbar[data-model-status="ready"]`, { timeout: LOAD_TIMEOUT });
}
const isFake = async () => (await page.locator('.modelbar__ready').first().innerText()).includes('Zufallsgewichte');
async function waitFresh(fig) {
  await page.waitForTimeout(REAL_DEBOUNCE + 50);
  await fig.locator('.calc-state[data-fresh="1"]').waitFor({ timeout: 60000 });
}
const REAL_DEBOUNCE = 250;
const barTokens = (fig) => fig.locator('.dist__bars .bar__tok').allInnerTexts();

if (withModels) {
  await page.goto(base + '#/kapitel/next-token', { waitUntil: 'networkidle' });
  const fig7 = figure('7.2');

  await check('Modell: Laden zeigt Fortschritt, danach rechnet Kap. 7 echt mit Stempel „Echt · int8“', async () => {
    await fig7.getByRole('radio', { name: 'Echt · Base' }).click();
    const sawProgress = page
      .waitForFunction(() => /MB von [\d.]+ MB/.test(document.querySelector('.modelbar__loading')?.textContent ?? ''), null, { timeout: LOAD_TIMEOUT })
      .then(() => true, () => false);
    await fig7.locator('[data-load="smollm2-135m/int8"]').click();
    assert(await sawProgress, 'kein Fortschritt in MB sichtbar');
    await page.waitForSelector('.modelbar[data-model-status="ready"]', { timeout: LOAD_TIMEOUT });
    await fig7.locator('.dist__bars[data-model="smollm2-135m/int8"]').waitFor({ timeout: 60000 });
    const stamp = await fig7.locator('.figure__caption .stamp').getAttribute('data-stamp');
    assert(stamp === 'real-q8', `Stempel ist „${stamp}“`);
  });

  await check('Kap. 7: Leitbeispiel live nachgerechnet, Abweichung innerhalb der int8-Toleranz', async () => {
    const fig1 = figure('7.1');
    await fig1.locator('[data-leit-live]').click();
    const t = await fig1.locator('[data-live-maxdiff]').innerText({ timeout: 60000 });
    const pp = Number(t.replace(/[^\d,]/g, '').replace(',', '.'));
    if (!(await isFake())) assert(pp < 3.5, `Abweichung ${pp} pp ≥ 3,5 pp`);
  });

  await check('Neueste Anfrage gewinnt: schnelle Eingabefolge zeigt das Ergebnis der letzten', async () => {
    const box = fig7.locator('textarea');
    const [A, B, C, D] = ['The weather today is', 'My favorite animal is the', 'The best way to learn is', 'Water boils at'];
    await box.fill(B);
    await waitFresh(fig7);
    const onlyB = await barTokens(fig7);
    // A → B → C schnell hintereinander (A und B werden noch abgeschickt, C kommt während der Rechnung)
    await box.fill(A);
    await page.waitForTimeout(REAL_DEBOUNCE + 30);
    await box.fill(B);
    await page.waitForTimeout(REAL_DEBOUNCE + 30);
    await box.fill(C);
    await waitFresh(fig7);
    const shown = await barTokens(fig7);
    // Kontrolle: C allein (nach einem Umweg über D)
    await box.fill(D);
    await waitFresh(fig7);
    await box.fill(C);
    await waitFresh(fig7);
    const control = await barTokens(fig7);
    assert(shown.join('|') === control.join('|'), `angezeigt ≠ Kontrolle:\n  ${shown.slice(0, 4)}\n  ${control.slice(0, 4)}`);
    assert(shown.join('|') !== onlyB.join('|'), 'Ergebnis für C gleicht dem für B – Test nicht aussagekräftig');
  });

  await check('Kap. 8: Schritte und Erzwingen mit dem echten Modell', async () => {
    await go('#/kapitel/generierung', '.gen-line');
    const fig = figure('8.1');
    const stamp = await fig.locator('.figure__caption .stamp').getAttribute('data-stamp');
    assert(stamp === 'real-q8', `Stempel ist „${stamp}“`);
    const stepBtn = fig.getByRole('button', { name: 'Ein Schritt' });
    for (let i = 0; i < 3; i++) {
      await stepBtn.click({ timeout: 60000 });
      await page.waitForFunction(() => [...document.querySelectorAll('button')].some((b) => b.textContent === 'Ein Schritt' && !b.disabled) || document.querySelector('.gen-line')?.textContent?.includes('Ende'), null, { timeout: 60000 });
    }
    assert((await fig.locator('.gen-tok').count()) >= 1, 'keine Tokens erzeugt');
    await fig.locator('.gen-tok').first().click();
    await fig.locator('.alt:not(.alt--chosen)').first().click({ timeout: 60000 });
    assert((await fig.locator('.tree__box.is-forced').count()) === 1, 'kein erzwungener Ast');
  });

  await check('Playground: Instruct-Chat antwortet und stoppt am End-Token; nur ein Modell im Worker', async () => {
    await go('#/playground', '.play-out');
    const fig = page.locator('figure.figure').first();
    await selectAndLoad(fig, 'Echt · Instruct', 'smollm2-135m-instruct/int8');
    const stamp = await fig.locator('.figure__caption .stamp').getAttribute('data-stamp');
    assert(stamp === 'real-q8', `Stempel ist „${stamp}“`);
    // Temperatur 0 → deterministische Antwort
    await setRange(fig.locator('input[type=range]').first(), 0);
    await fig.getByLabel('Nachricht').fill('What is the capital of France?');
    await fig.getByRole('button', { name: 'Senden' }).click();
    await page.waitForSelector('.chat[data-running="0"][data-last-stop]:not([data-last-stop=""])', { timeout: 120000 });
    const answer = await fig.locator('.chat__msg--assistant').last().innerText();
    const stop = await fig.locator('.chat').getAttribute('data-last-stop');
    if (!(await isFake())) {
      assert(stop === 'eos', `Antwort endete nicht am End-Token (${stop}): ${answer}`);
      assert(/Paris/.test(answer), `unerwartete Antwort: ${answer}`);
    }
    // nur ein echtes Modell geladen: Base wurde freigegeben
    await go('#/diagnose', '.checks');
    const row = page.locator('.check', { hasText: 'Geladenes Modell' });
    await row.getByRole('button').click();
    await row.locator('p', { hasText: 'geladen' }).waitFor({ timeout: 60000 });
    const txt = await row.innerText();
    assert(/Instruct/.test(txt) && !/Base/.test(txt), `Diagnose zeigt: ${txt.split('\n').slice(0, 2).join(' ')}`);
    assert(/Gewichte im Worker: [\d.]+ MB/.test(txt), 'Speicherangabe fehlt');
  });

  await check('Zweiter Besuch: Modell kommt aus dem Cache Storage, kein Shard-Download', async () => {
    const shardRequests = [];
    const onReq = (r) => /\/models\/.*\.bin$/.test(r.url()) && shardRequests.push(r.url());
    page.on('request', onReq);
    await page.goto(base + '#/kapitel/next-token', { waitUntil: 'networkidle' });
    const fig = figure('7.2');
    await fig.getByRole('radio', { name: 'Echt · Base' }).click();
    await fig.getByText('bereits gespeichert, kein Download').first().waitFor({ timeout: 30000 });
    await fig.locator('[data-load="smollm2-135m/int8"]').click();
    await page.waitForSelector('.modelbar[data-model-status="ready"]', { timeout: LOAD_TIMEOUT });
    page.off('request', onReq);
    assert(shardRequests.length === 0, `${shardRequests.length} Shard-Anfragen trotz Cache`);
    assert((await fig.locator('.modelbar__ready').innerText()).includes('aus dem Speicher des Browsers'), 'Hinweis „aus dem Speicher“ fehlt');
  });

  await check('Blockierter Download: Hinweis, Seite rechnet mit dem Spielzeugmodell weiter', async () => {
    const ctx2 = await browser.newContext({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' });
    const p2 = await ctx2.newPage();
    await p2.route('**/models/**', (r) => r.abort());
    await p2.goto(base + '#/kapitel/next-token', { waitUntil: 'networkidle' });
    const fig = figure('7.2', p2);
    await fig.getByRole('radio', { name: 'Echt · Base' }).click();
    await fig.locator('[data-load="smollm2-135m/int8"]').click();
    await fig.locator('.modelbar__error').waitFor({ timeout: 30000 });
    await fig.locator('.dist__bars[data-model="toy"]').waitFor({ timeout: 10000 });
    const stamp = await fig.locator('.figure__caption .stamp').getAttribute('data-stamp');
    assert(stamp === 'toy', `Stempel ist „${stamp}“`);
    const top = () => fig.locator('.dist__bars .bar').first().locator('.bar__p').innerText();
    const before = await top();
    await fig.locator('input[type=range]').first().evaluate((el) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '0.3');
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await p2.waitForTimeout(150);
    assert(before !== (await top()), 'Regler wirkt nach dem Rückfall nicht');
    await ctx2.close();
  });
}

await browser.close();
for (const [s, n] of results) console.log(`${s} ${n}`);
const failed = results.filter(([s]) => s === '✗').length;
const relevantErrors = errors.filter((e) => !/net::ERR_INTERNET_DISCONNECTED|net::ERR_FAILED|Failed to fetch/.test(e));
if (relevantErrors.length) console.log(`Konsolenfehler:\n  ${relevantErrors.join('\n  ')}`);
process.exit(failed || relevantErrors.length ? 1 : 0);
