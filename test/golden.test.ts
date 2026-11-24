import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { hasLocalModel, loadLocalModel } from '../scripts/local-model';
import { KvSessions } from '../src/engine/model/sessions';
import { BpeTokenizer } from '../src/engine/tokenizer/bpe';
import { renderChat, type ChatMessage } from '../src/engine/chat';
import { argmax } from '../src/engine/model/llama';
import { softmax } from '../src/engine/sampling';

/**
 * Golden-Tests: die eigene Engine mit den AUSGELIEFERTEN Shards gegen PyTorch (fp32).
 * Referenz: test/fixtures/golden/ (assets.yml, Ziel „golden“). Shards: .cache/models
 * (scripts/prepare-models.ts, im CI aus dem Cache). Toleranzen aus docs/spike.md.
 *
 * Lokal ohne Shards werden die Tests sichtbar übersprungen. Im CI (REQUIRE_MODELS=1) ist das
 * verboten: Fehlen Shards oder Referenz, schlägt der Lauf fehl.
 */

const REQUIRE = process.env.REQUIRE_MODELS === '1';
const GOLDEN_DIR = new URL('./fixtures/golden/', import.meta.url);
const goldenPath = new URL('golden.json', GOLDEN_DIR);

interface GoldenPrompt {
  id: string;
  model: 'base' | 'instruct';
  text?: string;
  messages?: ChatMessage[];
  rendered?: string;
  ids: number[];
  greedy_ids: number[];
}
interface Golden {
  models: Record<'base' | 'instruct', { repo: string; revision: string }>;
  greedy_steps: number;
  prompts: GoldenPrompt[];
}

const EDITIONS = [
  { model: 'smollm2-135m', key: 'base', edition: 'int8', exact: false },
  { model: 'smollm2-135m', key: 'base', edition: 'fp16', exact: true },
  { model: 'smollm2-135m-instruct', key: 'instruct', edition: 'int8', exact: false },
] as const;

/** Toleranzen (docs/spike.md, „Festgelegte Toleranzen für die Golden-Tests“) */
const TOL = {
  exact: { relL2: 1e-4, kl: 1e-6, greedy: 1 },
  int8: { top10Overlap: 8, kl: 4e-3, dp: 0.035 },
};

function compare(ours: Float32Array, ref: Float32Array) {
  const p = softmax(ref);
  const q = softmax(ours);
  let kl = 0;
  let dp = 0;
  let num = 0;
  let den = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) kl += p[i] * Math.log(p[i] / Math.max(q[i], 1e-30));
    dp = Math.max(dp, Math.abs(p[i] - q[i]));
    num += (ours[i] - ref[i]) ** 2;
    den += ref[i] ** 2;
  }
  const top = (x: Float32Array) => Array.from(x.keys()).sort((a, b) => x[b] - x[a]).slice(0, 10);
  const tr = top(ref);
  const to = top(ours);
  return { kl, dp, relL2: Math.sqrt(num / den), top1: tr[0] === to[0], topSame: tr.join() === to.join(), overlap: to.filter((t) => tr.includes(t)).length };
}

const haveGolden = existsSync(goldenPath);
const haveShards = EDITIONS.every((e) => hasLocalModel(e.model, e.edition));

describe('Golden-Voraussetzungen', () => {
  it(REQUIRE ? 'Referenz und Shards sind vorhanden (REQUIRE_MODELS=1)' : 'Referenz und Shards (lokal optional)', () => {
    if (!REQUIRE) {
      if (!haveGolden || !haveShards) console.warn('[golden] übersprungen: Referenz oder Shards fehlen (lokal erlaubt, im CI nicht)');
      return;
    }
    expect(haveGolden, 'test/fixtures/golden/golden.json fehlt').toBe(true);
    expect(haveShards, 'Shards in .cache/models fehlen').toBe(true);
  });
});

describe.runIf(haveGolden && haveShards)('Golden-Tests gegen PyTorch', () => {
  const golden = haveGolden ? (JSON.parse(readFileSync(goldenPath, 'utf8')) as Golden) : null!;
  const tokenizer = new BpeTokenizer(JSON.parse(readFileSync(new URL('../public/tokenizers/smollm2.json', import.meta.url), 'utf8')));
  const refLogits = (id: string) => {
    const b = readFileSync(new URL(`${id}.f32`, GOLDEN_DIR));
    return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  };

  it('Tokenisierung und Chat-Template stimmen mit Python überein', () => {
    for (const g of golden.prompts) {
      if (g.model === 'base') expect(tokenizer.encodeIds(g.text!), g.id).toEqual(g.ids);
      else {
        expect(renderChat(g.messages!), g.id).toBe(g.rendered);
        expect(tokenizer.encodeIds(g.rendered!), g.id).toEqual(g.ids);
      }
    }
  });

  for (const e of EDITIONS) {
    it(
      `${e.model}/${e.edition}: ${e.exact ? 'exakt' : 'int8-Toleranz'} bei allen Prompts`,
      () => {
        const { manifest, model } = loadLocalModel(e.model, e.edition);
        expect(manifest.revision, 'Shards und Referenz müssen dieselbe Revision haben').toBe(golden.models[e.key].revision);
        const sessions = new KvSessions(model, 1);
        // erst alle Prompts messen und die Tabelle ins Log schreiben, dann prüfen –
        // so stehen bei einem Fehlschlag alle Werte im Log, nicht nur der erste Verstoß
        const results = golden.prompts
          .filter((p) => p.model === e.key)
          .map((g) => {
            const ours = sessions.logits(g.id, g.ids);
            const c = compare(ours, refLogits(g.id));
            let greedy: number[] | null = null;
            if (e.exact) {
              // Greedy-Fortsetzung über den KV-Cache der Sitzung (Verlängerung um je ein Token)
              const seq = g.ids.slice();
              greedy = [];
              let logits = ours;
              for (let i = 0; i < golden.greedy_steps; i++) {
                const t = argmax(logits);
                greedy.push(t);
                seq.push(t);
                logits = sessions.logits(g.id, seq);
              }
            }
            return { g, c, greedy };
          });
        const rows = results.map(
          ({ g, c, greedy }) =>
            `${g.id.padEnd(14)} top1 ${c.top1 ? '✓' : '✗'} top10 ${c.overlap}/10 KL ${c.kl.toExponential(2)} Δp ${(c.dp * 100).toFixed(2)} pp relL2 ${c.relL2.toExponential(2)}` +
            (greedy ? ` greedy ${greedy.filter((t, i) => t === g.greedy_ids[i]).length}/${g.greedy_ids.length}` : ''),
        );
        console.log(`[golden] ${e.model}/${e.edition}\n  ${rows.join('\n  ')}`);
        for (const { g, c, greedy } of results) {
          if (e.exact) {
            expect.soft(c.topSame, `${g.id}: Top-10 identisch`).toBe(true);
            expect.soft(c.relL2, `${g.id}: rel. L2`).toBeLessThan(TOL.exact.relL2);
            expect.soft(c.kl, `${g.id}: KL`).toBeLessThan(TOL.exact.kl);
            expect.soft(greedy, `${g.id}: Greedy ${golden.greedy_steps}/${golden.greedy_steps}`).toEqual(g.greedy_ids);
          } else {
            expect.soft(c.top1, `${g.id}: Top-1 identisch`).toBe(true);
            expect.soft(c.overlap, `${g.id}: Top-10-Überlappung`).toBeGreaterThanOrEqual(TOL.int8.top10Overlap);
            expect.soft(c.kl, `${g.id}: KL`).toBeLessThan(TOL.int8.kl);
            expect.soft(c.dp, `${g.id}: max. Δp`).toBeLessThan(TOL.int8.dp);
          }
        }
      },
      300_000,
    );
  }
});
