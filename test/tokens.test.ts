import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ALL_TOKENS } from '../src/lib/palette';

/** PLAN.md 6.2, Prüfregel 1: Jeder Farbwert gehört zu genau einer Rolle (je Theme). */

const css = readFileSync(new URL('../src/styles/tokens.css', import.meta.url), 'utf8');

function block(selector: string): Record<string, string> {
  const start = css.indexOf(selector);
  if (start < 0) throw new Error(`Selektor ${selector} fehlt`);
  const body = css.slice(css.indexOf('{', start) + 1, css.indexOf('}', start));
  const out: Record<string, string> = {};
  for (const m of body.matchAll(/(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{3,8})\s*;/g)) out[m[1]] = m[2].toLowerCase();
  return out;
}

const themes = {
  light: block(":root[data-theme='light']"),
  dark: block(":root[data-theme='dark']"),
};

describe('Farbtokens', () => {
  for (const [name, tokens] of Object.entries(themes)) {
    it(`${name}: alle Rollen-Tokens sind definiert`, () => {
      for (const t of ALL_TOKENS) expect(tokens, t).toHaveProperty(t);
    });

    it(`${name}: kein Farbwert ist zwei Rollen zugeordnet`, () => {
      const byValue = new Map<string, string[]>();
      for (const [k, v] of Object.entries(tokens)) byValue.set(v, [...(byValue.get(v) ?? []), k]);
      const dupes = [...byValue.entries()].filter(([, ks]) => ks.length > 1);
      expect(dupes).toEqual([]);
    });
  }
});
