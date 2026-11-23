import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';

/** Jede Lizenzdatei muss auf #/credits verlinkt sein – und jeder Link muss eine Datei haben. */

const dir = new URL('../public/licenses/', import.meta.url);
const groups = parse(readFileSync(new URL('../content/credits.yaml', import.meta.url), 'utf8')) as {
  group: string;
  items: { name: string; license: string; licenseFile?: string }[];
}[];
const linked = groups.flatMap((g) => g.items.map((i) => i.licenseFile).filter(Boolean)) as string[];

describe('Credits & Lizenzen', () => {
  it('jede Datei in public/licenses/ ist in content/credits.yaml verlinkt', () => {
    const files = readdirSync(dir).filter((f) => !f.startsWith('.'));
    const missing = files.filter((f) => !linked.includes(f));
    expect(missing).toEqual([]);
  });

  it('jede verlinkte Lizenzdatei existiert', () => {
    const missing = linked.filter((f) => !existsSync(new URL(f, dir)));
    expect(missing).toEqual([]);
  });

  it('jeder Eintrag nennt eine Lizenz', () => {
    for (const g of groups) for (const i of g.items) expect(i.license, i.name).toBeTruthy();
  });
});
