import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';

const chapterDir = new URL('../content/chapters/', import.meta.url);
const files = readdirSync(chapterDir).filter((f) => f.endsWith('.mdx')).sort();
const chapters = files.map((f) => {
  const src = readFileSync(new URL(f, chapterDir), 'utf8');
  const fm = /^---\n([\s\S]*?)\n---/.exec(src);
  return { file: f, src, meta: (fm ? parse(fm[1]) : {}) as Record<string, unknown> };
});
const glossary = parse(readFileSync(new URL('../content/glossary.yaml', import.meta.url), 'utf8')) as {
  id: string;
  term: string;
  short: string;
  chapter?: string;
  see?: string[];
}[];
const glossaryIds = new Set(glossary.map((g) => g.id));
const slugs = new Set(chapters.map((c) => c.meta.slug));

describe('Kapitel', () => {
  it('16 Kapitel, durchnummeriert 0–15', () => {
    expect(chapters.map((c) => c.meta.number)).toEqual([...Array(16).keys()]);
  });

  it('Pflichtfelder im Frontmatter', () => {
    for (const c of chapters) {
      for (const k of ['number', 'slug', 'title', 'short', 'minutes', 'phase', 'ready', 'goals']) {
        expect(c.meta, `${c.file}: ${k}`).toHaveProperty(k);
      }
      expect(Array.isArray(c.meta.goals) && (c.meta.goals as unknown[]).length > 0, c.file).toBe(true);
    }
  });

  it('Slugs sind eindeutig und URL-tauglich', () => {
    expect(slugs.size).toBe(chapters.length);
    for (const s of slugs) expect(String(s)).toMatch(/^[a-z0-9-]+$/);
  });

  it('Dateiname passt zu Nummer und Slug', () => {
    for (const c of chapters) expect(c.file).toBe(`${String(c.meta.number).padStart(2, '0')}-${c.meta.slug}.mdx`);
  });
});

describe('Glossar', () => {
  it('IDs sind eindeutig', () => {
    expect(glossaryIds.size).toBe(glossary.length);
  });

  it('jeder Eintrag hat Begriff und Kurzerklärung', () => {
    for (const g of glossary) {
      expect(g.term, g.id).toBeTruthy();
      expect(g.short, g.id).toBeTruthy();
    }
  });

  it('Verweise auf Kapitel und andere Begriffe sind gültig', () => {
    for (const g of glossary) {
      if (g.chapter) expect(slugs.has(g.chapter), `${g.id} → ${g.chapter}`).toBe(true);
      for (const s of g.see ?? []) expect(glossaryIds.has(s), `${g.id} → ${s}`).toBe(true);
    }
  });

  it('keine hart eingetragenen Prozentwerte des Spielzeugmodells im Text (stattdessen <Fact m="toy" …>)', () => {
    const offending: string[] = [];
    for (const c of chapters) {
      c.src.split(/\n\s*\n/).forEach((para) => {
        if (/spielzeug|toy/i.test(para) && /\d+(,\d+)?\s?%/.test(para.replace(/<Fact[^>]*>/g, ''))) offending.push(`${c.file}: ${para.slice(0, 60)}…`);
      });
    }
    expect(offending).toEqual([]);
  });

  it('jedes <Term id="…"> in den Kapiteln existiert im Glossar', () => {
    const missing: string[] = [];
    for (const c of chapters) {
      for (const m of c.src.matchAll(/<Term\s+id="([^"]+)"/g)) if (!glossaryIds.has(m[1])) missing.push(`${c.file}: ${m[1]}`);
    }
    expect(missing).toEqual([]);
  });
});
