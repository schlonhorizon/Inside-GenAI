import type { ComponentType } from 'react';
import metas from 'virtual:chapter-meta';

export interface ChapterMeta {
  number: number;
  slug: string;
  title: string;
  /** Untertitel / Leitfrage */
  short: string;
  minutes: number;
  /** Phase, in der das Kapitel mit Inhalt gefüllt wird (z. B. "P1b") */
  phase: string;
  /** true = Inhalt fertig; false = Platzhalter */
  ready: boolean;
  goals: string[];
}


type MdxModule = { default: ComponentType; frontmatter: ChapterMeta };

// Frontmatter sofort (für Navigation, über vite.config.ts), Inhalt erst bei Bedarf (ein Chunk pro Kapitel).
const loaders = import.meta.glob<MdxModule>('/content/chapters/*.mdx');

export const CHAPTERS: ChapterMeta[] = (metas as (ChapterMeta & { file: string })[])
  .slice()
  .sort((a, b) => a.number - b.number);

const loaderBySlug = new Map<string, () => Promise<MdxModule>>();
for (const m of metas as (ChapterMeta & { file: string })[]) loaderBySlug.set(m.slug, loaders[m.file]);

export function getChapter(slug: string): ChapterMeta | undefined {
  return CHAPTERS.find((c) => c.slug === slug);
}

export function loadChapter(slug: string): Promise<MdxModule> | undefined {
  return loaderBySlug.get(slug)?.();
}

export function neighbours(slug: string): { prev?: ChapterMeta; next?: ChapterMeta } {
  const i = CHAPTERS.findIndex((c) => c.slug === slug);
  return { prev: CHAPTERS[i - 1], next: CHAPTERS[i + 1] };
}

export const pad = (n: number) => String(n).padStart(2, '0');
