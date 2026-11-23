import raw from '../../content/glossary.yaml';

export interface GlossaryEntry {
  id: string;
  term: string;
  /** englischer Fachbegriff bzw. deutsche Übersetzung, falls abweichend */
  alt?: string;
  short: string;
  long?: string;
  /** Kapitel-Slug, in dem der Begriff erklärt wird */
  chapter?: string;
  see?: string[];
}

export const GLOSSARY: GlossaryEntry[] = (raw as GlossaryEntry[]).slice().sort((a, b) => a.term.localeCompare(b.term, 'de'));

const byId = new Map(GLOSSARY.map((e) => [e.id, e]));
export const getTerm = (id: string) => byId.get(id);
