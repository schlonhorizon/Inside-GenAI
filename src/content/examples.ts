import raw from '../../content/examples.yaml';

export interface Examples {
  leitbeispiel: {
    prompt: string;
    model: string;
    modelLabel: string;
    revision: string;
    precision: string;
    source: string;
    greedy: string;
    instructAnswer: string;
    top: { token: string; p: number }[];
  };
  toyPrompts: string[];
  toyVocabGroups: { name: string; words: string[] }[];
  languagePairs: { de: string; en: string }[];
  bpeCorpus: string;
  letterCount: { word: string; letter: string };
}

export const EXAMPLES = raw as Examples;
