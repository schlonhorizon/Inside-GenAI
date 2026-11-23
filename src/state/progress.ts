import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { safeStorage } from './settings';

export interface ChapterProgress {
  visited: boolean;
  completed: boolean;
  /** bestes Quiz-Ergebnis 0..1 (ab P1b) */
  quiz?: number;
}

interface ProgressState {
  chapters: Record<string, ChapterProgress>;
  lastSlug?: string;
  markVisited: (slug: string) => void;
  setCompleted: (slug: string, completed: boolean) => void;
  reset: () => void;
}

export const useProgress = create<ProgressState>()(
  persist(
    (set) => ({
      chapters: {},
      markVisited: (slug) =>
        set((s) => ({
          lastSlug: slug,
          chapters: { ...s.chapters, [slug]: { completed: s.chapters[slug]?.completed ?? false, quiz: s.chapters[slug]?.quiz, visited: true } },
        })),
      setCompleted: (slug, completed) =>
        set((s) => ({
          chapters: { ...s.chapters, [slug]: { quiz: s.chapters[slug]?.quiz, visited: true, completed } },
        })),
      reset: () => set({ chapters: {}, lastSlug: undefined }),
    }),
    { name: 'igai.progress', version: 1, storage: safeStorage },
  ),
);
