import { useEffect, useState } from 'react';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type ThemePref = 'system' | 'light' | 'dark';
export type Theme = 'light' | 'dark';
export type MotionPref = 'system' | 'reduced' | 'full';

export interface SettingsState {
  theme: ThemePref;
  /** Mathe-Ebene: Formeln und „Unter der Haube“-Details einblenden */
  math: boolean;
  motion: MotionPref;
  setTheme: (t: ThemePref) => void;
  setMath: (m: boolean) => void;
  setMotion: (m: MotionPref) => void;
}

/** Speicherschlüssel – wird auch vom Inline-Skript in index.html gelesen. */
export const SETTINGS_KEY = 'igai.settings';

/** localStorage, das nie wirft (privates Fenster, blockierte Site-Daten). */
export const safeStorage = createJSONStorage(() => ({
  getItem: (k: string) => {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  setItem: (k: string, v: string) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* nur für diese Sitzung */
    }
  },
  removeItem: (k: string) => {
    try {
      localStorage.removeItem(k);
    } catch {
      /* ignorieren */
    }
  },
}));

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'system',
      math: true,
      motion: 'system',
      setTheme: (theme) => set({ theme }),
      setMath: (math) => set({ math }),
      setMotion: (motion) => set({ motion }),
    }),
    {
      name: SETTINGS_KEY,
      version: 1,
      storage: safeStorage,
      partialize: (s) => ({ theme: s.theme, math: s.math, motion: s.motion }),
    },
  ),
);

function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setMatches(m.matches);
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return matches;
}

export function useResolvedTheme(): Theme {
  const pref = useSettings((s) => s.theme);
  const systemDark = useMedia('(prefers-color-scheme: dark)');
  return pref === 'system' ? (systemDark ? 'dark' : 'light') : pref;
}

export function useReducedMotionPref(): boolean {
  const pref = useSettings((s) => s.motion);
  const systemReduced = useMedia('(prefers-reduced-motion: reduce)');
  return pref === 'system' ? systemReduced : pref === 'reduced';
}

/** Spiegelt Theme, Mathe-Ebene und Bewegung als Attribute auf <html>. */
export function useApplySettings() {
  const theme = useResolvedTheme();
  const math = useSettings((s) => s.math);
  const reduced = useReducedMotionPref();
  useEffect(() => {
    const el = document.documentElement;
    el.setAttribute('data-theme', theme);
    el.setAttribute('data-math', math ? 'on' : 'off');
    el.setAttribute('data-motion', reduced ? 'reduced' : 'full');
  }, [theme, math, reduced]);
  return { theme, reduced };
}
