import { useEffect, useState } from 'react';

export type ThemePref = 'system' | 'light' | 'dark';
export type Theme = 'light' | 'dark';

const KEY = 'igai.theme';

export function readStored<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(key) as T | null) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeStored(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* Speicher nicht verfügbar – Einstellung gilt nur für diese Sitzung */
  }
}

const media = () => window.matchMedia('(prefers-color-scheme: dark)');

function resolve(pref: ThemePref): Theme {
  if (pref === 'system') return media().matches ? 'dark' : 'light';
  return pref;
}

/** Theme-Präferenz (System/Hell/Dunkel) inkl. Live-Reaktion auf die Systemeinstellung. */
export function useTheme() {
  const [pref, setPref] = useState<ThemePref>(() => readStored<ThemePref>(KEY, 'system'));
  const [theme, setTheme] = useState<Theme>(() => resolve(pref));

  useEffect(() => {
    const apply = () => {
      const t = resolve(pref);
      setTheme(t);
      document.documentElement.setAttribute('data-theme', t);
    };
    apply();
    writeStored(KEY, pref);
    const m = media();
    m.addEventListener('change', apply);
    return () => m.removeEventListener('change', apply);
  }, [pref]);

  return { pref, setPref, theme };
}

/** Liest die aktuellen Werte von CSS-Custom-Properties (nach jedem Theme-Wechsel neu). */
export function readTokens(names: string[]): Record<string, string> {
  const cs = getComputedStyle(document.documentElement);
  const out: Record<string, string> = {};
  for (const n of names) out[n] = cs.getPropertyValue(n).trim().toUpperCase();
  return out;
}
