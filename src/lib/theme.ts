/** Liest die aktuellen Werte von CSS-Custom-Properties (nach jedem Theme-Wechsel neu). */
export function readTokens(names: string[]): Record<string, string> {
  const cs = getComputedStyle(document.documentElement);
  const out: Record<string, string> = {};
  for (const n of names) out[n] = cs.getPropertyValue(n).trim().toUpperCase();
  return out;
}
