/** Grenzen für das echte Modell im Browser (Rechenzeit und Arbeitsspeicher). */

/** Höchstens so viele Tokens darf ein Prompt (bzw. ein gerenderter Chat) haben. */
export const REAL_PROMPT_LIMIT = 128;
/** Länge jedes KV-Caches: Prompt plus erzeugte Tokens. */
export const REAL_MAX_SEQ = 256;
/** So viele Widgets behalten gleichzeitig ihren KV-Cache (LRU). */
export const KV_SESSIONS = 3;

/** Arbeitsspeicher der KV-Caches aller Sitzungen in Bytes */
export const kvBytes = (layers: number, kvDim: number, maxSeq = REAL_MAX_SEQ, sessions = KV_SESSIONS) =>
  layers * 2 * kvDim * maxSeq * 4 * sessions;
