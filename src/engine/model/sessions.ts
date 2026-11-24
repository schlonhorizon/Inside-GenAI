import type { KvCache, LlamaModel } from './llama.ts';

/**
 * KV-Caches pro Sitzung (eine Sitzung = ein Widget) mit Wiederverwendung des gemeinsamen Anfangs.
 *
 * Bei jeder Anfrage wird das längste gemeinsame Präfix zwischen der zuletzt gerechneten Folge
 * dieser Sitzung und der neuen Folge bestimmt. Der Cache wird darauf gekürzt, nur der Rest läuft
 * durch das Modell. Das deckt Verlängerung (Generieren), Abzweigung (Token erzwingen im
 * Antwort-Baum), Verkürzung (Zurück) und identische Folgen ab.
 *
 * Höchstens `capacity` Sitzungen behalten ihren Cache (LRU). Eine verdrängte Sitzung rechnet
 * beim nächsten Aufruf einfach komplett neu.
 */

interface Session {
  cache: KvCache;
  ids: number[];
  logits: Float32Array | null;
}

export interface SessionStats {
  /** Tokens, die bei der letzten Anfrage tatsächlich gerechnet wurden */
  computed: number;
  /** Tokens, die aus dem Cache kamen */
  reused: number;
}

export function commonPrefix(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

export class KvSessions {
  private sessions = new Map<string, Session>();
  readonly model: LlamaModel;
  readonly capacity: number;
  readonly maxSeq: number;
  last: SessionStats = { computed: 0, reused: 0 };

  constructor(model: LlamaModel, capacity = 3, maxSeq = model.maxSeq) {
    this.model = model;
    this.capacity = capacity;
    this.maxSeq = Math.min(maxSeq, model.maxSeq);
  }

  get size() {
    return this.sessions.size;
  }

  has(id: string) {
    return this.sessions.has(id);
  }

  /** Arbeitsspeicher aller Sitzungs-Caches in Bytes */
  get bytes() {
    let n = 0;
    for (const s of this.sessions.values()) n += s.cache.bytes;
    return n;
  }

  private session(id: string): Session {
    let s = this.sessions.get(id);
    if (s) {
      // als zuletzt benutzt markieren
      this.sessions.delete(id);
      this.sessions.set(id, s);
      return s;
    }
    let cache: KvCache;
    if (this.sessions.size >= this.capacity) {
      // ältesten Eintrag verdrängen und seinen Speicher weiterverwenden
      const [oldest, victim] = this.sessions.entries().next().value as [string, Session];
      this.sessions.delete(oldest);
      cache = victim.cache;
      cache.length = 0;
    } else {
      cache = this.model.newCache(this.maxSeq);
    }
    s = { cache, ids: [], logits: null };
    this.sessions.set(id, s);
    return s;
  }

  /** Logits für das nächste Token nach `ids` (Kopie). */
  logits(id: string, ids: number[]): Float32Array {
    if (!ids.length) throw new Error('Leere Tokenfolge');
    if (ids.length > this.maxSeq) throw new Error(`Kontext zu lang (${ids.length} > ${this.maxSeq} Tokens)`);
    const s = this.session(id);
    const lcp = commonPrefix(s.ids, ids);
    if (lcp === ids.length && lcp === s.ids.length && s.logits) {
      this.last = { computed: 0, reused: ids.length };
      return s.logits.slice();
    }
    // mindestens das letzte Token neu rechnen, sonst gibt es keine Logits
    const keep = Math.min(lcp, ids.length - 1, s.cache.length);
    this.model.useCache(s.cache);
    this.model.truncate(keep);
    let out: Float32Array;
    try {
      out = this.model.prefillBatch(ids.slice(keep));
    } catch (e) {
      s.cache.length = 0;
      s.ids = [];
      s.logits = null;
      throw e;
    }
    s.ids = ids.slice();
    s.logits = out.slice();
    this.last = { computed: ids.length - keep, reused: keep };
    return out.slice();
  }

  drop(id: string) {
    this.sessions.delete(id);
  }

  clear() {
    this.sessions.clear();
  }
}
