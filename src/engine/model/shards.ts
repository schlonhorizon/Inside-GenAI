import { sha256Hex, type ModelManifest } from './pack.ts';

/**
 * Lädt die Shards eines Modells im Browser (läuft im Worker):
 *   • Manifest holen (immer vom Server, damit neue Versionen erkannt werden)
 *   • je Shard zuerst im Cache Storage nachsehen, sonst als Stream laden (Fortschritt in Bytes)
 *   • Länge und SHA-256 jedes geladenen Shards prüfen, dann im Cache Storage ablegen
 *   • alte Versionen desselben Modells aus dem Cache Storage löschen
 * Ergebnis ist ein zusammenhängender Puffer, aus dem `unpackWeights` die Gewichte baut.
 */

export interface LoadProgress {
  loaded: number;
  total: number;
  /** Bytes, die aus dem Cache Storage kamen (kein Download) */
  fromCache: number;
  phase: 'manifest' | 'download' | 'prepare';
}

export const CACHE_PREFIX = 'igai-model-';

export const cacheName = (m: Pick<ModelManifest, 'model' | 'edition' | 'version'>) => `${CACHE_PREFIX}${m.model}-${m.edition}-${m.version}`;

function storage(): CacheStorage | null {
  try {
    return typeof caches !== 'undefined' ? caches : null;
  } catch {
    return null;
  }
}

export class ModelLoadError extends Error {
  constructor(
    message: string,
    readonly code: 'missing' | 'network' | 'integrity' | 'aborted',
  ) {
    super(message);
    // Der Name überlebt den Weg über Comlink (Fehler werden als {name, message} übertragen)
    this.name = `ModelLoadError:${code}`;
  }
}

export async function fetchManifest(url: string, signal?: AbortSignal): Promise<ModelManifest> {
  let r: Response;
  try {
    r = await fetch(url, { cache: 'no-cache', signal });
  } catch (e) {
    if (signal?.aborted) throw new ModelLoadError('Abgebrochen', 'aborted');
    throw new ModelLoadError(`Modell nicht erreichbar (${(e as Error).message})`, 'network');
  }
  if (r.status === 404) throw new ModelLoadError('Dieses Modell ist in dieser Fassung der Seite nicht enthalten.', 'missing');
  if (!r.ok) throw new ModelLoadError(`Modell nicht erreichbar (HTTP ${r.status})`, 'network');
  return (await r.json()) as ModelManifest;
}

/** Wie viele Bytes dieses Modells liegen schon im Cache Storage? */
export async function cachedBytes(manifest: ModelManifest, base: string): Promise<number> {
  const cs = storage();
  if (!cs) return 0;
  const cache = await cs.open(cacheName(manifest));
  let n = 0;
  for (const s of manifest.shards) if (await cache.match(new URL(s.file, base).href)) n += s.bytes;
  return n;
}

export async function fetchShards(
  manifest: ModelManifest,
  manifestUrl: string,
  onProgress: (p: LoadProgress) => void,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const buf = new ArrayBuffer(manifest.totalBytes);
  const total = manifest.totalBytes;
  let loaded = 0;
  let fromCache = 0;
  const report = (phase: LoadProgress['phase'] = 'download') => onProgress({ loaded, total, fromCache, phase });
  const cs = storage();
  const cache = cs ? await cs.open(cacheName(manifest)).catch(() => null) : null;

  // alte Versionen desselben Modells aufräumen
  if (cs) {
    const prefix = `${CACHE_PREFIX}${manifest.model}-${manifest.edition}-`;
    for (const k of await cs.keys()) if (k.startsWith(prefix) && k !== cacheName(manifest)) await cs.delete(k);
  }

  for (const shard of manifest.shards) {
    if (signal?.aborted) throw new ModelLoadError('Abgebrochen', 'aborted');
    const url = new URL(shard.file, manifestUrl).href;
    const target = new Uint8Array(buf, shard.offset, shard.bytes);

    const hit = cache ? await cache.match(url) : undefined;
    if (hit) {
      const data = new Uint8Array(await hit.arrayBuffer());
      if (data.byteLength === shard.bytes) {
        target.set(data);
        loaded += shard.bytes;
        fromCache += shard.bytes;
        report();
        continue;
      }
      await cache!.delete(url); // beschädigt → neu laden
    }

    let r: Response;
    try {
      r = await fetch(url, { signal });
    } catch (e) {
      if (signal?.aborted) throw new ModelLoadError('Abgebrochen', 'aborted');
      throw new ModelLoadError(`Download fehlgeschlagen (${(e as Error).message})`, 'network');
    }
    if (!r.ok || !r.body) throw new ModelLoadError(`Download fehlgeschlagen (HTTP ${r.status})`, 'network');
    const reader = r.body.getReader();
    let pos = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (pos + value.byteLength > shard.bytes) throw new ModelLoadError(`${shard.file}: mehr Daten als erwartet`, 'integrity');
        target.set(value, pos);
        pos += value.byteLength;
        loaded += value.byteLength;
        report();
      }
    } catch (e) {
      if (signal?.aborted) throw new ModelLoadError('Abgebrochen', 'aborted');
      if (e instanceof ModelLoadError) throw e;
      throw new ModelLoadError(`Download abgebrochen (${(e as Error).message})`, 'network');
    }
    if (pos !== shard.bytes) throw new ModelLoadError(`${shard.file}: unvollständig (${pos} von ${shard.bytes} Bytes)`, 'integrity');
    if ((await sha256Hex(target)) !== shard.sha256) throw new ModelLoadError(`${shard.file}: Prüfsumme falsch`, 'integrity');
    if (cache) {
      try {
        await cache.put(url, new Response(target.slice()));
      } catch {
        // Speicher voll o. Ä.: Modell funktioniert trotzdem, nur ohne Cache
      }
    }
  }
  report('prepare');
  return buf;
}
