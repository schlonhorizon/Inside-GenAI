import { create } from 'zustand';
import { loadTokenizer } from '../engine/tokenizer/load';
import {
  BASE_MODEL,
  EDITIONS,
  INSTRUCT_MODEL,
  RealLanguageModel,
  editionKey,
  findEdition,
  manifestUrl,
  workerClient,
  type LoadProgress,
  type WorkerInfo,
} from '../engine/real';
import * as Comlink from 'comlink';

/**
 * Zustand des echten Modells (flüchtig, PLAN.md 5.2):
 *   source   welches Modell die Widgets zeigen sollen (seitenweit; nur eines kann geladen sein)
 *   loaded   das im Worker geladene echte Modell (höchstens eines)
 * Nach erfolgreichem Laden zeigen die Widgets das echte Modell; schlägt es fehl, geht es mit dem
 * Spielzeugmodell weiter, und ein Hinweis erklärt warum.
 */

export type Source = 'toy' | 'base' | 'instruct';

export interface ModelState {
  source: Source;
  /** Fassung des Base-Modells, die bei „Echt (Base)“ verwendet wird */
  baseEdition: 'int8' | 'fp16';
  status: 'idle' | 'loading' | 'ready' | 'error';
  loadingKey: string | null;
  progress: LoadProgress | null;
  error: { key: string; message: string; code: string } | null;
  model: RealLanguageModel | null;
  info: WorkerInfo | null;
  setSource: (s: Source) => void;
  load: (key: string) => Promise<void>;
  cancel: () => void;
}

export const hasRealModels = EDITIONS.length > 0;

/** Schlüssel der Fassung, die zu einer Quelle gehört (z. B. "smollm2-135m/int8") */
export function keyForSource(source: Source, baseEdition: 'int8' | 'fp16'): string | null {
  if (source === 'base') return `${BASE_MODEL}/${baseEdition}`;
  if (source === 'instruct') return `${INSTRUCT_MODEL}/int8`;
  return null;
}

export const sourceAvailable = (s: Source) =>
  s === 'toy' || EDITIONS.some((e) => e.model === (s === 'base' ? BASE_MODEL : INSTRUCT_MODEL));

const errorCode = (e: unknown) => {
  const name = (e as Error)?.name ?? '';
  return name.startsWith('ModelLoadError:') ? name.slice('ModelLoadError:'.length) : 'other';
};

export const useModelStore = create<ModelState>((set, get) => ({
  source: 'toy',
  baseEdition: 'int8',
  status: 'idle',
  loadingKey: null,
  progress: null,
  error: null,
  model: null,
  info: null,

  setSource: (source) => set({ source }),

  async load(key) {
    const edition = findEdition(key);
    if (!edition) {
      set({ status: 'error', error: { key, message: 'Diese Fassung ist in dieser Seite nicht enthalten.', code: 'missing' } });
      return;
    }
    // höchstens ein echtes Modell: das bisherige wird im Worker freigegeben
    set({ status: 'loading', loadingKey: key, progress: null, error: null, model: null, info: null });
    try {
      const [tokenizer, info] = await Promise.all([
        loadTokenizer('smollm2'),
        workerClient().load(
          key,
          manifestUrl(key),
          Comlink.proxy((p: LoadProgress) => {
            if (get().loadingKey === key) set({ progress: p });
          }),
        ),
      ]);
      if (get().loadingKey !== key) return; // inzwischen anderes Modell angefordert
      const model = new RealLanguageModel(edition, tokenizer);
      const source: Source = edition.model === INSTRUCT_MODEL ? 'instruct' : 'base';
      set({ status: 'ready', loadingKey: null, model, info, source, baseEdition: source === 'base' ? edition.edition : get().baseEdition });
    } catch (e) {
      if (get().loadingKey !== key) return;
      const code = errorCode(e);
      set({
        status: code === 'aborted' ? 'idle' : 'error',
        loadingKey: null,
        progress: null,
        error: code === 'aborted' ? null : { key, message: (e as Error).message, code },
        // Rückfall: Widgets rechnen mit dem Spielzeugmodell weiter
        source: 'toy',
      });
    }
  },

  cancel() {
    if (get().status !== 'loading') return;
    workerClient().cancel();
  },
}));

/** Ist das zur aktuellen Quelle passende echte Modell geladen? Dann dieses, sonst null. */
export function useActiveRealModel(): RealLanguageModel | null {
  return useModelStore((s) => {
    const key = keyForSource(s.source, s.baseEdition);
    return key && s.model && s.model.id === key ? s.model : null;
  });
}

export { editionKey };
