import { useCallback, useMemo, useState } from 'react';
import { getFact } from '../content/facts';

/**
 * #/diagnose – zeigt in 30 Sekunden, was auf diesem Gerät / in diesem Netz geht (PLAN.md 4.4).
 */

type Status = 'idle' | 'running' | 'ok' | 'warn' | 'fail';
interface Check {
  status: Status;
  summary: string;
  details?: string[];
}
type CheckId = 'download' | 'cache' | 'webgpu' | 'bench' | 'env' | 'external';

const TESTSHARD_BYTES = 30 * 1024 * 1024; // muss zu scripts/make-testshard.mjs passen
const FALLBACK_PARAMS = 134.5e6; // Schätzwert, bis P2 die echte Config liefert
const MB = 1024 * 1024;

const fmt = (n: number, d = 1) => n.toLocaleString('de-DE', { maximumFractionDigits: d, minimumFractionDigits: d });

async function checkDownload(onProgress: (p: number) => void): Promise<Check> {
  const url = `${import.meta.env.BASE_URL}diagnose/testshard.bin?t=${Date.now()}`;
  const t0 = performance.now();
  let res: Response;
  try {
    res = await fetch(url, { cache: 'no-store' });
  } catch (e) {
    return { status: 'fail', summary: 'Download blockiert oder abgebrochen.', details: [String(e)] };
  }
  if (!res.ok || !res.body) {
    return {
      status: 'fail',
      summary: `Test-Shard nicht verfügbar (HTTP ${res.status}).`,
      details: ['Im Entwicklungsmodus gibt es keinen Test-Shard – nur in der gebauten Seite.'],
    };
  }
  const reader = res.body.getReader();
  let bytes = 0;
  const tFirst = performance.now();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes += value.byteLength;
    onProgress(bytes / TESTSHARD_BYTES);
  }
  const s = (performance.now() - t0) / 1000;
  const mbps = bytes / MB / s;
  const complete = bytes === TESTSHARD_BYTES;
  return {
    status: complete ? (mbps > 2 ? 'ok' : 'warn') : 'fail',
    summary: complete
      ? `${fmt(bytes / MB, 0)} MB in ${fmt(s)} s · ${fmt(mbps)} MB/s`
      : `Unvollständig: ${fmt(bytes / MB)} von ${fmt(TESTSHARD_BYTES / MB, 0)} MB angekommen (Proxy-Abbruch?)`,
    details: [
      `Zeit bis zum ersten Byte: ${fmt((tFirst - t0) / 1000, 2)} s`,
      `Hochgerechnet: ≈ ${fmt(100 / mbps, 0)} s pro 100 MB Modelldaten`,
      `Content-Encoding: ${res.headers.get('content-encoding') ?? 'keins'}`,
    ],
  };
}

async function checkCache(): Promise<Check> {
  const details: string[] = [`Sicherer Kontext (HTTPS): ${window.isSecureContext ? 'ja' : 'nein'}`];
  if (navigator.storage?.estimate) {
    const est = await navigator.storage.estimate();
    details.push(`Speicher: ${fmt((est.usage ?? 0) / MB)} MB belegt von ≈ ${fmt((est.quota ?? 0) / MB / 1024)} GB verfügbar`);
  }
  if (navigator.storage?.persisted) {
    details.push(`Dauerhafter Speicher zugesichert: ${(await navigator.storage.persisted()) ? 'ja' : 'nein'}`);
  }
  if (!('caches' in window)) {
    return { status: 'fail', summary: 'Cache API nicht verfügbar – das echte Modell müsste bei jedem Besuch neu laden.', details };
  }
  try {
    const cache = await caches.open('igai-diagnose');
    const key = `${import.meta.env.BASE_URL}__diagnose-probe`;
    await cache.put(key, new Response(new Uint8Array(1024 * 64)));
    const hit = await cache.match(key);
    const ok = !!hit && (await hit.arrayBuffer()).byteLength === 1024 * 64;
    await caches.delete('igai-diagnose');
    return { status: ok ? 'ok' : 'fail', summary: ok ? 'Cache API funktioniert (Schreiben + Lesen).' : 'Cache API liefert falsche Daten.', details };
  } catch (e) {
    return { status: 'fail', summary: 'Cache API blockiert.', details: [...details, String(e)] };
  }
}

async function checkWebGpu(): Promise<Check> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gpu = (navigator as any).gpu;
  if (!gpu) return { status: 'warn', summary: 'WebGPU nicht verfügbar – kein Problem, die Engine rechnet auf der CPU.' };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { status: 'warn', summary: 'WebGPU vorhanden, aber kein Grafik-Adapter freigegeben.' };
    const info = adapter.info ?? {};
    return {
      status: 'ok',
      summary: 'WebGPU verfügbar (optionaler Turbo ab P6).',
      details: [
        `Adapter: ${[info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' · ') || 'keine Angaben'}`,
        `maxBufferSize: ${fmt(adapter.limits.maxBufferSize / MB, 0)} MB`,
        `maxStorageBufferBindingSize: ${fmt(adapter.limits.maxStorageBufferBindingSize / MB, 0)} MB`,
      ],
    };
  } catch (e) {
    return { status: 'warn', summary: 'WebGPU-Abfrage fehlgeschlagen.', details: [String(e)] };
  }
}

function checkBench(): Promise<Check> {
  return new Promise((resolve) => {
    const worker = new Worker(new URL('../diagnose/bench.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<{ gmacPerSec: number; iterations: number }>) => {
      worker.terminate();
      const factParams = getFact('smollm2', 'params');
      const params = typeof factParams === 'number' ? factParams : FALLBACK_PARAMS;
      const msPerToken = (params / (e.data.gmacPerSec * 1e9)) * 1000;
      resolve({
        status: msPerToken < 600 ? 'ok' : 'warn',
        summary: `${fmt(e.data.gmacPerSec, 2)} GMAC/s · geschätzt ≈ ${fmt(msPerToken, 0)} ms pro Token (SmolLM2-135M, ein Thread)`,
        details: [
          `Grundlage: ${typeof factParams === 'number' ? 'Parameterzahl aus der Modell-Config' : 'Schätzwert der Parameterzahl (echte Config ab P2)'}`,
          'Nur die reine Rechenzeit der Matrixmultiplikationen; reale Werte liegen etwas darüber.',
          `${e.data.iterations} Durchläufe`,
        ],
      });
    };
    worker.onerror = (e) => {
      worker.terminate();
      resolve({ status: 'fail', summary: 'Web Worker konnte nicht starten.', details: [e.message] });
    };
    worker.postMessage({ durationMs: 1500 });
  });
}

async function checkEnv(): Promise<Check> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nav = navigator as any;
  return {
    status: 'ok',
    summary: `${nav.hardwareConcurrency ?? '?'} Kerne · ${nav.deviceMemory ? `≈ ${nav.deviceMemory} GB RAM` : 'RAM unbekannt'}`,
    details: [
      `Browser: ${navigator.userAgent}`,
      `Bildschirm: ${window.innerWidth}×${window.innerHeight} @${window.devicePixelRatio}x`,
      `Farbschema: ${window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dunkel' : 'hell'}`,
      `Reduzierte Bewegung: ${window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'ja' : 'nein'}`,
      `crossOriginIsolated: ${window.crossOriginIsolated ? 'ja' : 'nein'} (auf GitHub Pages erwartet: nein)`,
    ],
  };
}

async function checkExternal(): Promise<Check> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 8000);
  try {
    await fetch('https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M', { mode: 'no-cors', cache: 'no-store', signal: ctrl.signal });
    return { status: 'ok', summary: 'huggingface.co ist erreichbar.', details: ['Nur zur Information: Die Seite lädt das Modell von der eigenen Adresse, nicht von dort.'] };
  } catch {
    return { status: 'warn', summary: 'huggingface.co ist nicht erreichbar (z. B. Firmennetz).', details: ['Kein Problem: Das Modell wird von der eigenen Adresse (github.io) ausgeliefert.'] };
  } finally {
    clearTimeout(t);
  }
}

const LABELS: Record<CheckId, string> = {
  download: 'Download (30-MB-Test-Shard)',
  cache: 'Speicher & Cache API',
  webgpu: 'WebGPU',
  bench: 'Rechen-Benchmark',
  env: 'Umgebung',
  external: 'Externe Erreichbarkeit (optional)',
};
const AUTO: CheckId[] = ['env', 'cache', 'webgpu', 'bench', 'download'];
const ICON: Record<Status, string> = { idle: '○', running: '◌', ok: '✓', warn: '!', fail: '✗' };

export default function DiagnosePage() {
  const [checks, setChecks] = useState<Partial<Record<CheckId, Check>>>({});
  const [progress, setProgress] = useState(0);
  const [copied, setCopied] = useState(false);

  const run = useCallback(async (id: CheckId) => {
    setChecks((c) => ({ ...c, [id]: { status: 'running', summary: 'läuft …' } }));
    const fn: Record<CheckId, () => Promise<Check>> = {
      download: () => checkDownload(setProgress),
      cache: checkCache,
      webgpu: checkWebGpu,
      bench: checkBench,
      env: checkEnv,
      external: checkExternal,
    };
    let result: Check;
    try {
      result = await fn[id]();
    } catch (e) {
      result = { status: 'fail', summary: 'Unerwarteter Fehler.', details: [String(e)] };
    }
    setChecks((c) => ({ ...c, [id]: result }));
  }, []);

  const runAll = async () => {
    setProgress(0);
    for (const id of AUTO) await run(id);
  };

  const report = useMemo(() => {
    const lines = [`Inside GenAI · Diagnose · ${new Date().toLocaleString('de-DE')}`, `Seite: ${location.href}`, ''];
    for (const id of [...AUTO, 'external'] as CheckId[]) {
      const c = checks[id];
      if (!c) continue;
      lines.push(`[${ICON[c.status]}] ${LABELS[id]}: ${c.summary}`);
      for (const d of c.details ?? []) lines.push(`    ${d}`);
    }
    return lines.join('\n');
  }, [checks]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      (document.getElementById('diag-report') as HTMLTextAreaElement | null)?.select();
    }
  };

  const running = Object.values(checks).some((c) => c?.status === 'running');

  return (
    <article className="page diagnose">
      <header className="page__head">
        <span className="meta">Werkzeug</span>
        <h1>Diagnose</h1>
        <p className="lead muted">
          Prüft, ob dieses Gerät und dieses Netz für das echte Modell taugen: Download-Geschwindigkeit, Zwischenspeicher,
          Grafikbeschleunigung und Rechenleistung. Es werden keine Daten an Dritte gesendet.
        </p>
        <div className="diagnose__actions">
          <button type="button" className="btn btn--primary" onClick={runAll} disabled={running}>
            {running ? 'Läuft …' : 'Alle Tests starten'}
          </button>
          <button type="button" className="btn" onClick={() => run('external')} disabled={running}>
            huggingface.co testen
          </button>
        </div>
      </header>

      <ul className="checks">
        {([...AUTO, 'external'] as CheckId[]).map((id) => {
          const c = checks[id];
          const st = c?.status ?? 'idle';
          return (
            <li key={id} className={`check check--${st}`}>
              <span className="check__icon mono" aria-hidden="true">
                {ICON[st]}
              </span>
              <div className="check__body">
                <div className="check__head">
                  <strong>{LABELS[id]}</strong>
                  <button type="button" className="btn btn--small" onClick={() => run(id)} disabled={st === 'running'}>
                    {c ? 'Wiederholen' : 'Starten'}
                  </button>
                </div>
                <p>{c?.summary ?? 'Noch nicht ausgeführt.'}</p>
                {id === 'download' && st === 'running' && (
                  <div className="check__bar" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
                    <span style={{ width: `${Math.min(100, progress * 100)}%` }} />
                  </div>
                )}
                {c?.details && (
                  <ul className="check__details small muted">
                    {c.details.map((d) => (
                      <li key={d}>{d}</li>
                    ))}
                  </ul>
                )}
              </div>
            </li>
          );
        })}
      </ul>

      <section className="diagnose__report">
        <div className="check__head">
          <h2>Bericht</h2>
          <button type="button" className="btn" onClick={copy} disabled={!Object.keys(checks).length}>
            {copied ? 'Kopiert ✓' : 'Bericht kopieren'}
          </button>
        </div>
        <textarea id="diag-report" className="mono" readOnly value={report} rows={14} aria-label="Diagnosebericht" />
      </section>
    </article>
  );
}
