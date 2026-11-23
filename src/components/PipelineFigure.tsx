import { useEffect, useRef, useState } from 'react';
import { animate, motion, useInView, useMotionValue, useReducedMotion, useTransform } from 'motion/react';

/**
 * Mini-Pipeline als Line-Art: Linien zeichnen sich beim Erscheinen,
 * ein Puls (Akzent) zeigt den Datenfluss inkl. autoregressivem Loop.
 * Reduced Motion: alles sofort fertig gezeichnet, kein Puls.
 */

const Y = 112; // Mittellinie des Datenflusses
const NODES = [
  { id: 'tok', x: 100, w: 112, label: 'Tokenizer', shape: 'IDs [T]' },
  { id: 'tf', x: 262, w: 184, label: 'Transformer', shape: '[T × d_model]' },
  { id: 'sm', x: 496, w: 112, label: 'Softmax', shape: 'p über Vokabular' },
  { id: 'sa', x: 658, w: 112, label: 'Sampling', shape: '1 Token' },
];
const H = 64;
const MAIN_PATH = `M 18 ${Y} H 808`;
const LOOP_PATH = `M 714 ${Y + H / 2} V 196 H 156 V ${Y + H / 2}`;

function usePulse(pathRef: React.RefObject<SVGPathElement | null>, progress: ReturnType<typeof useMotionValue<number>>) {
  const x = useTransform(progress, (p) => {
    const el = pathRef.current;
    if (!el) return -100;
    return el.getPointAtLength(p * el.getTotalLength()).x;
  });
  const y = useTransform(progress, (p) => {
    const el = pathRef.current;
    if (!el) return -100;
    return el.getPointAtLength(p * el.getTotalLength()).y;
  });
  return { x, y };
}

export function PipelineFigure() {
  const reduce = useReducedMotion();
  const wrapRef = useRef<HTMLDivElement>(null);
  const inView = useInView(wrapRef, { once: true, amount: 0.4 });
  const [playing, setPlaying] = useState(true);
  const [runKey, setRunKey] = useState(0);

  const mainRef = useRef<SVGPathElement>(null);
  const loopRef = useRef<SVGPathElement>(null);
  const pMain = useMotionValue(0);
  const pLoop = useMotionValue(0);
  const main = usePulse(mainRef, pMain);
  const loop = usePulse(loopRef, pLoop);
  const [phase, setPhase] = useState<'idle' | 'main' | 'loop'>('idle');

  useEffect(() => {
    if (reduce || !inView || !playing) {
      setPhase('idle');
      return;
    }
    let cancelled = false;
    let controls: { stop: () => void } | undefined;
    const run = async () => {
      // erst Linien zeichnen lassen
      await new Promise((r) => setTimeout(r, 1200));
      while (!cancelled) {
        setPhase('main');
        pMain.set(0);
        controls = animate(pMain, 1, { duration: 2.4, ease: 'linear' });
        await controls;
        if (cancelled) break;
        setPhase('loop');
        pLoop.set(0);
        controls = animate(pLoop, 1, { duration: 1.6, ease: 'linear' });
        await controls;
        if (cancelled) break;
        setPhase('idle');
        await new Promise((r) => setTimeout(r, 500));
      }
    };
    run();
    return () => {
      cancelled = true;
      controls?.stop();
    };
  }, [reduce, inView, playing, runKey, pMain, pLoop]);

  const draw = (delay: number) =>
    reduce
      ? { initial: false as const }
      : {
          initial: { pathLength: 0 },
          animate: inView ? { pathLength: 1 } : { pathLength: 0 },
          transition: { duration: 0.8, delay, ease: [0.2, 0.7, 0.2, 1] as const },
        };

  // aktiver Knoten = der, den der Puls gerade durchläuft
  const activeNode = useTransform(pMain, (p) => {
    const x = 18 + p * (808 - 18);
    const n = NODES.find((n) => x >= n.x && x <= n.x + n.w);
    return n?.id ?? '';
  });
  const [active, setActive] = useState('');
  useEffect(() => activeNode.on('change', (v) => setActive(v)), [activeNode]);

  return (
    <div ref={wrapRef} className="pipeline">
      <svg viewBox="0 0 880 236" className="pipeline__svg" role="img" aria-labelledby="pipe-title pipe-desc">
        <title id="pipe-title">Mini-Pipeline eines Sprachmodells</title>
        <desc id="pipe-desc">
          Prompt → Tokenizer → Transformer (N Layer) → Softmax → Sampling → nächstes Token; das Token wird angehängt
          und der Ablauf wiederholt sich.
        </desc>
        <defs>
          <marker id="chev" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M 1 1 L 8 5 L 1 9" fill="none" stroke="var(--ink)" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
          </marker>
          <marker id="chev-muted" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M 1 1 L 8 5 L 1 9" fill="none" stroke="var(--ink-muted)" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
          </marker>
        </defs>

        {/* Prompt-Label */}
        <text x="18" y={Y - 14} className="svg-mono svg-muted">prompt</text>

        {/* Verbindungen (Segmente zwischen den Knoten) */}
        {[
          [18, 100],
          [212, 262],
          [458, 496],
          [608, 658],
          [770, 808],
        ].map(([a, b], i) => (
          <motion.path
            key={i}
            d={`M ${a} ${Y} H ${b}`}
            className="svg-line"
            markerEnd="url(#chev)"
            {...draw(0.15 * i)}
          />
        ))}
        <path ref={mainRef} d={MAIN_PATH} fill="none" stroke="none" />

        {/* Loop zurück (autoregressiv) */}
        <motion.path
          ref={loopRef}
          d={LOOP_PATH}
          className="svg-line svg-line--muted"
          markerEnd="url(#chev-muted)"
          {...draw(0.9)}
        />
        <text x="435" y="190" textAnchor="middle" className="svg-mono svg-muted">
          Token anhängen · wiederholen
        </text>

        {/* Knoten */}
        {NODES.map((n, i) => (
          <g key={n.id}>
            <motion.rect
              x={n.x}
              y={Y - H / 2}
              width={n.w}
              height={H}
              rx="6"
              className={`svg-node ${active === n.id && phase === 'main' ? 'is-active' : ''}`}
              {...draw(0.1 + 0.15 * i)}
            />
            {n.id === 'tf' && (
              <>
                {/* angedeuteter Stapel */}
                <motion.path
                  d={`M ${n.x + 6} ${Y - H / 2 - 6} H ${n.x + n.w + 6} V ${Y + H / 2 - 6}`}
                  className="svg-line svg-line--faint"
                  {...draw(0.6)}
                />
                <motion.path
                  d={`M ${n.x + 12} ${Y - H / 2 - 12} H ${n.x + n.w + 12} V ${Y + H / 2 - 12}`}
                  className="svg-line svg-line--faint"
                  {...draw(0.7)}
                />
              </>
            )}
            <text x={n.x + n.w / 2} y={Y + 5} textAnchor="middle" className="svg-label">
              {n.label}
            </text>
            <text x={n.x + n.w / 2} y={Y + H / 2 + 18} textAnchor="middle" className="svg-mono svg-muted">
              {n.shape}
            </text>
          </g>
        ))}

        {/* Bemaßung über dem Transformer */}
        <g className="svg-dim">
          <path d="M 262 46 V 30 M 458 46 V 30" />
          <path d="M 262 38 H 458" />
          <path d="M 258 42 L 266 34 M 454 42 L 462 34" />
          <rect x="316" y="29" width="88" height="18" className="svg-dim__bg" />
          <text x="360" y="42" textAnchor="middle" className="svg-mono">× N Layer</text>
        </g>

        {/* Callout */}
        <g className="svg-callout">
          <path d="M 552 80 L 552 50 L 580 30" />
          <circle cx="590" cy="24" r="10" />
          <text x="590" y="28" textAnchor="middle" className="svg-mono">1</text>
          <text x="606" y="28" className="svg-small">hier entsteht die Verteilung</text>
        </g>

        {/* Ausgabe-Token */}
        <text x="814" y={Y + 5} className="svg-out mono">weil</text>

        {/* Puls */}
        {!reduce && phase === 'main' && <motion.circle r="5" cx={main.x} cy={main.y} className="svg-pulse" />}
        {!reduce && phase === 'loop' && <motion.circle r="4" cx={loop.x} cy={loop.y} className="svg-pulse" />}
      </svg>

      <div className="pipeline__controls">
        {reduce ? (
          <span className="meta">Reduzierte Bewegung aktiv – Diagramm statisch</span>
        ) : (
          <>
            <button type="button" className="btn" onClick={() => setPlaying((p) => !p)}>
              {playing ? 'Pause' : 'Abspielen'}
            </button>
            <button type="button" className="btn" onClick={() => setRunKey((k) => k + 1)}>
              Neu starten
            </button>
          </>
        )}
      </div>
    </div>
  );
}
