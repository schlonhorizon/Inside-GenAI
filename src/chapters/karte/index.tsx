import { Figure } from '../../components/primitives';
import { getChapter, pad } from '../../content/chapters';
import { useProgress } from '../../state/progress';
import { useReducedMotionPref } from '../../state/settings';
import { href } from '../../app/router';

/**
 * Die große Karte: Pipeline vom Prompt zum nächsten Token, jeder Block führt in sein Kapitel.
 */

interface MapNode {
  slug: string;
  x: number;
  y: number;
  w: number;
  title: string;
  sub: string;
  dashed?: boolean;
}

const Y = 130;
const H = 62;
const NODES: MapNode[] = [
  { slug: 'tokenisierung', x: 96, y: Y, w: 124, title: 'Tokenizer', sub: 'Text → IDs' },
  { slug: 'embeddings', x: 248, y: Y, w: 124, title: 'Embedding', sub: 'ID → Vektor' },
  { slug: 'position', x: 248, y: Y + 74, w: 124, title: '+ Position', sub: 'Reihenfolge' },
  { slug: 'attention', x: 400, y: Y, w: 150, title: 'Attention', sub: 'worauf achten?' },
  { slug: 'transformer-block', x: 400, y: Y + 74, w: 150, title: 'Block × N', sub: 'MLP · Residual' },
  { slug: 'next-token', x: 578, y: Y, w: 128, title: 'Softmax', sub: 'Logits → p' },
  { slug: 'generierung', x: 734, y: Y, w: 128, title: 'Sampling', sub: 'ziehen · Loop' },
  { slug: 'grundlagen', x: 400, y: 28, w: 150, title: 'Neuronale Netze', sub: 'die Bausteine', dashed: true },
  { slug: 'training', x: 330, y: 318, w: 140, title: 'Training', sub: 'Gewichte lernen', dashed: true },
  { slug: 'assistant', x: 490, y: 318, w: 150, title: 'Base → Assistent', sub: 'Fine-Tuning', dashed: true },
];

function Mark({ slug, x, y }: { slug: string; x: number; y: number }) {
  const p = useProgress((s) => s.chapters[slug]);
  const state = p?.completed ? 'done' : p?.visited ? 'seen' : 'new';
  return (
    <g transform={`translate(${x} ${y})`} className={`map-mark map-mark--${state}`}>
      <circle r="5" />
      {state === 'seen' && <path d="M 0 -5 A 5 5 0 0 0 0 5 Z" />}
    </g>
  );
}

export function MapFigure() {
  const reduce = useReducedMotionPref();
  const main = `M 20 ${Y + H / 2} H 920`;
  const loop = `M 798 ${Y + H} V 296 H 158 V ${Y + H}`;
  return (
    <Figure num="0.1" caption="Der Weg vom Prompt zum nächsten Token. Jeder Block führt in sein Kapitel; Punkte zeigen deinen Fortschritt." stamp="illustrative">
      <svg viewBox="0 0 960 400" className="map" role="img" aria-labelledby="map-title">
        <title id="map-title">Karte der Pipeline eines Sprachmodells mit Links zu den Kapiteln</title>
        <defs>
          <marker id="map-chev" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M 1 1 L 8 5 L 1 9" fill="none" stroke="var(--ink)" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
          </marker>
        </defs>
        <text x="20" y={Y + H / 2 - 10} className="svg-mono svg-muted">
          Prompt
        </text>
        {[
          [20, 96],
          [220, 248],
          [372, 400],
          [550, 578],
          [706, 734],
          [862, 900],
        ].map(([a, b]) => (
          <path key={a} d={`M ${a} ${Y + H / 2} H ${b}`} className="svg-line" markerEnd="url(#map-chev)" />
        ))}
        <text x="904" y={Y + H / 2 + 5} className="svg-out mono">
          Token
        </text>
        <path d={loop} className="svg-line svg-line--muted" markerEnd="url(#map-chev)" />
        <text x="170" y="288" className="svg-mono svg-muted">
          Token anhängen · wiederholen
        </text>
        {/* gestrichelte Beziehungen: Bausteine und Herkunft der Gewichte */}
        <path d={`M 475 ${28 + H} V ${Y}`} className="svg-line svg-line--faint map-dash" />
        <path d={`M 400 318 L 450 ${Y + 74 + H}`} className="svg-line svg-line--faint map-dash" />
        <path d={`M 565 318 L 520 ${Y + 74 + H}`} className="svg-line svg-line--faint map-dash" />
        <text x="652" y="354" className="svg-small">
          Herkunft der Gewichte
        </text>

        {NODES.map((n) => {
          const c = getChapter(n.slug);
          return (
            <a key={n.slug} href={href.chapter(n.slug)} className="map-node" aria-label={`Kapitel ${c?.number}: ${c?.title}`}>
              <rect x={n.x} y={n.y} width={n.w} height={H} rx="6" className={`svg-node ${n.dashed ? 'map-node--dashed' : ''}`} />
              <text x={n.x + 12} y={n.y + 20} className="svg-mono svg-muted">
                {pad(c?.number ?? 0)}
              </text>
              <text x={n.x + 12} y={n.y + 38} className="svg-label">
                {n.title}
              </text>
              <text x={n.x + 12} y={n.y + 54} className="svg-small">
                {n.sub}
              </text>
              <Mark slug={n.slug} x={n.x + n.w - 12} y={n.y + 14} />
            </a>
          );
        })}

        {!reduce && (
          <>
            <circle r="5" className="svg-pulse">
              <animateMotion dur="4.5s" repeatCount="indefinite" path={main} />
            </circle>
            <circle r="4" className="svg-pulse">
              <animateMotion dur="4.5s" begin="2.25s" repeatCount="indefinite" path={loop} />
            </circle>
          </>
        )}
      </svg>
    </Figure>
  );
}
