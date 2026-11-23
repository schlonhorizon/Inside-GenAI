import { useEffect, useMemo, useRef, useState } from 'react';
import { Figure } from '../../components/primitives';
import { EXAMPLES } from '../../content/examples';
import { distribution, mulberry32, rankOrder, sampleIndex, type SamplingParams } from '../../engine/sampling';
import type { ToyLanguageModel } from '../../engine/lm';
import { useReducedMotionPref } from '../../state/settings';
import { PromptInput } from '../shared/PromptInput';
import { pct, SamplingControls, showWhitespace, WithToyModel } from '../shared/Shared';

/* ── Datenmodell des Antwort-Baums ───────────────────────────────── */

interface TreeNode {
  id: number;
  parent: number | null;
  token: number | null; // null = Wurzel (Prompt)
  p: number; // Wahrscheinlichkeit dieses Tokens im Kontext des Elternknotens
  forced: boolean;
  children: number[];
}

interface Tree {
  nodes: TreeNode[];
  current: number;
}

const MAX_NEW = 12;
const newTree = (): Tree => ({ nodes: [{ id: 0, parent: null, token: null, p: 1, forced: false, children: [] }], current: 0 });

function path(tree: Tree, id: number): TreeNode[] {
  const out: TreeNode[] = [];
  let n: TreeNode | undefined = tree.nodes[id];
  while (n && n.parent !== null) {
    out.unshift(n);
    n = tree.nodes[n.parent];
  }
  return out;
}

function addChild(tree: Tree, parent: number, token: number, p: number, forced: boolean): Tree {
  const existing = tree.nodes[parent].children.find((c) => tree.nodes[c].token === token);
  if (existing !== undefined) return { ...tree, current: existing };
  const id = tree.nodes.length;
  const nodes = tree.nodes.map((n) => (n.id === parent ? { ...n, children: [...n.children, id] } : n));
  nodes.push({ id, parent, token, p, forced, children: [] });
  return { nodes, current: id };
}

/* ── Generator ────────────────────────────────────────────────────── */

function GeneratorInner({ model }: { model: ToyLanguageModel }) {
  const reduce = useReducedMotionPref();
  const [prompt, setPrompt] = useState(EXAMPLES.toyPrompts[0]);
  const [params, setParams] = useState<SamplingParams>({ temperature: 1, topK: 0, topP: 1, minP: 0 });
  const [seed, setSeed] = useState(7);
  const [tree, setTree] = useState<Tree>(newTree);
  const [draws, setDraws] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [inspect, setInspect] = useState<number | null>(null);

  const promptIds = useMemo(() => model.promptIds(prompt), [model, prompt]);
  const cur = path(tree, tree.current);
  const ctx = (nodes: TreeNode[]) => [...promptIds, ...nodes.map((n) => n.token!)];
  const ended = cur.length >= MAX_NEW || (cur.length > 0 && model.eosIds.includes(cur[cur.length - 1].token!));

  const step = () => {
    if (ended) return false;
    const d = distribution(model.logitsFor(ctx(cur)), params);
    const tok = sampleIndex(d.final, mulberry32(seed * 7919 + draws)());
    setDraws((x) => x + 1);
    setTree((t) => addChild(t, t.current, tok, d.probs[tok], false));
    return true;
  };

  const stepRef = useRef(step);
  stepRef.current = step;
  useEffect(() => {
    if (!playing) return;
    if (ended) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => stepRef.current(), reduce ? 150 : 550);
    return () => clearTimeout(t);
  }, [playing, ended, tree, reduce]);

  const restart = (p = prompt) => {
    setPrompt(p);
    setTree(newTree());
    setDraws(0);
    setInspect(null);
    setPlaying(false);
  };

  // Alternativen an der inspizierten Stelle (Index in cur)
  const alt = useMemo(() => {
    if (inspect === null || inspect >= cur.length) return null;
    const d = distribution(model.logitsFor(ctx(cur.slice(0, inspect))), params);
    return { d, order: rankOrder(d.probs).slice(0, 6), chosen: cur[inspect].token! };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inspect, tree, params, promptIds, model]);

  const force = (token: number) => {
    if (inspect === null || !alt) return;
    const parent = inspect === 0 ? 0 : cur[inspect - 1].id;
    setTree((t) => addChild(t, parent, token, alt.d.probs[token], token !== alt.chosen));
    setInspect(null);
  };

  return (
    <div className="widget">
      <PromptInput model={model} value={prompt} onChange={(v) => restart(v)} />
      <SamplingControls value={params} onChange={setParams} vocabSize={model.vocabSize} seed={seed} onSeed={setSeed} />
      <div className="row-gap">
        <button type="button" className="btn btn--primary" onClick={() => setPlaying(!playing)} disabled={ended && !playing}>
          {playing ? 'Pause' : 'Abspielen'}
        </button>
        <button type="button" className="btn" onClick={() => step()} disabled={ended || playing}>
          Ein Schritt
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => setTree((t) => ({ ...t, current: t.nodes[t.current].parent ?? 0 }))}
          disabled={tree.current === 0 || playing}
        >
          Zurück
        </button>
        <button type="button" className="btn" onClick={() => restart()}>
          Neu
        </button>
        <span className="mono small muted">
          {cur.length} / {MAX_NEW} Tokens{ended ? ' · fertig' : ''}
        </span>
      </div>

      <div className="gen-line" aria-live="polite">
        <span className="gen-prompt mono">{model.decode(promptIds)}</span>
        {cur.map((n, i) => (
          <button
            key={n.id}
            type="button"
            className={`gen-tok ${n.forced ? 'gen-tok--forced' : ''}`}
            aria-pressed={inspect === i}
            onClick={() => setInspect(inspect === i ? null : i)}
            title="Klicken: Alternativen an dieser Stelle"
          >
            <span className="mono">{model.eosIds.includes(n.token!) ? '⏎ Ende' : showWhitespace(model.token(n.token!).text)}</span>
            <span className="gen-tok__p" style={{ width: `${Math.max(4, n.p * 100)}%` }} />
          </button>
        ))}
      </div>

      {alt && (
        <div className="alts">
          <span className="meta">Alternativen an Position {inspect! + 1} – klicken, um dieses Token zu erzwingen</span>
          <div className="alts__list">
            {alt.order.map((i) => (
              <button key={i} type="button" className={`alt ${i === alt.chosen ? 'alt--chosen' : ''}`} onClick={() => force(i)}>
                <span className="mono">{model.eosIds.includes(i) ? '⏎ Ende' : showWhitespace(model.token(i).text)}</span>
                <span className="mono small muted">{pct(alt.d.probs[i])}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <AnswerTree tree={tree} model={model} onSelect={(id) => setTree((t) => ({ ...t, current: id }))} />
    </div>
  );
}

/* ── Baum-Darstellung ─────────────────────────────────────────────── */

const COL = 96;
const ROW = 44;

function AnswerTree({ tree, model, onSelect }: { tree: Tree; model: ToyLanguageModel; onSelect: (id: number) => void }) {
  const layout = useMemo(() => {
    const pos = new Map<number, { x: number; y: number }>();
    let leaf = 0;
    const visit = (id: number, depth: number): number => {
      const n = tree.nodes[id];
      let y: number;
      if (!n.children.length) y = leaf++ * ROW;
      else {
        const ys = n.children.map((c) => visit(c, depth + 1));
        y = (ys[0] + ys[ys.length - 1]) / 2;
      }
      pos.set(id, { x: depth * COL, y });
      return y;
    };
    visit(0, 0);
    return { pos, leaves: Math.max(1, leaf) };
  }, [tree]);

  const onPath = new Set(path(tree, tree.current).map((n) => n.id).concat(0));
  const cumulative = (id: number) => path(tree, id).reduce((acc, n) => acc * n.p, 1);
  const depth = Math.max(...[...layout.pos.values()].map((p) => p.x)) / COL;
  const w = (depth + 1) * COL + 120;
  const h = layout.leaves * ROW + 20;

  if (tree.nodes.length === 1) return <p className="small muted tree-empty">Der Antwort-Baum wächst, sobald du Tokens erzeugst.</p>;

  return (
    <div className="tree-wrap">
      <svg width={w} height={h} className="tree" role="img" aria-label="Antwort-Baum: erzeugte und erzwungene Fortsetzungen">
        <g transform="translate(10 20)">
          {tree.nodes.map((n) => {
            if (n.parent === null) return null;
            const a = layout.pos.get(n.parent)!;
            const b = layout.pos.get(n.id)!;
            const active = onPath.has(n.id);
            return (
              <path
                key={`e${n.id}`}
                d={`M ${a.x + 56} ${a.y} C ${a.x + 76} ${a.y}, ${b.x - 20} ${b.y}, ${b.x} ${b.y}`}
                className={`tree__edge ${active ? 'is-active' : ''}`}
              />
            );
          })}
          {tree.nodes.map((n) => {
            const p = layout.pos.get(n.id)!;
            const active = onPath.has(n.id);
            const label = n.token === null ? 'Prompt' : model.eosIds.includes(n.token) ? '⏎' : model.token(n.token).text;
            const leaf = !n.children.length && n.parent !== null;
            return (
              <g key={n.id} transform={`translate(${p.x} ${p.y})`} className="tree__node" onClick={() => onSelect(n.id)}>
                <rect x={0} y={-12} width={56} height={24} rx={6} className={`tree__box ${active ? 'is-active' : ''} ${n.forced ? 'is-forced' : ''}`} />
                <text x={28} y={4} textAnchor="middle" className="tree__label">
                  {label.length > 9 ? label.slice(0, 8) + '…' : label}
                </text>
                {n.parent !== null && (
                  <text x={28} y={22} textAnchor="middle" className="tree__p">
                    {pct(n.p, 0)}
                  </text>
                )}
                {leaf && (
                  <text x={64} y={4} className="tree__cum">
                    Σ {pct(cumulative(n.id), 2)}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      <p className="small muted">
        Unter jedem Token: seine Wahrscheinlichkeit an dieser Stelle. Am Astende: Wahrscheinlichkeit des ganzen Asts (Produkt).
        Gestrichelt = erzwungen. Klick auf einen Knoten springt dorthin.
      </p>
    </div>
  );
}

export function Generator() {
  return (
    <Figure num="8.1" caption="Token für Token: Jede Entscheidung verändert alles, was danach kommt." stamp="toy">
      <WithToyModel>{(m) => <GeneratorInner model={m} />}</WithToyModel>
    </Figure>
  );
}

/* ── KV-Cache (Erklärgrafik) ──────────────────────────────────────── */

export function KvCacheFigure() {
  const tokens = ['die', 'bank', 'erhöht', 'den', 'zins'];
  return (
    <Figure num="8.2" caption="KV-Cache: Keys und Values früherer Tokens werden gespeichert, pro Schritt kommt nur eine neue Zeile hinzu." stamp="illustrative">
      <svg viewBox="0 0 600 190" className="kv" role="img" aria-label="KV-Cache: vier gespeicherte Zeilen, eine neue">
        <text x="10" y="18" className="svg-mono svg-muted">
          Token
        </text>
        <text x="150" y="18" className="svg-mono svg-muted">
          Key
        </text>
        <text x="300" y="18" className="svg-mono svg-muted">
          Value
        </text>
        {tokens.map((t, i) => {
          const isNew = i === tokens.length - 1;
          const y = 30 + i * 30;
          return (
            <g key={t}>
              <text x="10" y={y + 16} className={isNew ? 'svg-out svg-mono' : 'svg-mono'}>
                {t}
              </text>
              {[150, 300].map((x) => (
                <g key={x}>
                  {Array.from({ length: 6 }, (_, j) => (
                    <rect
                      key={j}
                      x={x + j * 20}
                      y={y}
                      width={16}
                      height={20}
                      rx={3}
                      className={isNew ? 'kv__cell kv__cell--new' : 'kv__cell'}
                    />
                  ))}
                </g>
              ))}
              {!isNew && (
                <text x="460" y={y + 15} className="svg-small">
                  aus dem Cache
                </text>
              )}
              {isNew && (
                <text x="460" y={y + 15} className="svg-small">
                  neu berechnet
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </Figure>
  );
}
