import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseSafeTensors } from '../src/engine/model/safetensors';
import { configFromHf, LlamaModel, loadLlamaWeights, parseFormat, weightBytes } from '../src/engine/model/llama';
import { quantizeGroups, quantizeRows } from '../src/engine/model/quantize';
import { linearRow, type Linear } from '../src/engine/core/ops';
import { KvSessions } from '../src/engine/model/sessions';
import { packWeights, sha256Hex, shardRanges, unpackWeights } from '../src/engine/model/pack';

/**
 * Engine gegen eine unabhängige NumPy-Referenz (scripts/spike/make_tiny_fixture.py):
 * winziges Zufalls-Llama mit GQA, RoPE, SwiGLU, gebundenen Embeddings.
 */

const dir = new URL('./fixtures/', import.meta.url);
const buf = readFileSync(new URL('tiny-llama.safetensors', dir));
const st = parseSafeTensors(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const cfg = configFromHf(JSON.parse(readFileSync(new URL('tiny-llama.config.json', dir), 'utf8')));
const ref = JSON.parse(readFileSync(new URL('tiny-llama.reference.json', dir), 'utf8')) as {
  tokens: number[];
  logits: number[][];
};

function relL2(a: ArrayLike<number>, b: ArrayLike<number>) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < b.length; i++) {
    num += (a[i] - b[i]) ** 2;
    den += b[i] ** 2;
  }
  return Math.sqrt(num / den);
}

describe('Llama-Engine (Mini-Modell)', () => {
  it('fp32: Logits an jeder Position stimmen mit der Referenz überein', () => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, 'f32'), 32);
    ref.tokens.forEach((t, i) => {
      const logits = model.step(t)!;
      let maxAbs = 0;
      for (let j = 0; j < logits.length; j++) maxAbs = Math.max(maxAbs, Math.abs(logits[j] - ref.logits[i][j]));
      expect(maxAbs, `Position ${i}`).toBeLessThan(1e-3);
    });
  });

  it('int8: Logits bleiben nah an der Referenz (relativer L2-Fehler < 5 %)', () => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, 'q8'), 32);
    ref.tokens.forEach((t, i) => {
      const logits = model.step(t)!;
      expect(relL2(logits, ref.logits[i]), `Position ${i}`).toBeLessThan(0.05);
    });
  });

  it('Gruppen-int8 rekonstruiert jedes Gewicht genauer als int8 pro Zeile', () => {
    for (const name of st.names().filter((n) => st.header[n].shape.length === 2)) {
      const { shape, data } = st.get(name);
      const [rows, cols] = shape;
      const err = (l: Linear) => {
        const out = new Float32Array(cols);
        let e = 0;
        for (let r = 0; r < rows; r++) {
          linearRow(l, r, out);
          for (let c = 0; c < cols; c++) e += (out[c] - data[r * cols + c]) ** 2;
        }
        return e;
      };
      expect(err(quantizeGroups(data, rows, cols, 16)), name).toBeLessThan(err(quantizeRows(data, rows, cols)));
    }
  });

  it.each(['q8g16', 'q8g16+s16', 'attn:q8,mlp:q8,down:q8,emb:f16', 'f16'])('%s: Logits nah an der Referenz', (precision) => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, precision), 32);
    ref.tokens.forEach((t, i) => {
      expect(relL2(model.step(t)!, ref.logits[i]), `Position ${i}`).toBeLessThan(0.05);
    });
  });

  it('Formatangaben werden korrekt gelesen', () => {
    expect(parseFormat('q8g64+s16')).toEqual({ attn: 'q8g64', mlp: 'q8g64', down: 'q8g64', emb: 'q8g64', scales16: true });
    expect(parseFormat('attn:q8,emb:f16')).toMatchObject({ attn: 'q8', mlp: 'f32', down: 'f32', emb: 'f16', scales16: false });
    expect(() => parseFormat('q4')).toThrow();
  });

  it('prefill + reset liefern dasselbe wie Einzelschritte', () => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, 'f32'), 32);
    const a = Array.from(model.prefill(ref.tokens));
    model.reset();
    const b = Array.from(model.prefill(ref.tokens));
    expect(b).toEqual(a);
  });
});

describe('Batch-Prefill und KV-Sitzungen (Mini-Modell)', () => {
  const weights = loadLlamaWeights(st, cfg, 'q8g16+s16');
  const fresh = (ids: number[]) => {
    const m = new LlamaModel(cfg, weights, 64);
    return Array.from(m.prefill(ids));
  };

  it('prefillBatch ist bitgleich mit Token für Token (f32 und int8)', () => {
    for (const w of [loadLlamaWeights(st, cfg, 'f32'), weights]) {
      const a = new LlamaModel(cfg, w, 64);
      const b = new LlamaModel(cfg, w, 64);
      expect(Array.from(b.prefillBatch(ref.tokens))).toEqual(Array.from(a.prefill(ref.tokens)));
      // Verlängerung im Batch an einen bestehenden Cache
      expect(Array.from(b.prefillBatch([3, 4, 5]))).toEqual(Array.from(a.prefill([3, 4, 5])));
    }
  });

  it('Batch-Prefill über die Blockgröße hinaus (> 64 Tokens) bleibt exakt', async () => {
    const { LlamaModel: M } = await import('../src/engine/model/llama');
    const ids = Array.from({ length: 100 }, (_, i) => (i * 7) % cfg.vocab);
    const a = new M(cfg, weights, 128);
    const b = new M(cfg, weights, 128);
    expect(Array.from(b.prefillBatch(ids))).toEqual(Array.from(a.prefill(ids)));
  });

  const close = (a: Float32Array, b: number[]) => {
    let m = 0;
    for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
  };

  it('Abzweigung, Verlängerung, Verkürzung und identische Folge = Neuberechnung', () => {
    const s = new KvSessions(new LlamaModel(cfg, weights, 64), 3);
    const A = [1, 2, 3, 4, 5, 6];
    const cases: [string, number[], number][] = [
      ['Start', A, 6],
      ['Verlängerung', [...A, 7, 8], 2],
      ['Abzweigung', [1, 2, 3, 9, 10], 2],
      ['Verkürzung', [1, 2, 3], 1],
      ['identisch', [1, 2, 3], 0],
      ['ganz anders', [11, 12], 2],
    ];
    for (const [name, ids, computed] of cases) {
      expect(close(s.logits('w', ids), fresh(ids)), name).toBeLessThan(1e-5);
      expect(s.last.computed, `${name}: gerechnete Tokens`).toBe(computed);
    }
  });

  it('LRU: zwei Sitzungen im Wechsel verdrängen sich nicht, die vierte verdrängt die älteste', () => {
    const s = new KvSessions(new LlamaModel(cfg, weights, 64), 3);
    const A = [1, 2, 3, 4];
    const B = [5, 6, 7, 8];
    s.logits('a', A);
    s.logits('b', B);
    expect(close(s.logits('a', [...A, 9]), fresh([...A, 9]))).toBeLessThan(1e-5);
    expect(s.last.computed).toBe(1);
    expect(close(s.logits('b', [...B, 9]), fresh([...B, 9]))).toBeLessThan(1e-5);
    expect(s.last.computed).toBe(1);
    s.logits('c', [1]);
    s.logits('d', [2]); // verdrängt „a“ (am längsten unbenutzt)
    expect(s.has('a')).toBe(false);
    expect(s.has('b')).toBe(true);
    expect(close(s.logits('a', A), fresh(A))).toBeLessThan(1e-5);
    expect(s.last.computed).toBe(4);
  });

  it('zu lange Folge: klare Fehlermeldung, Sitzung bleibt benutzbar', () => {
    const s = new KvSessions(new LlamaModel(cfg, weights, 16), 2);
    expect(() => s.logits('x', Array.from({ length: 20 }, () => 1))).toThrow(/zu lang/);
    expect(close(s.logits('x', [1, 2]), fresh([1, 2]))).toBeLessThan(1e-5);
  });
});

describe('Auslieferungsformat (Shards)', () => {
  it.each(['attn:f16,mlp:q8g16,down:q8g16,emb:q8g16+s16', 'f16', 'q8+s16', 'f32'])(
    '%s: packen → zusammensetzen ergibt dieselben Logits (bitgleich)',
    (spec) => {
      const w = loadLlamaWeights(st, cfg, spec);
      const { tensors, data } = packWeights(w);
      // über Shard-Grenzen verteilen und wieder zusammensetzen, wie beim Download
      const ranges = shardRanges(data.byteLength, 1000);
      const buf = new Uint8Array(data.byteLength);
      for (const r of ranges) buf.set(data.subarray(r.offset, r.offset + r.bytes), r.offset);
      const u = unpackWeights(tensors, buf.buffer, cfg.layers);
      const a = new LlamaModel(cfg, w, 32);
      const b = new LlamaModel(cfg, u, 32);
      expect(Array.from(b.prefillBatch(ref.tokens))).toEqual(Array.from(a.prefillBatch(ref.tokens)));
      expect(weightBytes(u)).toBe(weightBytes(w));
    },
  );

  it('Skalen ohne fp16-Rundung werden abgelehnt (sonst wäre die Auslieferung ungenauer als gemessen)', () => {
    expect(() => packWeights(loadLlamaWeights(st, cfg, 'q8g16'))).toThrow(/s16/);
  });

  it('SHA-256 wie erwartet', async () => {
    expect(await sha256Hex(new TextEncoder().encode('abc'))).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  });
});
