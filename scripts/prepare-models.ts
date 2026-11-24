/**
 * Modell-Shards für die Auslieferung über GitHub Pages (läuft nur im CI, PLAN.md 4.3).
 *
 *   node --experimental-strip-types --max-old-space-size=6144 scripts/prepare-models.ts [--fake] [--out .cache/models]
 *
 * Lädt die gepinnten Revisionen vom Hugging Face Hub, bringt die Gewichte mit DERSELBEN Funktion
 * ins Zielformat, die auch die Engine verwendet (loadLlamaWeights → packWeights), und schreibt:
 *
 *   <out>/<modell>/config.json                       (für model-facts.json / <Fact>)
 *   <out>/<modell>/<fassung>/manifest.json           (Tensorliste, Shards mit SHA-256, Config)
 *   <out>/<modell>/<fassung>/shard-NNN.bin           (je ≤ 32 MB)
 *
 * --fake erzeugt stattdessen ein kleines Zufallsmodell (2 Layer, ≈ 37 MB, 2 Shards) mit dem echten SmolLM2-Vokabular. Es dient
 * nur dazu, Laden, Worker und Oberfläche lokal ohne Hub-Zugang zu testen (Manifest: fake = true).
 */
import { createHash } from 'node:crypto';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseSafeTensors } from '../src/engine/model/safetensors.ts';
import { configFromHf, loadLlamaWeights, type LlamaWeights } from '../src/engine/model/weights.ts';
import { packWeights, residentBytes, shardRanges, type ModelManifest } from '../src/engine/model/pack.ts';
import { linearF32, quantizeGroups, toF16Precision } from '../src/engine/model/quantize.ts';

/** Gewählte int8-Fassung aus dem Engine-Spike (docs/spike.md) */
export const INT8_SPEC = 'attn:f16,mlp:q8g32,down:q8g32,emb:q8g32+s16';
export const FP16_SPEC = 'f16';

export const MODELS = [
  {
    id: 'smollm2-135m',
    repo: 'HuggingFaceTB/SmolLM2-135M',
    revision: '93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
    editions: { int8: INT8_SPEC, fp16: FP16_SPEC },
  },
  {
    id: 'smollm2-135m-instruct',
    repo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    revision: '12fd25f77366fa6b3b4b768ec3050bf629380bac',
    editions: { int8: INT8_SPEC },
  },
] as const;

const args = process.argv.slice(2);
const fake = args.includes('--fake');
const outRoot = args.includes('--out') ? args[args.indexOf('--out') + 1] : '.cache/models';
const MB = 1024 * 1024;

async function download(repo: string, revision: string, file: string): Promise<ArrayBuffer> {
  const url = `https://huggingface.co/${repo}/resolve/${revision}/${file}`;
  for (let attempt = 1; ; attempt++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.arrayBuffer();
    } catch (e) {
      if (attempt >= 4) throw new Error(`${url}: ${(e as Error).message}`);
      console.log(`[prepare] ${file}: Versuch ${attempt} fehlgeschlagen (${(e as Error).message}), neuer Versuch …`);
      await new Promise((r) => setTimeout(r, 2000 * attempt));
    }
  }
}

const json = (buf: ArrayBuffer) => JSON.parse(new TextDecoder().decode(buf));

function writeEdition(
  dir: string,
  meta: Omit<ModelManifest, 'format' | 'version' | 'tensors' | 'totalBytes' | 'shards' | 'weightBytes'>,
  weights: LlamaWeights,
): ModelManifest {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const { tensors, data } = packWeights(weights);
  const all = createHash('sha256');
  const shards = shardRanges(data.byteLength).map((r, i) => {
    const file = `shard-${String(i).padStart(3, '0')}.bin`;
    const bytes = data.subarray(r.offset, r.offset + r.bytes);
    writeFileSync(join(dir, file), bytes);
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    all.update(sha256);
    return { file, offset: r.offset, bytes: r.bytes, sha256 };
  });
  const manifest: ModelManifest = {
    format: 1,
    ...meta,
    version: all.digest('hex').slice(0, 12),
    tensors,
    totalBytes: data.byteLength,
    shards,
    weightBytes: residentBytes(tensors, data.byteLength),
  };
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify(manifest) + '\n');
  console.log(
    `[prepare] ${meta.model}/${meta.edition}: ${shards.length} Shards, ${(data.byteLength / MB).toFixed(1)} MB Download, ` +
      `${(manifest.weightBytes / MB).toFixed(0)} MB im Speicher, Version ${manifest.version}`,
  );
  return manifest;
}

/** Winziges Zufallsmodell mit SmolLM2-Vokabular (nur lokal, zum Testen der Oberfläche) */
function fakeWeights(config: Record<string, number | boolean>): LlamaWeights {
  const cfg = configFromHf(config);
  let seed = 1;
  const rnd = (n: number, scale: number) => {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      a[i] = (seed / 2147483648 - 0.5) * 2 * scale;
    }
    return a;
  };
  const d = cfg.dModel;
  const f16 = (r: number, c: number) => linearF32(toF16Precision(rnd(r * c, 0.2)), r, c);
  const q = (r: number, c: number) => quantizeGroups(rnd(r * c, 0.2), r, c, 32, true);
  const embed = q(cfg.vocab, d);
  return {
    embed,
    lmHead: embed,
    finalNorm: new Float32Array(d).fill(1),
    layers: Array.from({ length: cfg.layers }, () => ({
      inNorm: new Float32Array(d).fill(1),
      postNorm: new Float32Array(d).fill(1),
      q: f16(cfg.heads * cfg.headDim, d),
      k: f16(cfg.kvHeads * cfg.headDim, d),
      v: f16(cfg.kvHeads * cfg.headDim, d),
      o: f16(d, cfg.heads * cfg.headDim),
      gate: q(cfg.dFF, d),
      up: q(cfg.dFF, d),
      down: q(d, cfg.dFF),
    })),
  };
}

async function main() {
  mkdirSync(outRoot, { recursive: true });
  for (const m of MODELS) {
    if (fake) {
      const config = {
        vocab_size: 49152,
        hidden_size: 576,
        num_hidden_layers: 2,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        intermediate_size: 1536,
        rms_norm_eps: 1e-5,
        rope_theta: 100000,
        max_position_embeddings: 8192,
        tie_word_embeddings: true,
      };
      const generation = { bos_token_id: 0, eos_token_id: m.id.endsWith('instruct') ? 2 : 0 };
      for (const edition of Object.keys(m.editions)) {
        writeEdition(join(outRoot, m.id, edition), { model: m.id, edition, repo: m.repo, revision: 'fake', spec: 'fake', config, generation, fake: true }, fakeWeights(config));
      }
      continue;
    }
    console.log(`[prepare] ${m.repo}@${m.revision.slice(0, 10)} wird geladen …`);
    const t0 = performance.now();
    const [configBuf, genBuf, stBuf] = await Promise.all([
      download(m.repo, m.revision, 'config.json'),
      download(m.repo, m.revision, 'generation_config.json'),
      download(m.repo, m.revision, 'model.safetensors'),
    ]);
    console.log(`[prepare] Download ${(stBuf.byteLength / MB).toFixed(0)} MB in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    const config = json(configBuf);
    const generation = json(genBuf);
    mkdirSync(join(outRoot, m.id), { recursive: true });
    writeFileSync(join(outRoot, m.id, 'config.json'), JSON.stringify(config, null, 2) + '\n');
    const cfg = configFromHf(config);
    const st = parseSafeTensors(stBuf);
    for (const [edition, spec] of Object.entries(m.editions)) {
      const weights = loadLlamaWeights(st, cfg, spec);
      writeEdition(join(outRoot, m.id, edition), { model: m.id, edition, repo: m.repo, revision: m.revision, spec, config, generation }, weights);
    }
  }
}

main().catch((e) => {
  console.error(`[prepare] Fehler: ${(e as Error).message}`);
  process.exit(1);
});
