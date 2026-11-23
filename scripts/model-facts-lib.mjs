/**
 * Leitet Kennzahlen aus einer Llama-artigen config.json (Hugging Face) ab.
 * Parameterzählung: Embedding (V·d, ggf. getrennter LM-Head) + pro Layer
 * Attention (q, k, v, o) + MLP (gate, up, down) + 2 RMSNorms, plus finale Norm.
 */
export function llamaFacts(c) {
  const d = c.hidden_size;
  const layers = c.num_hidden_layers;
  const heads = c.num_attention_heads;
  const kvHeads = c.num_key_value_heads ?? heads;
  const headDim = c.head_dim ?? d / heads;
  const inter = c.intermediate_size;
  const vocab = c.vocab_size;
  const tied = c.tie_word_embeddings ?? false;

  const embed = vocab * d;
  const attn = d * heads * headDim + 2 * d * kvHeads * headDim + heads * headDim * d;
  const mlp = 3 * d * inter;
  const norms = 2 * d;
  const perLayer = attn + mlp + norms;
  const params = embed + layers * perLayer + d + (tied ? 0 : vocab * d);

  return {
    params,
    layers,
    d_model: d,
    heads,
    kv_heads: kvHeads,
    head_dim: headDim,
    d_ff: inter,
    vocab,
    context: c.max_position_embeddings,
    params_embedding: embed,
    params_per_layer: perLayer,
    tied_embeddings: tied ? 'ja' : 'nein',
  };
}
