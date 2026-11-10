# 002 — Incremental decode (1.4.0)

What is true about `tx_dec_reset` / `tx_dec_step` (`src/decode.cyr`) that the code does
not say on its own: why a step is bit-identical to the full forward, why the cache stops
at T, how a stale cache is caught, and what it costs in memory.

## What a step computes

`tx_dec_step(tok)` runs position p = `tx_dec_len()`. It embeds `tok` at p, runs the
attention sublayer for that one position, with the query against the keys and values of
positions 0..p, and appends p's own key and value to the cache. Then it runs the MLP
sublayer, the final RMSNorm and the head for that position. Its V logits land in
`ki_logits[0..V)`, and it returns p.

## Why it is bit-identical to `tx_fwd_packed`

The claim: the step's logits are the same bits as row p of `tx_fwd_packed` over the
same p + 1 tokens. The suite checks it at every window length on five models, and fuzz
property P11 on every random model. It holds because of two facts about the forward.

- **Every stage except attention is per position.** That covers the embedding, both
  RMSNorms (one row each), the per-token int8 activation quant, and every BitLinear.
  `ternary_matmul_free_simd` loops over rows, and each row uses only its own codes and
  scale. GELU and the residual adds are elementwise. A row computed alone is the same
  bits as the same row computed among T.
- **Attention is causal.** Row p reads the keys and values of rows 0..p only, so those
  are all the cache needs. The step computes p's row with `_attn_row`, the function
  `attn_fwd` runs for every row (factored out at 1.4.0; the golden dump proves the
  refactor changed no bit). Every dot product, the max, the exponential sum and the
  context sum therefore run in the same order. A second copy of that loop would have
  had to be kept in step by hand.

The cached keys and values are bit-identical to the full forward's too. Each is the
BitLinear row of position j's normed input, which does not depend on any other
position.

Rows 0..p of a forward over p + 1 tokens are rows 0..p of a forward over more tokens
(the suite has checked prefix serving since 1.1.2). So a sequence of steps also
reproduces a single full forward's rows, in order.

## The window stops at T

Positions are absolute: the position embedding has T rows, and token t gets row t. A
cache therefore holds at most T positions, and a step on a full cache fails loud
(`tx_dec_len()` tells a consumer it is full).

A sliding window cannot reuse the cache. Sliding by one token moves every token to a
new position, which changes its embedding and so every key and value after it. Past T
tokens a consumer has two choices.

- **Serve each further token with `tx_fwd_packed` over the last T.** This is what a
  sliding window has always cost, and it is bit-identical to a consumer that never used
  the decoder.
- **Reset and re-feed a shorter tail, then step again.** It is cheaper per token, but
  the context is shorter, so the output differs.

The first T tokens are where the decoder pays. At the bench config (V = 256, T = 32,
C = 128, F = 512), decoding a 32-token window costs 73.5 µs per token with steps, against
1.14 ms with a full forward per token: 15.5× (`docs/benchmarks.md`).

A prompt needs no separate prefill call. Feeding it through n steps does the same work as
one full forward over n positions, row for row. At the bench config, 32 steps take 2.35 ms
and one 32-position forward takes 2.33 ms.

## Staleness: the cache belongs to one model

A cache holds keys and values computed from one model's weights, embeddings and gains.
Every call that changes what serving reads bumps `tx_rev`, the model's revision:
`tx_init`, `tx_param_init` and `tx_sgd` (`src/block.cyr`), `tx_pack_init` and `tx_pack`
(`src/kernel.cyr`), and a successful `tx_pack_deserialize`. `tx_dec_reset` records the
revision, and `tx_dec_step` fails loud when it has changed.

The check matters most for a same-dims reload. Since 1.4.0 that load reuses the
model's buffers ([ADR 0003](../adr/0003-a-load-reuses-only-a-loaded-models-buffers.md)),
so nothing else would show that the model changed. A server that reloads while a reply
is decoding must call `tx_dec_reset` for the next reply. hoosh's epoch check, which
ends a reply when a load lands mid-reply, already does this.

The check is by revision, not by value. A `tx_sgd` with a zero learning rate still makes
the cache stale.

## Memory

The cache is two buffers of T · C f64: keys and values, one C-wide row per position.
That is `tx_dec_alloc_bytes(T, C)` = 16 · T · C bytes (65,536 at the bench config).

- The first `tx_dec_reset` allocates the cache, and so does a reset for a model whose
  T · C is larger than any before.
- Every other reset reuses it: row p sits at p · C, so any model with T · C no larger
  fits.
- A step allocates nothing. Its scratch is row 0 of the buffers `tx_fwd_packed` uses
  (`tx_resid0`, `blk_*`, `mlp_*`, `ki_*`). A step and a full forward therefore overwrite
  each other's outputs, but never the cache. The suite interleaves the two and checks
  the cache survives.

Like everything else in tentib, the cache is module-global: one sequence per process at
a time. Several concurrent sequences need model handles (the 2.0.0 backlog).
