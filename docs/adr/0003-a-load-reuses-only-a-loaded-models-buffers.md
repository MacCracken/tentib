# 0003 — A load reuses only a loaded model's buffers

**Status**: Accepted
**Date**: 2026-11-09

## Context

Through 1.3.1, every successful `tx_pack_deserialize` ran the four `*_init` calls.
rosnet's allocator never frees, so the model being replaced stayed in memory for the
life of the process. A server that reloads its config reloads its model. hoosh 2.11.0
measured 802,376 bytes kept per load at T = 30, C = 16, and works around it by
comparing SHA-256 digests to skip loads of an unchanged file. A changed model of the
same shape still cost a model's memory per reload, and so did every load at the bench
config: 12,192,728 bytes for a 357,504-byte artifact.

A load of the same V, T, C, F could overwrite the live model's buffers instead, which
needs no signature change. But a load must stay observationally a load. Serving must be
bit-identical to a fresh load's, and a refused buffer must change nothing. Any state a
fresh load would have reset must not leak through.

A fresh load leaves three groups of buffers.

- **What serving reads:** the packed store, γ, the embeddings, the gains and the bias.
  A load overwrites all of them in full, and `wsum` is recomputed.
- **Scratch:** `blk_*`, `mlp_*`, `ki_*`, `tx_resid0`, `tx_fnorm`. Every forward writes
  these before it reads them.
- **Latent weights and gradients:** `tx_Wa`, `tx_Wm`, `tx_Whead` and their `tx_d*`
  gradients, plus `tx_logits`. A fresh load leaves them as `tx_init`'s zeros, and a
  packed-only model never writes them: the paths that would are guarded. After
  `tx_param_init` the latents are drawn again, but the gradients are not.

## Decision

**A load overwrites the live model's buffers only when the live model is itself a
loaded one (`tx_packed_only`) and every init family was last sized for exactly the
artifact's V, T, C, F** (`_pk_reusable`, `src/serial.cyr`). Any other load allocates
`tx_model_alloc_bytes(V, T, C, F)` as before. The `max_bytes` check uses the model's
figure in both cases.

The condition covers the server case, a load followed by reloads. It excludes a model
built in the process, whose latent and gradient buffers hold training state a fresh load
would have zeroed. Under the condition, the third group still holds the zeros of the
first fresh load in the chain, so a reuse load leaves the process in the same state as a
fresh one. The only exceptions are scratch, and `ki_logits` holding the last serve's
logits instead of zeros.

## Consequences

- **Positive.**
  - A reload of a same-shaped model allocates nothing. A server's memory no longer
    grows with every model change, only with a change of shape.
  - The load is faster: it skips the allocation and the first touch of fresh pages.
    At the bench config it takes 4.22 ms, against 6.73 ms for a fresh load
    (`docs/benchmarks.md`).
  - Serving after it is bit-identical to a fresh load. The suite runs 40 alternating
    loads of two models, and fuzz P12 loads a second model over the first and back.
- **Negative.**
  - `tx_model_alloc_bytes` is no longer "what every load allocates". It is what a load
    allocates unless it reuses.
  - A process that trains a model and then loads files of the same shape keeps the
    trained model's buffers, and its first load allocates.
  - A load still runs every validation pass over the whole buffer, so a same-dims load
    costs about as much as the checksum and copy.
- **Neutral.** `tx_dec_step` cannot tell a same-dims reload from no reload by looking at
  the buffers, so every load bumps `tx_rev` and the decoder checks it
  ([architecture 002](../architecture/002-incremental-decode.md)). Freeable model handles,
  in the 2.0.0 backlog, would replace this rule.

## Alternatives considered

- **Reuse over any model of the same dims, zeroing the latent and gradient buffers.**
  This would also cover the train-then-load process. But the zeroing touches
  2·(4C² + 2CF + CV) + VC + TC + … slots, about 4 MB at the bench config, on every
  reload, to serve a case no consumer has. Rejected.
- **Reuse whenever the buffers are at least as large.** This would admit, for example,
  a smaller T into a larger model's buffers. But tentib's dims checks compare against
  the init dims (`tx_T0` and the rest), and those would have to change meaning: T is the
  posemb row count, and `ki_*` is sized by T as well. Exact dims keep every existing
  guard true. Rejected.
- **A separate `tx_pack_reload` entry point.** It would leave `tx_pack_deserialize`
  allocating on every call, so every consumer would have to learn a second call to stop
  the leak, for a reuse rule the loader can apply itself. Rejected.
- **Skip the `max_bytes` check when nothing would be allocated.** Whether a buffer loads
  would then depend on what was loaded before it. The cap is the consumer's statement
  of the largest model it accepts, not of the memory left. Rejected.
