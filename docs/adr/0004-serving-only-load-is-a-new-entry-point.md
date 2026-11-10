# 0004 — A serving-only load is a new entry point, not a change to `tx_pack_deserialize`

**Status**: Accepted
**Date**: 2026-11-09

## Context

`tx_pack_deserialize` loads a model by running `tx_init`, `tx_int_init` and `tx_pack_init`, so a
loaded model has every buffer an in-process model has: the latent weights, their gradients, the
block's training scratch and the integer path's K×N scratch. Serving reads none of them. At the
bench config (V = 256, T = 32, C = 128, F = 512) a load allocates 12,192,728 bytes, and serving
needs 1,434,536 of them. On the Raspberry Pi 4 that is the gap between a model that fits its 2 GB
and one that does not. The roadmap's 1.5.x band planned "a serving-only allocation for loaded
models", additive, with the artifact format unchanged.

The 1.x freeze says a minor adds symbols and never changes a frozen signature or its
semantics. What `tx_pack_deserialize` allocates is documented semantics, in three places:
- `api.md` says a successful load "runs `tx_init` + `tx_int_init` + `tx_pack_init`, exactly
  `tx_model_alloc_bytes(V, T, C, F)` bytes";
- the format doc's decoder table checks `tx_model_alloc_bytes ≤ max_bytes`;
- since 1.4.0, a same-dims reload reuses those buffers.

Consumers can see it:
- **hoosh's suite** asserts that a load of another shape keeps `tx_model_alloc_bytes` on the heap.
- **Some calls work after a load today and need the training buffers:**
  - `tx_param_init`, which "makes an ordinary model again";
  - `tx_zero_grads`;
  - `block_fwd` / `block_bwd`, on the block scratch;
  - `bl_forward_q`, on the integer K×N scratch.

  With those buffers gone, they would crash or have to start refusing.

## Decision

**Serving-only loading is a new entry point, `tx_pack_deserialize_serving(buf, n, max_bytes)`,
with its own memory figure, `tx_serve_alloc_bytes(V, T, C, F)`. `tx_pack_deserialize` is
unchanged, byte for byte.**

- **The two share one loader (`_pk_load`).** They read the same bytes, run the same checks in
  the same order, return the same status codes, and copy in the same model. Only the memory
  figure the `max_bytes` check uses and the allocation differ.
- **A serving load allocates exactly `tx_serve_alloc_bytes`:** the packed store, the
  parameters serving reads, and the scratch one forward writes. It sets every other buffer
  pointer of those families to 0, so nothing can use a stale buffer silently.
- **Every call that needs a missing buffer refuses the model, loud:**
  - the latent-weight paths, which already refuse any loaded model;
  - `tx_param_init` and `tx_zero_grads`, through a new `tx_serve_only` flag;
  - `block_fwd` / `block_bwd`, because the block's dims are left at 0;
  - `bl_forward_q`, because the integer scratch is sized with no K×N room.

  `tx_init` makes an ordinary model again.
- **A serving load over a model loaded for serving with the same dims allocates nothing.** It
  is the same reuse rule as 1.4.0's ADR 0003, for the serving family. A serving load and a full
  load never reuse each other's buffers.

## Consequences

- **Positive:**
  - A server allocates 8.5× less per load at the bench config.
  - A fresh load is faster: 6.25 → 4.24 ms on x86_64, and 81.6 → 36.9 ms on the Pi 4.
  - Serving is bit-identical: the suite checks it at three shapes, decoding included, and fuzz
    P13 and P10 check it on random models and mutated artifacts.
  - Nothing an existing consumer does changes: the golden dump is byte-identical, and hoosh
    2.11.1's suite passes against it unchanged.
- **Negative:**
  - Two loaders to document, and a consumer has to choose.
  - A serving-only model is a third model state, beside in-process and fully loaded, with its
    own guard set. The suite covers each refused call in a forked child.
- **Neutral:**
  - hoosh adopts it by switching one call, a hoosh change.
  - Model handles (2.0.0) would make one load the only load, sized for what it is used for.

## Alternatives considered

- **Change what `tx_pack_deserialize` allocates.** The roadmap's original wording. Rejected
  under the freeze: it changes documented semantics and fails hoosh's memory test. It also turns
  `tx_param_init`, `tx_zero_grads`, `block_fwd` and `bl_forward_q` after a load from working into
  crashing, or into new refusals.
- **A mode switch** (`tx_pack_set_load_mode(serving)`), read by `tx_pack_deserialize`. It is
  additive, but it is hidden global state that changes what a frozen call does. A reader of
  the call site cannot tell what it allocates. Rejected for an explicit entry point.
- **Allocate the training buffers lazily,** for example in `tx_param_init` after a serving
  load. That keeps every call working, but it puts allocation in calls documented as
  allocation-free, and it hides the cost. Rejected: `tx_init` already exists for exactly that.
