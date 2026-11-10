# tentib — Public API (frozen at v1.0.0; settled at v0.6.0)

> Satisfies the v1.0 criterion *"Public API frozen — every exported symbol
> documented and tested."*

## Freeze policy

The **stable public API** below was settled at v0.6.0 and is **frozen for the
1.x series as of the 1.0.0 cut**: signatures and semantics will not change
without a major version bump; minor versions may **add** symbols. Every public symbol is exercised
**directly** by at least one committed driver — the demo (`src/main.cyr`), the suite
(`tests/tentib.tcyr`, 139 assertions at 1.1.2), or the bench harness (`tests/tentib.bcyr`) — noted per
row. The 0.6.0 suite group "API-freeze gates" closes the last indirect-only cases
(`ref_dot`, `bl_forward_q`, the manual `tx_sgd` quartet).

**Not frozen (internal mechanism):** the STE masks + FD probes
(`bl_ste_mask_w/x`, `bl_surr_lin_loss`, `bl_loss_of_x`, `bl_quant_loss_wonly`,
`rms_loss`, `gelu_loss`, `attn_loss`, `block_loss`, `bl_xent_loss`), numeric
helpers (`_rms_eps`, `_tanh`, `_gelu_c`, `_gelu_a`), the head-LM internals
(`lm_init`, `lm_eval_sweep`, `lm_correct` — `lm_train`-local buffers), the
sublayer chains (`attn_sublayer_fwd/loss/bwd`, `mlp_sublayer_fwd/loss/bwd`,
`attn_sublayer_q`, `mlp_sublayer_q`, `block_q`), and every `blk_*`/`mlp_*`
scratch global. These are the reference's *mechanism* — readable and FD-gated,
but free to change between minor versions. (1.1.2: `bl_ste_mask_x` masks where
the forward's clamp changed the code, `|x/s| ≥ 127.5`; with `bl_act_quant`'s
scale that is never, so the mask is the identity. The `blk_db` / `mlp_db`
scratch is gone: the bias-free projections pass `db = 0`.)

## Conventions & caller contract

- **Everything is i64.** f64 values travel as raw bit patterns in i64; "→ f64"
  below means "returns f64 bits". Tensors are heap pointers to flat row-major f64
  runs; element access via `tload`/`tget` and `tstore`/`tset`.
- **No bounds checks** (the rosnet substrate contract): callers guarantee buffer
  sizes. Sizing formulas are given per `*_init`.
- **Bias arguments:** pass `b = 0` for "no bias" — any nonzero pointer is read.
- **Ternary weights** `Wq` are i64 arrays holding {−1, 0, +1}; activations
  quantize per-row to int8 codes in [−127, +127] (also stored as i64).
- **Guarded preconditions (0.8.0):** violations that would mean silent corruption
  or silently-wrong numerics **fail loud** via `guard()` — non-ternary weights at
  pack time (`tpack2`, `tsimd_pack_w`), K > 2²² in `ternary_matmul_free_simd`
  (the i32 exactness bound), and packed-store misuse (`tx_pack` before its inits,
  `tx_fwd_packed` before `tx_pack`; since 1.4.0 a `tx_dec_step` on a cache older than
  the model, on a full cache, or before any packed model; since 1.5.0 every call that needs
  a training buffer, on a model loaded for serving only). Cold paths only; hot loops
  are unchecked.
  Audit record: [`audit/2026-07-06-audit.md`](audit/2026-07-06-audit.md).
- **Ids and dims (1.1.2):** token ids (`tx_fwd`, `tx_fwd_q`, `tx_fwd_packed`,
  `tx_bwd`) and target ids (`softmax_xent_fwd`/`_bwd`) must lie in [0, V); each
  `*_init` records its dims, and later calls must pass the same V, C, F and a T
  in [1, init T] (a shorter T serves the prefix's rows, bit for bit), after the
  init ran. `bl_forward_q` checks M, K, N against the `tx_int_init` sizing.
  Violations fail loud; these are checked once per call, O(T) at most.
- **Failure channel:** `guard()` writes `tentib guard: <reason>` to stderr and
  exits 1. tentib's library code calls no consumer printer (a consumer defines
  only akshara's four hooks). Audit record:
  [`audit/2026-11-09-audit.md`](audit/2026-11-09-audit.md); posture:
  [ADR 0002](adr/0002-hardening-preserves-bits.md).

## Allocation & lifetime (the 0.6.0 alloc-clean audit)

`t_alloc` is rosnet's bump allocator — **it never frees**. The audit walked every
path; the invariant is:

- **Allocation happens ONLY in** `blk_init` / `tx_init` / `tx_int_init` /
  `tx_pack_init` (the `*_init` pattern), **at the top of the one-shot trainers**
  (`lm_train`, `tx_train` — which begins with `tx_init`), **in a successful
  `tx_pack_deserialize`** (1.2.0: it runs `tx_init` + `tx_int_init` + `tx_pack_init`,
  exactly `tx_model_alloc_bytes(V, T, C, F)` bytes; since 1.4.0 **not** when it replaces
  a loaded model of the same V, T, C, F, whose buffers it overwrites), **in a successful
  `tx_pack_deserialize_serving`** (1.5.0: exactly `tx_serve_alloc_bytes(V, T, C, F)`, what
  serving reads; nothing when it replaces a model loaded for serving with the same dims), and
  **in the first `tx_dec_reset`** (1.4.0: the decode cache, `tx_dec_alloc_bytes(T, C)`, again
  only for a model larger than any before). Each such call allocates fresh buffers;
  call them a **bounded number of times per process**.
- **Every forward, backward, and kernel path is allocation-free** and reuses the
  init-owned (or caller-passed) scratch: `tx_fwd`, `tx_fwd_q` (all modes),
  `bl_forward_*`, `block_fwd/bwd`, all `ternary_matmul_free*` kernels,
  `act_quant_int`, `tsimd_pack_w`, `tpack2`/`tunpack2`, and (1.4.0) `tx_dec_step`.
  Repeated inference costs zero allocation.
- Module-global scratch families: `blk_*`/`mlp_*` (owned by `blk_init`), `tx_*`
  (owned by `tx_init`), `ki_*` (owned by `tx_int_init`). Re-calling an `*_init`
  re-points the family at fresh buffers (the old ones are unreachable, not freed).

## M0 — ternary core (`src/ternary.cyr`)

| symbol | contract | tested by |
|--------|----------|-----------|
| `tload(p, i)` → f64 / `tstore(p, i, v)` | element access on a flat f64 tensor | suite (throughout) |
| `guard(cond, msg)` | fail-loud precondition guard (message on stderr, exit 1) — the 0.8.0 hardening primitive, tarka convention | suite (pass path; 20 fail paths in forked children) |
| `absmean(w, n)` → f64 | the b1.58 scale γ = mean(abs(w)); 0 for n ≤ 0 | demo, suite, bench |
| `ternary_one(w, gamma)` → i64 | one weight → {−1, 0, +1} = clip(round(w/γ)) | suite |
| `ternary_quantize(w, n, Wq)` → f64 | quantize n weights into `Wq`; returns γ | demo, suite, bench |
| `ternary_dot(Wq, x, n, gamma)` → f64 | matmul-free dot: signed-accumulate · γ | demo, suite |
| `ref_dot(w, x, n)` → f64 | full-precision baseline dot | demo, suite |

## M1 — BitLinear + STE (`src/bitlinear.cyr`)

| symbol | contract | tested by |
|--------|----------|-----------|
| `bl_weight_effective(W, n, Wq, Weff)` → f64 | latent → ternary `Wq` + dequant `Weff` = γ·Wq; returns γ | demo, suite, bench |
| `bl_act_quant(x, M, K, xq, sx)` | per-row absmax int8 quant → **dequantized f64** `xq` + scales `sx`; a NaN in a row makes that row's scale NaN | suite |
| `bl_forward_wonly(x, W, b, y, M, K, N, Wq, Weff)` → f64 | BitLinear, weight-only quant (f64 matmul on `Weff`); returns γ | suite |
| `bl_forward_full(x, W, b, y, M, K, N, Wq, Weff, xe, sx)` → f64 | BitLinear, weights + activations quantized (the integer kernel's f64 reference); `xe` = dequantized acts | demo, suite, bench |
| `bl_backward_wonly(x, Weff, du, dx, dW, db, M, K, N, W, gamma, scr)` | STE backward: `dW` through the clip-mask surrogate, `dx` through `Weff` | suite (FD-gated) |

## M2 — layers, losses, the trainable LMs (`src/layers.cyr`, `src/model.cyr`, `src/block.cyr`)

| symbol | contract | tested by |
|--------|----------|-----------|
| `rmsnorm_fwd(x, g, y, C)` → f64 | pre-norm RMSNorm row; returns rstd (feed to bwd) | suite (FD-gated) |
| `rmsnorm_bwd(x, g, dy, dx, dg, C, rstd)` | RMSNorm backward | suite (FD-gated) |
| `gelu_fwd(x, y, n)` / `gelu_bwd(x, dy, dx, n)` | tanh-GELU elementwise fwd/bwd | suite (FD-gated) |
| `attn_fwd(Q, K, V, P, ctx, T, C)` | single-head causal scaled-dot-product attention; `P` = T×T probs | suite (FD-gated) |
| `attn_bwd(Q, K, V, P, dctx, dQ, dK, dV, dprow, T, C)` | attention backward | suite (FD-gated) |
| `softmax_xent_fwd(logits, targets, probs, T, V)` → f64 | mean next-token CE; fills `probs` | suite (FD-gated) |
| `softmax_xent_bwd(probs, targets, dlogits, T, V)` | CE backward (probs − onehot)/T | suite |
| `lm_train(V, D, steps, seed, stats)` | the M2 head-LM: embedding → BitLinear head → CE, SGD; `stats` = [CE₀, CE_final, argmax-correct] | demo, suite |
| `blk_init(T, C, F)` | allocate the block scratch (`blk_*`/`mlp_*`) — call before any block/tx use (`tx_init` calls it) | suite, bench (via `tx_init`) |
| `block_fwd(x, gns, Wa, Wm, T, C, F)` | full pre-norm block (attn + MLP sublayers); output lands in `mlp_out` | suite (FD-gated) |
| `block_bwd(dy, x, gns, Wa, Wm, dx, dgns, dWa, dWm, T, C, F)` | block backward | suite (FD-gated) |
| `tx_init(V, T, C, F)` | allocate the transformer params + grads + scratch (`tx_*`) | suite, bench |
| `tx_param_init(V, T, C, F, seed)` | randn-init params (std 0.1), gains = 1, bias = 0 | suite, bench |
| `tx_fwd(tokens, V, T, C, F)` | whole-model f64 forward (weight-only-quant linears); logits → `tx_logits` | demo, suite, bench |
| `tx_loss(tokens, targets, V, T, C, F)` → f64 | `tx_fwd` + CE (fills `tx_probs`) | suite |
| `tx_zero_grads(V, T, C, F)` / `tx_bwd(tokens, targets, V, T, C, F)` | zero / accumulate the full LM gradients (call `tx_loss` first) | suite (FD-gated) |
| `tx_sgd(V, T, C, F, nlr)` | one SGD step on latents + embeds/gains/bias; `nlr` is the **negative** learning rate | suite (quartet gate) |
| `tx_train(tokens, targets, V, T, C, F, steps, seed, stats)` | init + train `steps`; `stats` = [CE₀, CE_final] | demo, suite |

The manual loop `tx_zero_grads → tx_loss → tx_bwd → tx_sgd` is public and gated
(the 0.6.0 quartet test); `tx_train` is that loop plus init.

## M3 + 0.4.1 — the matmul-free integer kernels (`src/kernel.cyr`)

| symbol | contract | tested by |
|--------|----------|-----------|
| `act_quant_int(x, M, K, qx, sx)` | per-row absmax int8 quant keeping **integer codes** `qx` ∈ [−127, +127] + scales `sx`; a NaN in a row makes that row's scale NaN, so its outputs are NaN in every kernel | demo, suite, bench |
| `ternary_matmul_free(qx, sx, Wq, gamma, b, y, M, K, N)` | the scalar kernel: i64 signed-accumulate (add/sub/skip, **no multiply**), one dequant (γ·sx) + bias per output | demo, suite, bench |
| `ternary_matmul_free_bl(...)` (same signature) | branchless variant (mask arithmetic) — **bit-identical** to the branchy kernel | demo, suite, bench |
| `tpack2(Wq, n, packed)` / `tunpack2(packed, i)` → i64 | 2-bit packed ternary storage (32× vs f64), exact round-trip | demo, suite |
| `tsimd_pack_w(Wq, K, N, w8t, wsum)` | **pack-once** for the SIMD kernel: transpose to `w8t` [N,K] i8 bytes + per-output `wsum` over the 16-lane span (the u8-offset correction). `w8t` = K·N bytes, `wsum` = N i64 slots | demo, suite, bench |
| `ternary_matmul_free_simd(qx, sx, w8t, wsum, gamma, b, y, M, K, N, xu8)` | the 0.4.1 `iv_dp8` kernel — **bit-identical** to the scalar kernel (never saturates for ternary weights: pair-sums ≤ 510; iv_dp8 sums in i32, so exact for K < ~8.4M, guarded at K ≤ 2²²). Codes `qx` must lie in [−128, +127] (unchecked; `act_quant_int` gives [−127, +127]). `xu8` = K bytes scratch. Any K (16-lane span + branchless tail) | demo, suite, bench |
| `tx_int_init(V, T, C, F)` | allocate the whole-model integer scratch (`ki_*`) — call after `tx_init` | demo, suite, bench |
| `bl_forward_q(mode, x, W, b, y, M, K, N)` → f64 | one BitLinear via mode **0** = scalar integer, **1** = full-quant f64 reference, **2** = integer SIMD; re-quantizes (mode 2: + re-packs) per call; returns γ | suite (direct gate) |
| `tx_fwd_q(mode, tokens, V, T, C, F)` | whole-model inference through all 7 BitLinears in the chosen mode; logits → `ki_logits`. Modes 0 and 2 are bit-identical; mode 1 is the f64 rounding reference | demo, suite, bench |
| `logits_argmax(logits, V)` → i64 | argmax over one logits row (ties → first) | demo, suite |

## 0.7.0 — pack-once deployment inference (`src/kernel.cyr`, additive)

The consumer serving path (`tx_fwd_q` re-quantizes per call by reference-fairness
design; a deployment packs once). Contract: call after `tx_init` (weights live)
**and** `tx_int_init` (packing quantizes through `ki_Wq`/`ki_We`; serving
activation-quants into `ki_q`/`ki_sx`). Worked example:
[`examples/quickstart.cyr`](../examples/quickstart.cyr) (public-API-only,
self-checking — running it is a test).

| symbol | contract | tested by |
|--------|----------|-----------|
| `tx_pack_init(V, T, C, F)` | allocate the packed store (7 per-layer i8 weight buffers + `wsum` tables + γs) | suite, bench, example |
| `tx_pack(V, T, C, F)` | quantize + pack all 7 BitLinears from the live `tx_*` latents — **once**; ~8 ns/weight (incl. the 0.8.0 guards); fails loud if the inits haven't run | suite, bench, example |
| `tx_fwd_packed(tokens, V, T, C, F)` | serving forward from the packed store — per call only the per-token activation quant runs, weights untouched; logits → `ki_logits`. **Bit-identical** to `tx_fwd_q` modes 0/2 (suite-gated, twice — the store is serve-time read-only) | suite, bench, example |

Internal (not frozen): `bl_forward_pk`, `attn_sublayer_pk`, `mlp_sublayer_pk`,
`pk_*` store globals.

## 1.2.0 — packed-model serialization (`src/serial.cyr`, additive)

A packed model as bytes: write the live packed store with `tx_pack_serialize`, and
later (another process) load it with `tx_pack_deserialize` and serve it through
`tx_fwd_packed`, bit-identically. tentib does no file I/O; the consumer owns the
bytes. The format, every check the decoder makes, and its invariants are specified
in [`architecture/001-packed-model-format.md`](architecture/001-packed-model-format.md).
Worked example: [`examples/quickstart.cyr`](../examples/quickstart.cyr) step 5.

| symbol | contract | tested by |
|--------|----------|-----------|
| `tx_pack_serial_bytes()` → i64 | bytes `tx_pack_serialize` writes for the live packed model (fails loud before `tx_pack` / a load) | suite, bench, example, demo |
| `tx_pack_serialize(buf, cap)` → i64 | write it to `buf`; returns the bytes written, **−1** if `cap` is too small, **−2** if the model holds a non-finite parameter or an invalid γ (nothing written in either case) | suite, fuzz, bench, example, demo |
| `tx_pack_deserialize(buf, n, max_bytes)` → i64 | load a model from `buf[0..n)`, allocating at most `max_bytes`; returns **0**, or a negative status below, in which case nothing was allocated or changed | suite, fuzz, bench, example, demo |
| `tx_model_alloc_bytes(V, T, C, F)` → i64 | the bytes a model of these dims allocates (the four `*_init` calls; what a load allocates); −1 for a dim outside [1, 2²⁰] | suite (against `alloc_used()`) |
| `tx_pack_dims(dims)` | **1.3.0.** writes the live packed model's V, T (the longest context it serves), C, F to `dims[0..4)` — the `tx_fwd_packed` arguments of a model a consumer loaded but did not size; fails loud when no packed model is live | suite, example (`examples/serve.cyr`) |
| `pk_err_length()` … `pk_err_param()` → i64 | the status codes: `length` −1, `magic` −2, `version` −3, `header` −4, `checksum` −5, `too_big` −6, `weights` −7, `gamma` −8, `param` −9 | suite (each one), fuzz |
| `tx_pack_peek(buf, n, info)` → i64 | **1.4.0.** runs the load's header checks without loading; on success writes `info[0..5)` = V, T, C, F and the declared vocabulary and returns **0**, else returns the `pk_err_*` code the load would return for that header and writes nothing. Changes no state. A 0 does not promise the load succeeds (`max_bytes`, the checksum and every field are checked by the load) | suite, fuzz (P10: agrees with the load), example |
| `tx_pack_vocab()` → i64 | **1.4.0.** the vocabulary the live packed model declares: `pk_vocab_none()` or `pk_vocab_bytes()`; a load sets it from the artifact's flags word; fails loud when no packed model is live | suite, fuzz |
| `tx_pack_set_vocab(kind)` | **1.4.0.** declares the packed model's vocabulary for `tx_pack_serialize` to record: `pk_vocab_bytes()` (token id = byte value; needs V = 256) or `pk_vocab_none()`. `tx_pack_init` resets it to none and `tx_pack` keeps it. Fails loud on another kind, on bytes with V ≠ 256, or before `tx_pack_init` | suite, fuzz |
| `pk_vocab_none()`, `pk_vocab_bytes()` → i64 | **1.4.0.** the vocabulary kinds, 0 and 1 (flags bit 0) | suite |
| `tx_pack_deserialize_serving(buf, n, max_bytes)` → i64 | **1.5.0.** `tx_pack_deserialize` for a server: the same bytes, checks, status codes and served logits (bit for bit), but it allocates only what serving reads, at most `max_bytes` measured against `tx_serve_alloc_bytes`. The model serves, decodes, reports its dims and vocabulary and serializes; every call that needs a training buffer refuses it (below). A serving load over a model loaded for serving with the same dims allocates nothing | suite, fuzz (P10, P13), bench, example |
| `tx_serve_alloc_bytes(V, T, C, F)` → i64 | **1.5.0.** the bytes a serving load allocates: the packed store, the embeddings, gains and bias, and one forward's scratch (1,434,536 at the bench config, against `tx_model_alloc_bytes`' 12,192,728); −1 for a dim outside [1, 2²⁰] | suite (against `alloc_used()`), bench, example |

**Decoding is an untrusted-input surface.** Bad *data* is a status code, never a
`guard()` exit; the memory a load may take is the caller's `max_bytes`. A loaded
model is **packed-only**: it has no latent weights, so `tx_fwd`, `tx_fwd_q`,
`tx_bwd`, `tx_sgd` and `tx_pack` fail loud on it, while `tx_fwd_packed` serves any
T up to the stored T. `tx_init` or `tx_param_init` makes an ordinary model again.

**Loading for serving only (1.5.0).** `tx_pack_deserialize_serving` is the load for a
server; `tx_pack_deserialize` is unchanged and still allocates the full model. A
serving-only model has no latent weights, gradients, block training scratch or K×N integer
scratch, so these calls refuse it (fail loud):
- `tx_fwd`, `tx_fwd_q`, `tx_bwd`, `tx_sgd` and `tx_pack`, as on any loaded model;
- `tx_param_init` and `tx_zero_grads` (new guards, on this state only);
- `block_fwd` / `block_bwd`;
- `bl_forward_q`.

`tx_init` makes an ordinary model again, and `tx_int_init` / `blk_init` re-provide their
families. Why a new entry point:
[ADR 0004](adr/0004-serving-only-load-is-a-new-entry-point.md).

**Reloading (1.4.0).** A load that replaces a *loaded* model of the same V, T, C, F
overwrites its buffers and allocates nothing; any other load allocates
`tx_model_alloc_bytes(V, T, C, F)` as before. Serving after either is bit-identical.
A model built in the process (`tx_init`, `tx_param_init`, `tx_pack`) is never
overwritten. The `max_bytes` check uses the model's figure either way, so whether a
buffer loads does not depend on what was loaded before. The rule and its alternatives:
[ADR 0003](adr/0003-a-load-reuses-only-a-loaded-models-buffers.md).

**The vocabulary flag (1.4.0).** Flags bit 0 declares the byte vocabulary. The format
stays version 1, and a model with none declared writes the bytes 1.2.0 – 1.3.1 wrote.
1.2.0 – 1.3.1 decoders accept only flags 0, so they refuse a flagged artifact with
`pk_err_header`.

## 1.3.0 — consuming tentib as a library (`dist/tentib.cyr`)

`cyrius distlib --all` bundles the library modules (`src/ternary.cyr` through
`src/serial.cyr`, in include order) into `dist/tentib.cyr`, and the AGNOS GPU tile
path into `dist/tentib-gpu.cyr`. A consumer declares `[deps.tentib]` with
`modules = ["dist/tentib.cyr"]`; `cyrius deps` brings rosnet and tyche with it, and
not akshara (a test-scope dependency of tentib's). The bundle's requires block
carries the one stdlib leaf it needs (`math`), and it needs no consumer-defined hook.
How-to: [`guides/consuming-tentib.md`](guides/consuming-tentib.md); the CI-gated
example is [`examples/serve.cyr`](../examples/serve.cyr) (built with `--no-deps`,
from the bundles alone; one process saves a model file, a second loads and serves it).

## 1.4.0 — incremental decode (`src/decode.cyr`, additive)

A server generates one token at a time. `tx_fwd_packed` recomputes the whole window
for each token; the decoder runs one new position against a cache of the earlier
positions' keys and values. The constraints (why it is bit-identical, the window,
staleness, memory) are [`architecture/002-incremental-decode.md`](architecture/002-incremental-decode.md).

| symbol | contract | tested by |
|--------|----------|-----------|
| `tx_dec_reset()` | start a sequence on the live packed model (the cache holds no positions). The first call allocates the cache, `tx_dec_alloc_bytes(T, C)`, and so does a call for a model larger than any before; any other call allocates nothing. Fails loud when no packed model is live | suite, fuzz, bench, example |
| `tx_dec_step(tok)` → i64 | run position p = `tx_dec_len()` for token id `tok` and append its keys and values; returns p, logits → `ki_logits[0..V)`. **Bit-identical** to row p of `tx_fwd_packed` over the same p + 1 tokens (suite at every length; fuzz P11). Fails loud when no packed model is live, when the cache is older than the model (any `tx_init`, `tx_param_init`, `tx_sgd`, `tx_pack_init`, `tx_pack` or load since the reset), when the cache holds T positions, or on an id outside [0, V) | suite, fuzz, bench, example |
| `tx_dec_len()` → i64 | positions in the cache, 0..T | suite, fuzz, example |
| `tx_dec_alloc_bytes(T, C)` → i64 | the cache's bytes: keys and values, T·C f64 each (16·T·C); −1 for a dim outside [1, 2²⁰] | suite (against `alloc_used()`), bench |

Internal (not frozen): `dec_*` globals, `tx_rev` (the model revision the staleness
check reads), `_attn_row` / `_attn_scale` (`src/layers.cyr`; `attn_fwd` runs every
row through `_attn_row`, the step runs its one row).

**Internal in 1.5.1 (not frozen): the 2-bit weight store.**
- **Names:** `_tmf_simd2`, `_pk2_from_w8t`, `_pk2_row_bytes`, `_pk_lut_build`,
  `_tx_pack_deserialize_compact`, `_tx_compact_alloc_bytes`, `pk_kind` / `pk_w2t` /
  `pk_xu8m` / `pk_wrow` / `pk_lut`.
- **What a compact load gives:** a serving-only model whose weights are 2 bits each (a
  quarter of the i8 copy). Every public serving call (`tx_fwd_packed`, `tx_dec_step`,
  `tx_pack_serialize`) returns the same bits from it.
- **Status:** it is a measurement vehicle until a minor decides to expose it
  (`docs/benchmarks.md`'s 1.5.1 note has the trade). Its names and layout may change before
  then.

## Documented output cells (module globals, read-only for consumers)

| global | written by | contents |
|--------|-----------|----------|
| `tx_logits` | `tx_fwd` / `tx_loss` | T×V f64 logits (f64 path) |
| `ki_logits` | `tx_fwd_q` / `tx_fwd_packed` | T×V f64 logits (integer paths) |
| `ki_logits` | `tx_dec_step` (1.4.0) | the new position's V logits, in row 0 |
| `mlp_out` | `block_fwd` | T×C block output (feeds the final norm) |

## Performance contract

See [`benchmarks.md`](benchmarks.md): the SIMD kernel is pack-once-then-hot —
`tsimd_pack_w` costs ~4 ns/weight once; the kernel then beats the f64-SIMD matmul
5–18× (growing with layer size). `bl_forward_q`/`tx_fwd_q` re-quantize per call by
design (reference fairness); **the pack-once deployment entry point shipped in
0.7.0 (additive)** — `tx_pack` + `tx_fwd_packed` serve the whole model at **5.7×
the f64 path** (13,675 vs 2,378 tok/s at the bench config, 0.8.0-guarded run),
the one-time pack amortizing after ~3 forwards. (Those are the 2026-07-06 run of
record's figures; the 1.1.1 and 1.1.2 changes are in [`benchmarks.md`](benchmarks.md)'s
dated notes, the CHANGELOG and `bench-history.csv`.)
