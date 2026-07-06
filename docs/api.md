# tentib — Public API (settled at v0.6.0; frozen through the 1.x series)

> Satisfies the v1.0 criterion *"Public API frozen — every exported symbol
> documented and tested."*

## Freeze policy

The **stable public API** below is settled as of v0.6.0 and is frozen for the 1.x
series at the 1.0.0 cut: signatures and semantics will not change without a major
version bump; minor versions may **add** symbols. Every public symbol is exercised
**directly** by at least one committed driver — the demo (`src/main.cyr`), the suite
(`tests/tentib.tcyr`, 95/95), or the bench harness (`tests/tentib.bcyr`) — noted per
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
but free to change between minor versions.

## Conventions & caller contract

- **Everything is i64.** f64 values travel as raw bit patterns in i64; "→ f64"
  below means "returns f64 bits". Tensors are heap pointers to flat row-major f64
  runs; element access via `tload`/`tget` and `tstore`/`tset`.
- **No bounds checks** (the rosnet substrate contract): callers guarantee buffer
  sizes. Sizing formulas are given per `*_init`.
- **Bias arguments:** pass `b = 0` for "no bias" — any nonzero pointer is read.
- **Ternary weights** `Wq` are i64 arrays holding {−1, 0, +1}; activations
  quantize per-row to int8 codes in [−127, +127] (also stored as i64).

## Allocation & lifetime (the 0.6.0 alloc-clean audit)

`t_alloc` is rosnet's bump allocator — **it never frees**. The audit walked every
path; the invariant is:

- **Allocation happens ONLY in** `blk_init` / `tx_init` / `tx_int_init` (the
  `*_init` pattern) **and at the top of the one-shot trainers** (`lm_train`,
  `tx_train` — which begins with `tx_init`). Each such call allocates fresh
  buffers; call them a **bounded number of times per process**.
- **Every forward, backward, and kernel path is allocation-free** and reuses the
  init-owned (or caller-passed) scratch: `tx_fwd`, `tx_fwd_q` (all modes),
  `bl_forward_*`, `block_fwd/bwd`, all `ternary_matmul_free*` kernels,
  `act_quant_int`, `tsimd_pack_w`, `tpack2`/`tunpack2`. Repeated inference costs
  zero allocation.
- Module-global scratch families: `blk_*`/`mlp_*` (owned by `blk_init`), `tx_*`
  (owned by `tx_init`), `ki_*` (owned by `tx_int_init`). Re-calling an `*_init`
  re-points the family at fresh buffers (the old ones are unreachable, not freed).

## M0 — ternary core (`src/ternary.cyr`)

| symbol | contract | tested by |
|--------|----------|-----------|
| `tload(p, i)` → f64 / `tstore(p, i, v)` | element access on a flat f64 tensor | suite (throughout) |
| `absmean(w, n)` → f64 | the b1.58 scale γ = mean(abs(w)) | demo, suite, bench |
| `ternary_one(w, gamma)` → i64 | one weight → {−1, 0, +1} = clip(round(w/γ)) | suite |
| `ternary_quantize(w, n, Wq)` → f64 | quantize n weights into `Wq`; returns γ | demo, suite, bench |
| `ternary_dot(Wq, x, n, gamma)` → f64 | matmul-free dot: signed-accumulate · γ | demo, suite |
| `ref_dot(w, x, n)` → f64 | full-precision baseline dot | demo, suite |

## M1 — BitLinear + STE (`src/bitlinear.cyr`)

| symbol | contract | tested by |
|--------|----------|-----------|
| `bl_weight_effective(W, n, Wq, Weff)` → f64 | latent → ternary `Wq` + dequant `Weff` = γ·Wq; returns γ | demo, suite, bench |
| `bl_act_quant(x, M, K, xq, sx)` | per-row absmax int8 quant → **dequantized f64** `xq` + scales `sx` | suite |
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
| `act_quant_int(x, M, K, qx, sx)` | per-row absmax int8 quant keeping **integer codes** `qx` ∈ [−127, +127] + scales `sx` | demo, suite, bench |
| `ternary_matmul_free(qx, sx, Wq, gamma, b, y, M, K, N)` | the scalar kernel: i64 signed-accumulate (add/sub/skip, **no multiply**), one dequant (γ·sx) + bias per output | demo, suite, bench |
| `ternary_matmul_free_bl(...)` (same signature) | branchless variant (mask arithmetic) — **bit-identical** to the branchy kernel | demo, suite, bench |
| `tpack2(Wq, n, packed)` / `tunpack2(packed, i)` → i64 | 2-bit packed ternary storage (32× vs f64), exact round-trip | demo, suite |
| `tsimd_pack_w(Wq, K, N, w8t, wsum)` | **pack-once** for the SIMD kernel: transpose to `w8t` [N,K] i8 bytes + per-output `wsum` over the 16-lane span (the u8-offset correction). `w8t` = K·N bytes, `wsum` = N i64 slots | demo, suite, bench |
| `ternary_matmul_free_simd(qx, sx, w8t, wsum, gamma, b, y, M, K, N, xu8)` | the 0.4.1 `iv_dp8` kernel — **bit-identical** to the scalar kernel (never saturates for ternary weights: pair-sums ≤ 510; exact to K ~ 2²⁴). `xu8` = K bytes scratch. Any K (16-lane span + branchless tail) | demo, suite, bench |
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
| `tx_pack(V, T, C, F)` | quantize + pack all 7 BitLinears from the live `tx_*` latents — **once**; ~4 ns/weight | suite, bench, example |
| `tx_fwd_packed(tokens, V, T, C, F)` | serving forward from the packed store — per call only the per-token activation quant runs, weights untouched; logits → `ki_logits`. **Bit-identical** to `tx_fwd_q` modes 0/2 (suite-gated, twice — the store is serve-time read-only) | suite, bench, example |

Internal (not frozen): `bl_forward_pk`, `attn_sublayer_pk`, `mlp_sublayer_pk`,
`pk_*` store globals.

## Documented output cells (module globals, read-only for consumers)

| global | written by | contents |
|--------|-----------|----------|
| `tx_logits` | `tx_fwd` / `tx_loss` | T×V f64 logits (f64 path) |
| `ki_logits` | `tx_fwd_q` | T×V f64 logits (integer paths) |
| `mlp_out` | `block_fwd` | T×C block output (feeds the final norm) |

## Performance contract

See [`benchmarks.md`](benchmarks.md): the SIMD kernel is pack-once-then-hot —
`tsimd_pack_w` costs ~4 ns/weight once; the kernel then beats the f64-SIMD matmul
5–18× (growing with layer size). `bl_forward_q`/`tx_fwd_q` re-quantize per call by
design (reference fairness); **the pack-once deployment entry point shipped in
0.7.0 (additive)** — `tx_pack` + `tx_fwd_packed` serve the whole model at **5.8×
the f64 path** (13,472 vs 2,316 tok/s at the bench config), the one-time pack
amortizing after ~3 forwards.
