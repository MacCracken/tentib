# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.2.0** — M1 (BitLinear + STE), cut 2026-06-23. **M2 bite-A (minimal ternary LM
trains from scratch) built + gated on the 0.2.0 line, staged for the v0.3.0 cut**
(v0.3.0 = the full ternary transformer; bite-A is its first sub-bite).
Prior: **0.1.0** (M0 — scaffold + ternary quantizer).

## Toolchain

- **Cyrius pin**: `6.2.37` (in `cyrius.cyml [package].cyrius`)

## Source

- `src/ternary.cyr` — the M0 core (raw f64, no deps): `absmean` (γ), `ternary_one`
  / `ternary_quantize` (clip(round(w/γ),−1,+1) → {−1,0,+1}), `ternary_dot`
  (matmul-free signed-accumulate · γ), `ref_dot` (full-precision baseline).
- `src/bitlinear.cyr` — **M1 BitLinear** on rosnet: `bl_weight_effective` (ternary
  Weff=γ·Wq), `bl_act_quant` (per-token int8), `bl_forward_wonly`/`bl_forward_full`;
  STE backward `bl_backward_wonly` + `bl_ste_mask_w`/`bl_ste_mask_x` (dW no-γ-factor
  clip-mask, dx through Weff); FD-gate losses `bl_surr_lin_loss`/`bl_loss_of_x`/
  `bl_quant_loss_wonly`.
- `src/model.cyr` — **M2 ternary LM**: `softmax_xent_fwd`/`_bwd` (ported from
  attn11), `lm_init`/`lm_forward`/`lm_train`/`lm_eval_sweep`/`lm_correct`, and
  `bl_xent_loss` (the FD-gate's quantized-loss probe). Embedding (full-precision) →
  BitLinear head (ternary) → softmax-CE, SGD on the latent weight.
- `src/layers.cyr` — **M2 differentiable layers**: `rmsnorm_fwd`/`bwd` (pre-norm),
  `gelu_fwd`/`bwd` (+ a `_tanh` from `f64_exp`), each with a `*_loss` FD probe.
- `src/main.cyr` — demo driver (M0 quantizer + M1 BitLinear + M2 LM-trains-from-scratch).

## Tests

- `tests/tentib.tcyr` — **60/60** green: rosnet smoke, M0 quantization, BitLinear
  forward, the **M1 STE FD-gate** (dW vs surrogate, dx vs Weff, γ-cancellation @ γ=3,
  falsifiers, descent), int8 activation quant; **M2** — softmax-CE FD gate, the
  **end-to-end gradient gate** (head dW vs surrogate w/ softmax-CE dy, embedding
  grad vs live loss, falsifier), the **ternary-LM descent** (final CE < initial,
  argmax ≥ 6/8), and **RMSNorm + GELU** fwd/bwd FD gates.
- `tests/tentib.bcyr` / `.fcyr` — benchmark / fuzz stubs (no-op).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math
- **rosnet** 0.2.0 (latent f64 weights + `linear_fwd`/`linear_bwd`) — M1, active
- **tyche** 0.1.1 (PRNG; rosnet's `t_randn` → `rng_normal`) — M1, active
- **akshara** (tokenizer) → M2 (commented in `cyrius.cyml`)

## Consumers

_None yet._ (Eventual: hoosh / murti serving the ternary model.)

## Next

**M2 continues** toward the full ternary transformer (v0.3.0). bite-A (minimal LM
trains from scratch) done; next bites: **RMSNorm → GELU-MLP block → self-attention
→ full transformer block → akshara corpus**. See [`roadmap.md`](roadmap.md).
