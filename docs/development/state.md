# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.3.0** — **M2: a ternary transformer trains from scratch**, cut 2026-06-23
(awaiting user tag). Bites A (minimal LM) · B (RMSNorm) · C (GELU) · D (attention) ·
E1–E3 (full block) · E4 (transformer LM trains) · F (real akshara corpus, CE → 0.11).
Every component + the full assembly FD-gated; 80/80. Prior: **0.2.0** (M1 — BitLinear
+ STE), **0.1.0** (M0 — ternary quantizer).

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
  `gelu_fwd`/`bwd` (+ a `_tanh` from `f64_exp`), `attn_fwd`/`bwd` (single-head causal
  scaled-dot-product attention), each with a `*_loss` FD probe.
- `src/block.cyr` — **M2 full transformer block + LM** (E1-E4): `attn_sublayer`
  (E1) + `mlp_sublayer` (E2) + `block_fwd`/`bwd` (E3, pre-norm) + `tx_*` — the
  multi-token LM (embed + pos → block → final RMSNorm → ternary head → softmax-CE)
  with `tx_train` (E4). Module-global scratch; each level FD-gated; the LM trains
  (CE 1.86 → 0.04 on a synthetic sequence).
- `src/main.cyr` — demo driver (M0 quantizer + M1 BitLinear + M2 LM-trains-from-scratch).

## Tests

- `tests/tentib.tcyr` — **60/60** green: rosnet smoke, M0 quantization, BitLinear
  forward, the **M1 STE FD-gate** (dW vs surrogate, dx vs Weff, γ-cancellation @ γ=3,
  falsifiers, descent), int8 activation quant; **M2** — softmax-CE FD gate, the
  **end-to-end gradient gate** (head dW vs surrogate w/ softmax-CE dy, embedding
  grad vs live loss, falsifier), the **ternary-LM descent** (final CE < initial,
  argmax ≥ 6/8), **RMSNorm + GELU + attention** fwd/bwd FD gates, the **E1/E2/E3
  end-to-end dx gates** (attention sublayer, MLP sublayer, full block), the **E4a LM
  embedding-gradient FD gate**, and the **E4b descent** (the ternary transformer
  trains: final CE < initial, < 0.5).
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

**M3 (v0.4.0)** — the packed-ternary + int8 **matmul-free inference kernel** (where
"no multiply" becomes a measured tok/s; logit-parity vs the f64-latent forward).
See [`roadmap.md`](roadmap.md).
