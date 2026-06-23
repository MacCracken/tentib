# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.3.0** — M2 (a ternary transformer trains from scratch), released 2026-06-23.
**M3a (the matmul-free integer inference kernel) built + gated on the 0.3.0 line,
staged for v0.4.0**: int8 activations + ternary weights → integer signed-accumulate
(no multiply), logit-parity < 1e-9 vs the f64 forward, 2-bit packed weights (32×).
82/82. Prior: **0.2.0** (M1 — BitLinear + STE), **0.1.0** (M0 — ternary quantizer).

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
- `src/kernel.cyr` — **M3 matmul-free kernel**: `act_quant_int` (int8 codes) +
  `ternary_matmul_free` (i64 signed-accumulate, no multiply, dequant) +
  `tpack2`/`tunpack2` (2-bit packed ternary storage).
- `src/main.cyr` — demo driver (M0 quantizer · M1 BitLinear · M2 transformer-trains ·
  M3 matmul-free kernel).

## Tests

- `tests/tentib.tcyr` — **60/60** green: rosnet smoke, M0 quantization, BitLinear
  forward, the **M1 STE FD-gate** (dW vs surrogate, dx vs Weff, γ-cancellation @ γ=3,
  falsifiers, descent), int8 activation quant; **M2** — softmax-CE FD gate, the
  **end-to-end gradient gate** (head dW vs surrogate w/ softmax-CE dy, embedding
  grad vs live loss, falsifier), the **ternary-LM descent** (final CE < initial,
  argmax ≥ 6/8), **RMSNorm + GELU + attention** fwd/bwd FD gates, the **E1/E2/E3
  end-to-end dx gates** (attention sublayer, MLP sublayer, full block), the **E4a LM
  embedding-gradient FD gate**, the **E4b descent** + **bite-F** (trains on akshara
  text), and **M3** (matmul-free integer kernel: logit-parity < 1e-9, 2-bit pack
  round-trip).
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

**M3 continues toward v0.4.0**: M3a (the matmul-free integer kernel) done; next is
**M3b** (a branchless int-SIMD kernel — the throughput lever) and **M3c** (the kernel
through the whole trained transformer + a tok/s benchmark). See [`roadmap.md`](roadmap.md).
