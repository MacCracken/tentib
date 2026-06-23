# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.1.0** released 2026-06-23 (M0). **M1 (BitLinear + STE) built + gated on the
0.1.0 line, staged for the 0.2.0 cut** (VERSION file still 0.1.0 pending tag).

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
- `src/main.cyr` — demo driver (M0 quantizer + an M1 BitLinear layer).

## Tests

- `tests/tentib.tcyr` — **34/34** green: rosnet smoke, M0 quantization (6),
  BitLinear forward, the **finite-difference STE gate** (dW vs surrogate, dx vs
  Weff, γ-cancellation @ γ=3, two falsifiers, descent), int8 activation quant.
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

Cut **v0.2.0** (M1). Then **M2** — a ternary transformer that trains from scratch
over an akshara corpus (loss descends, FD-gated). See [`roadmap.md`](roadmap.md).
