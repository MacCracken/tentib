# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.1.0** — M0 (scaffold + ternary quantizer), 2026-06-23. Builds + tests green.

## Toolchain

- **Cyrius pin**: `6.2.37` (in `cyrius.cyml [package].cyrius`)

## Source

- `src/ternary.cyr` — the M0 core (raw f64, no deps): `absmean` (γ), `ternary_one`
  / `ternary_quantize` (clip(round(w/γ),−1,+1) → {−1,0,+1}), `ternary_dot`
  (matmul-free signed-accumulate · γ), `ref_dot` (full-precision baseline).
- `src/main.cyr` — demo driver (quantize an 8-vector; print ternary weights, γ,
  matmul-free dot vs full-precision dot + error).

## Tests

- `tests/tentib.tcyr` — ternary-quantization correctness: **6/6** green (γ = absmean,
  four ternary values incl. the zero state, matmul-free dot == γ·signed-sum).
- `tests/tentib.bcyr` / `.fcyr` — benchmark / fuzz stubs (no-op).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math

External substrate (commented out until the milestone that needs it):

- **rosnet** (latent f64 weights + grad) + **tyche** (PRNG) → M1
- **akshara** (tokenizer) → M2

## Consumers

_None yet._ (Eventual: hoosh / murti serving the ternary model.)

## Next

M1 — BitLinear + straight-through estimator on rosnet, finite-difference-gated.
See [`roadmap.md`](roadmap.md).
