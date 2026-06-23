# tentib — Roadmap

> Milestone plan through v1.0. State lives in [`state.md`](state.md); this file is
> the sequencing. Demand-gated and post-beta per the
> [integer-native-ml.md](https://github.com/MacCracken/agnosticos/blob/main/docs/development/planning/integer-native-ml.md)
> map — opens as a sovereignty-demo / when a downstream pull arrives.

## v1.0 criteria

- [ ] Public API frozen — every exported symbol documented and tested
- [ ] BitLinear forward + STE backward, finite-difference-gated (the surrogate)
- [ ] A ternary transformer that trains from scratch (loss descends) via akshara
- [ ] Packed-ternary + int8 matmul-free inference kernel reproduces the model's logits
- [ ] Benchmarks in `docs/benchmarks.md` (tok/s of the integer kernel; quality delta vs same-size f64 attn11, honest small-scale)
- [ ] CHANGELOG complete from v0.1.0; security/hardening audit pass

## Milestones

### M0 — Scaffold + ternary quantizer (v0.1.0) — ✅ shipped 2026-06-23

- `cyrius init` scaffold (pin 6.2.37); doc-tree + scope ADR.
- `src/ternary.cyr` — self-contained on raw f64 (no deps): `absmean` (γ),
  `ternary_quantize` (clip(round(w/γ),−1,+1)), `ternary_dot` (matmul-free
  signed-accumulate), `ref_dot` (baseline). Demo + 6/6 correctness suite.

### M1 — BitLinear + the straight-through estimator (v0.2.0)

- Wire `[deps.rosnet]` (latent f64 weights + `linear` fwd/bwd) + `[deps.tyche]`.
- **BitLinear forward**: ternary weight quant (absmean) + int8 activation quant
  (absmax per token) + signed-accumulate + rescale; RMSNorm/SubLN before quant.
- **STE backward**: gradient to the latent weights as identity, masked where the
  weight saturated to ±1; activation-quant STE likewise.
- **Gate (the headline correctness story):** FD-check every *differentiable*
  sub-op of BitLinear; define the STE surrogate explicitly and verify it matches
  the non-quantized surrogate's gradient within tolerance. Prove the *surrogate*,
  not the discontinuity. (`tests/tentib.tcyr` grows the grad-check block.)

### M2 — Ternary transformer trains from scratch (v0.3.0)

- `Linear → BitLinear` across attn11's block shape (attention + MLP), int8
  activation path; `[deps.akshara]` for the char-LM corpus.
- **Acceptance:** loss descends under STE training, FD-gated; the ternary sibling
  of attn11's first loss curve. Honest small-scale quality delta vs same-size f64.

### M3 — Packed-ternary + int8 matmul-free inference kernel (v0.4.0)

- Packed {−1,0,+1} storage (2-bit / trit-packed) + the integer signed-accumulate
  inference kernel — where "no multiply" becomes a measured **tok/s**.
- **Acceptance:** integer kernel reproduces the f64-latent forward's logits within
  the activation-quant tolerance (integer-exact where applicable); tok/s vs the
  f64 path, benchmarked under the B-series fairness rules vs a named reference
  (bitnet.cpp / the b1.58 paper).

### M4 — Optimization + docs → v1.0

- Allocation-clean, API freeze + `docs/api.md`, `docs/benchmarks.md`, a downstream
  consumer (e.g. hoosh/murti serving the ternary model), security audit → clean cut.

## Out of scope (for v1.0)

- **GPU ternary kernels** — mabda-backed; an accelerator, not a prerequisite (CPU
  f64 train + integer inference kernel is the v1.0 surface).
- **Rotation-PTQ of imported checkpoints** (QuaRot / SpinQuant) — a *separate*
  technique (quantize-the-import), paired with the Type-3 weight-import reference.
- **Scale claims** — b1.58's "matches fp16 at 3B+" is cited, not re-demonstrated.
- **The matmul-free token mixer** (Scalable MatMul-free LM) — a research-watch
  sibling, ties into the Type-2 recurrent reference; not v1.0.
