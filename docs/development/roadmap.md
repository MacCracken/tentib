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

### M1 — BitLinear + the straight-through estimator (v0.2.0) — ✅ cut 2026-06-23

- ✅ Wired `[deps.rosnet]` 0.2.0 + `[deps.tyche]` 0.1.1.
- ✅ **BitLinear forward**: per-tensor absmean ternary weight quant + per-token
  absmax int8 activation quant; scales baked into the dequantized operands (no
  trailing rescale), matmul via rosnet `linear_fwd`. (RMSNorm/SubLN deferred — the
  bare layer is enough to prove the STE; the norm lands with the M2 block.)
- ✅ **STE backward**: `dW` straight-through to the latent weight, clip-masked
  (`|W/γ|<1`), **no γ factor** (γ cancels); `dx` through `Weff`; activation STE mask.
- ✅ **Gate (the headline):** analytic STE `dW` == FD of the linearized surrogate
  on unsaturated entries; `dx` == FD through `Weff`; **γ-cancellation exercised at
  γ=3**; falsifiers (FD-through-quantized diverges; masked-entry surrogate FD
  nonzero); descent. Suite **34/34**. *Prove the surrogate, not the discontinuity.*
- `VERSION`→0.2.0 + CHANGELOG `[0.2.0]` cut 2026-06-23 (user tags). RMSNorm carried into M2.

### M2 — Ternary transformer trains from scratch (v0.3.0) — in progress

`Linear → BitLinear` across attn11's block shape, FD-gated; the ternary sibling of
attn11's first loss curve. Sub-bites:

- ✅ **bite-A (2026-06-23): minimal ternary LM trains from scratch.** `src/model.cyr`
  — softmax-CE (ported from attn11) + embedding → BitLinear head → softmax-CE; SGD
  on the latent weight. CE 2.07 → ~0.57 on the synthetic successor bigram (7/8 argmax).
  FD-gated end-to-end (head dW vs surrogate w/ the real loss dy, embedding grad vs
  live loss, falsifier). 57/57.
- ✅ **bite-B (2026-06-23): RMSNorm** fwd+bwd (`src/layers.cyr`), FD-gated (dx + dg).
  Pre-norm, no centering; backward cross-checked vs attn11's `ln_bwd`.
- ✅ **bite-C (2026-06-23): GELU** fwd+bwd (`src/layers.cyr`), FD-gated. tanh-approx,
  ported from attn11; `f64_tanh` isn't a builtin here so `tanh` is built from
  `f64_exp` (`f64_sqrt` confirmed available). *(MLP = up/down BitLinear + GELU
  assembles at bite-E.)*
- ✅ **bite-D (2026-06-23): scaled-dot-product self-attention core** (`src/layers.cyr`
  `attn_fwd`/`attn_bwd`) — single-head, causal, full-precision Q/K/V; the softmax-
  attention backward (`dQ`/`dK`/`dV`) cross-checked vs attn11's `attn_core_bwd`,
  all three FD-gated. The last *novel* op of M2.
- **bite-E**: assemble the full transformer block over a multi-token sequence
  (`src/block.cyr`, module-global scratch). Each step end-to-end dx-FD-gated.
  - ✅ **E1 (2026-06-23): attention sublayer** — `RMSNorm → BitLinear Q/K/V →
    causal attention → BitLinear O → +residual`, M=T whole-sequence projections;
    dx == FD over the full chain. 66/66.
  - ✅ **E2 (2026-06-23): MLP sublayer** (`RMSNorm → BitLinear-up → GELU →
    BitLinear-down → +residual`), dx == FD. 
  - ✅ **E3 (2026-06-23): full pre-norm block** = E1 ∘ E2; the per-sublayer residual
    fold makes `block_bwd` thread `dout → dxmid → dx`. Whole block dx == FD. 70/70.
  - ✅ **E4 (2026-06-23): the ternary transformer LM trains from scratch** —
    token-embed + learned pos-embed → block → final RMSNorm → separate ternary
    BitLinear head → per-position softmax-CE. **E4a** fwd/bwd end-to-end FD-gated
    (`d_tokemb`/`d_posemb == FD`); **E4b** trains: CE `1.86 → 0.04` on a synthetic
    sequence (the "first loss curve"). 76/76. **M2 core complete.**
- ⏳ **bite-F**: wire `[deps.akshara]`, swap synthetic for a real tokenized corpus
  (tarka 0.2.0→0.2.1 pattern) → **cut v0.3.0**.
- Later refinement: swap SGD for Adam once the multi-layer landscape needs it.

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
