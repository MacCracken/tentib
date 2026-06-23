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

### M2 — Ternary transformer trains from scratch (v0.3.0) — ✅ cut 2026-06-23

`Linear → BitLinear` across attn11's block shape, FD-gated; the ternary sibling of
attn11's first loss curve. **Done — 80/80, trains on real akshara text (CE → 0.11).**
Sub-bites:

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
- ✅ **bite-F (2026-06-23): trains on a real akshara corpus.** `[deps.akshara]` 0.1.0;
  `corpus_set`/`gd_ld` byte-tokenize `"hello world"` (V=8); the transformer trains
  next-token, CE `2.08 → 0.11`. **v0.3.0 cut.**
- Later refinement: swap SGD for Adam once the multi-layer landscape needs it.

### M3 — Packed-ternary + int8 matmul-free inference kernel (v0.4.0) — ✅ cut 2026-06-23

The matmul-free kernel, carried through the whole trained transformer. **86/86.**

- ✅ **M3a (2026-06-23): the matmul-free integer kernel** (`src/kernel.cyr`):
  `ternary_matmul_free` (int8 activations + ternary weights → i64 signed-accumulate,
  no multiply → dequant + bias) + `act_quant_int` + `tpack2`/`tunpack2` (2-bit storage).
  **Logit parity < 1e-9** vs `bl_forward_full`; round-trip-exact 2-bit packing
  (32× smaller). 16384 → 0 inner-loop multiplies on a 128×128 layer.
- ◐ **M3b: the throughput lever — scalar half done (0.4.0), SIMD half → 0.4.1.**
  - ✅ **branchless scalar kernel** (`ternary_matmul_free_bl`): mask arithmetic replaces
    the unpredictable per-weight branch; bit-identical output; **84.5 µs vs branchy
    112.6 µs (~25%)** on a 128×128 layer. Still multiply-free (AND / ADD / SUB).
  - ⏳ **integer-SIMD kernel → v0.4.1, gated on the toolchain.** rosnet's SIMD-f64
    matmul (16.5 µs) still beats the scalar integer path; the b1.58 win needs
    INTEGER-SIMD lanes (int8 × sign-select, 16–32 / instr), which this Cyrius lacks
    (f64-only, 2-wide SSE2). Filed:
    [`proposals/2026-06-23-cyrius-integer-simd.md`](proposals/2026-06-23-cyrius-integer-simd.md).
- ✅ **M3c (2026-06-23): whole-model integer inference** — the kernel through every
  BitLinear of the trained transformer (`tx_fwd_q`): exact parity < 1e-9 vs the
  full-quant f64 forward (all 7 BitLinears integer), an honest activation-quant delta
  (~0.01) vs the weight-only trained model, **argmax agreement 10/10** positions, and
  a forward-latency benchmark (24 µs integer vs 16.5 µs f64).
- **Toolchain footgun filed:** Cyrius does not check call arity (an 8→9 param change
  silently miscompiled three call sites — `cyrius build` said `OK`).
  [`issues/2026-06-23-cyrius-call-arity-no-check.md`](issues/2026-06-23-cyrius-call-arity-no-check.md).

### M3.1 — integer-SIMD kernel (v0.4.1) — gated on cyrius integer SIMD

The vectorized int8 × ternary kernel — the b1.58 *"multiply-free is also faster"*
demonstration vs bitnet.cpp / the b1.58 paper under the B-series fairness rules. Opens
when the cyrius integer-SIMD proposal lands (typed `iNxM` vectors + int8/16/32 ops).

(original M3 acceptance below)


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
