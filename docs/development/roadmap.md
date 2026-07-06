# tentib — Roadmap

> **Self-contained SemVer plan.** An agent can pick up the next version from this repo
> alone — the design rationale is inlined below + lives in-repo under `docs/`. (The
> ecosystem map at agnosticos `docs/development/planning/integer-native-ml.md` is
> background, not required.) Volatile state — current version, test count, deps — is in
> [`state.md`](state.md); the per-version detail is in [`../../CHANGELOG.md`](../../CHANGELOG.md).

## What tentib is

**tentib** = `bitnet` reversed — the sovereign **integer-native / ternary (1.58-bit)**
ML reference. BitNet b1.58 weights collapse to **{−1, 0, +1}**, so `weight · activation`
becomes **add / subtract / skip — no multiply**. The thesis: attn11 proved learning is
expressible in an everything-is-i64 sovereign language; tentib takes the i64 claim to
the **weights themselves**.

Discipline (inherited from attn11, non-negotiable): **Cyrius-native, no BLAS / libc /
autodiff; finite-difference-gated** (prove the *surrogate*, not the discontinuity — the
STE is the canonical case); **benchmarked vs a named real-world reference**;
**honest-scale** — b1.58's "matches fp16" is a ≥3B *scale* result, so at tiny
from-scratch scale **report the quality gap** (like attn11's MTP honest-negative), never
hide it.

## What it stands on

- **rosnet** (f64 latent tensors + `linear_fwd` / `linear_bwd` — the matmul BitLinear wraps)
- **tyche** (PRNG; rosnet's `t_randn` → `rng_normal`)
- **akshara** (the byte-vocab tokenizer the proof-of-life trains over)
- **Latent weights stay full-precision f64**; quantization happens only inside the
  forward pass (the BitNet shadow-weight loop). The optimizer updates the latent weights;
  the forward re-quantizes ternary each step. b1.58 ternarizes **only the linear layers**
  — embeddings, RMSNorm, GELU, and the attention core stay f64.

## Shipped

| Version | Milestone | Result |
|---------|-----------|--------|
| **0.1.0** | M0 — ternary quantizer | absmean γ, `clip(round(w/γ),−1,+1)`, matmul-free dot. **6/6**. |
| **0.2.0** | M1 — BitLinear + STE | ternary W + int8 acts on rosnet; STE backward FD-gated (the surrogate, γ-cancellation @ γ=3, falsifiers). **34/34**. |
| **0.3.0** | M2 — ternary transformer trains | RMSNorm + causal attn + GELU-MLP, all 6 linears ternary; trains on akshara text (CE → 0.11); every level end-to-end FD-gated. **80/80**. |
| **0.4.0** | M3 — matmul-free integer inference | the kernel through all 7 BitLinears of the trained transformer; exact parity < 1e-9 vs full-quant f64, argmax 10/10; 2-bit packed weights (32×); branchless scalar kernel (~25% over branchy). **86/86**. |
| **0.4.1** | integer-SIMD ternary kernel | the gate lifted (cyrius 6.4.6/6.4.7 int SIMD, `iv_dp8`): `ternary_matmul_free_simd` (pack-once i8 transpose + u8-offset codes + 16-lane `iv_dp8` + branchless tail) **bit-identical** to scalar and **~7.5× faster than rosnet f64-SIMD** on 128×128 (1,946 vs 14,670 ns; ~45× over branchless) — *multiply-free is also faster* met. Whole-model = mode 2. Pin 6.2.37 → 6.4.10. **90/90**. |
| **0.5.0** | benchmarks (`docs/benchmarks.md`) | the real `tests/tentib.bcyr` harness, B-series fairness: layer-sweep SIMD advantage **grows 5.0×→18.4×** over f64-SIMD (128²→768²; lane width + i8 cache residency); whole-model **3,549 vs 2,307 tok/s** (per-call re-pack included — pack-once deployment bounded by the sweep, → 0.7.0); memory 32× measured; **honest quality delta** vs matched-size f64 attn11 (CE 0.006 vs 0.11 toy-scale; b1.58 ≥3B parity cited). **90/90**. |

Full detail per version in [`../../CHANGELOG.md`](../../CHANGELOG.md).

## Remaining → v1.0 (SemVer plan)

Each version is an **independently-cuttable bite** with its own acceptance gate. Cut
process: substantive work at the version → bump `VERSION` + the `[x.y.z]` CHANGELOG
section → suite green → **the user tags** (never `git commit`/`tag` yourself).

> **Version numbers below are the proposed decomposition** — adjust granularity with the
> maintainer. The *ordering* (0.4.1 gated first, audit last) is the load-bearing part.

### 0.4.1 — integer-SIMD ternary kernel  ✅ **SHIPPED 2026-07-06** (see the table above)

The gate held exactly as designed: cyrius 6.4.6/6.4.7 (SIMD arc Phase 3) answered the
filed proposal with integer vector types + `iv_dp8` (the widening u8·i8 → i32 dot),
and the kernel landed the same release-window with the acceptance met — bit-identical
to scalar AND ~7.5× faster than rosnet's `linear_fwd` on the same layer.

### 0.5.0 — benchmarks (`docs/benchmarks.md`)  ✅ **SHIPPED 2026-07-06** (see the table above)

All three planned pieces landed with the honest caveats stated: tok/s under B-series
fairness (whole-model + the per-layer sweep the whole-model number is bounded by),
the measured 32× memory story, and the quality delta vs a matched-size f64 attn11
with the b1.58/bitnet.cpp references cited. The per-call-re-pack honesty note feeds
directly into 0.7.0's pack-once deployment entry point.

### 0.6.0 — allocation-clean + API freeze + `docs/api.md`  *[v1.0 criterion: API frozen]*

- **Allocation-clean:** rosnet's bump allocator never frees; repeated inference must
  reuse scratch (the `*_init` pattern already does — audit every path, document the
  lifetime contract, no per-call allocation in the hot loop).
- **API freeze:** settle the exported surface (the quantizer, BitLinear fwd/bwd, the
  block/LM, the kernel, `tx_fwd_q` / `tx_int_init` / `logits_argmax`). Document every
  exported symbol + its contract in `docs/api.md`; ensure each is test-covered.
- **Acceptance:** `docs/api.md` complete; no undocumented exported symbol; the suite
  exercises every public entry point.

### 0.7.0 — downstream-consumer readiness

The second-consumer pass — a clean inference entry point a server can call.

- **Deliverable:** a documented, stable inference path (load latent weights → quantize +
  pack once → integer inference) + a worked example in `docs/examples/`. The eventual real
  consumer is **hoosh / murti** serving the ternary model; tentib provides the surface.
- **Acceptance:** the example runs integer inference from the **public API alone** (no
  internal-symbol reach-in) and is tested.

### 0.8.0 — security/hardening audit + CHANGELOG completeness  *[v1.0 criterion]*

- Hardening review: bounds on the packed-ternary codec, the int8-quant clamps,
  integer-overflow on the accumulate at realistic K, alloc sizing in `tx_int_init`.
- CHANGELOG complete + accurate from 0.1.0; all in-repo doc cross-links valid.
- **Acceptance:** audit notes recorded; no open hardening finding; CHANGELOG verified.

### 1.0.0 — clean cut

All v1.0 criteria green (below); public API frozen; clean cut.

## v1.0 criteria → version

- [x] BitLinear forward + STE backward, finite-difference-gated (the surrogate) — **0.2.0**
- [x] A ternary transformer that trains from scratch (loss descends) via akshara — **0.3.0**
- [x] Packed-ternary + int8 matmul-free inference kernel reproduces the model's logits — **0.4.0**
- [x] Benchmarks in `docs/benchmarks.md` (tok/s; quality delta vs same-size f64 attn11) — **0.5.0** (the speed half landed via **0.4.1**)
- [ ] Public API frozen — every exported symbol documented and tested — **0.6.0**
- [ ] CHANGELOG complete from v0.1.0; security/hardening audit pass — **0.8.0**

## Out of scope (for v1.0)

- **GPU ternary kernels** — mabda-backed; an accelerator, not a prerequisite (CPU f64
  train + integer inference kernel is the v1.0 surface).
- **Rotation-PTQ of imported checkpoints** (QuaRot / SpinQuant) — the *separate*
  quantize-the-import technique, paired with the Type-3 weight-import reference.
- **Scale claims** — b1.58's "matches fp16 at 3B+" is cited, not re-demonstrated.
- **The matmul-free token mixer** (Scalable MatMul-free LM) — a research-watch sibling.

## References (for the remaining work)

- **b1.58** — Ma et al., *The Era of 1-bit LLMs* (2024), arXiv:2402.17764 (the absmean
  ternary + int8 recipe; the scaling claims to cite).
- **bitnet.cpp** — Microsoft's CPU inference kernels (TL1/TL2/I2_S) — the integer-SIMD
  packing/lookup shape for 0.4.1.
- **STE** — Bengio, Léonard, Courville (2013), arXiv:1308.3432 — the surrogate the
  FD-gate validates.
- In-repo: [`proposals/2026-06-23-cyrius-integer-simd.md`](proposals/2026-06-23-cyrius-integer-simd.md)
  (the 0.4.1 design) · [`issues/2026-06-23-cyrius-call-arity-no-check.md`](issues/2026-06-23-cyrius-call-arity-no-check.md)
  (a toolchain footgun — always run the suite after a signature change; a clean `build`
  is not sufficient).
