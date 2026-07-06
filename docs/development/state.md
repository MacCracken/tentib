# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.5.0** — **benchmarks** (`docs/benchmarks.md`, a v1.0 criterion), cut 2026-07-06
(user tags). The real `tests/tentib.bcyr` harness under B-series fairness (single
pinned CPU, warm cache, identical shapes): layer-sweep SIMD advantage **grows with
size** (vs rosnet f64-SIMD **5.0× @128² → 18.4× @768²**; lane width + i8 cache
residency), prep amortization (~4 ns/wt pack, once), whole-model **3,549 tok/s vs
2,307 f64** (per-call re-pack included; pack-once deployment = 0.7.0), memory 32×
measured, and the **honest quality delta** vs matched-size f64 attn11 1.13.0
(CE 0.006 vs 0.11 at toy scale; b1.58 ≥3B parity + bitnet.cpp cited, not
re-demonstrated). Suite **90/90**. Prior: **0.4.1** (the integer-SIMD ternary
kernel — the toolchain gate lifted by cyrius 6.4.6/6.4.7 `iv_dp8`; bit-identical
to scalar + beats the f64-SIMD matmul; whole-model = `bl_forward_q` mode 2),
**0.4.0** (M3 — whole-model matmul-free integer inference, exact parity < 1e-9,
argmax 10/10, 2-bit packed 32×), **0.3.0** (M2 — ternary transformer trains),
**0.2.0** (M1 — BitLinear + STE), **0.1.0** (M0 — ternary quantizer).

## Toolchain

- **Cyrius pin**: `6.4.10` (in `cyrius.cyml [package].cyrius`; bumped from 6.2.37
  at 0.4.1 — `cyrius lib sync --full`, 86/86 re-validated pre-kernel-work)

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
- `src/kernel.cyr` — **M3 matmul-free kernel + whole-model integer inference**:
  `act_quant_int` (int8 codes) + `ternary_matmul_free` (i64 signed-accumulate, no
  multiply, dequant + bias) + **M3b** `ternary_matmul_free_bl` (branchless mask
  arithmetic, bit-identical) + `tpack2`/`tunpack2` (2-bit packed storage). **M3c**
  whole-model integer inference: `bl_forward_q` (mode 0 integer / mode 1 full-quant-f64
  reference / mode 2 integer-SIMD) + `attn_sublayer_q`/`mlp_sublayer_q`/`block_q`/
  `tx_fwd_q` (the trained transformer's 7 BitLinears, integer) + `tx_int_init` +
  `logits_argmax`. **0.4.1 SIMD kernel**: `tsimd_pack_w` (Wq [K,N] i64 → transposed
  [N,K] i8 + per-output `wsum` offset-correction) + `ternary_matmul_free_simd`
  (u8-offset codes, `iv_dp8` 16-lane inner loop, branchless tail — bit-identical,
  exactness bounds documented in-file).
- `src/main.cyr` — demo driver (M0 quantizer · M1 BitLinear · M2 transformer-trains ·
  M3 matmul-free kernel · M3b branchless bench · M3c whole-model integer inference).

## Tests

- `tests/tentib.tcyr` — **90/90** green: rosnet smoke, M0 quantization, BitLinear
  forward, the **M1 STE FD-gate** (dW vs surrogate, dx vs Weff, γ-cancellation @ γ=3,
  falsifiers, descent), int8 activation quant; **M2** — softmax-CE FD gate, the
  **end-to-end gradient gate** (head dW vs surrogate w/ softmax-CE dy, embedding
  grad vs live loss, falsifier), the **ternary-LM descent** (final CE < initial,
  argmax ≥ 6/8), **RMSNorm + GELU + attention** fwd/bwd FD gates, the **E1/E2/E3
  end-to-end dx gates** (attention sublayer, MLP sublayer, full block), the **E4a LM
  embedding-gradient FD gate**, the **E4b descent** + **bite-F** (trains on akshara
  text), **M3** (matmul-free integer kernel: logit-parity < 1e-9, 2-bit pack
  round-trip), **M3b** (branchless kernel bit-identical to branchy), **M3d** (0.4.1:
  SIMD kernel bit-identical to scalar across three K regimes — pure-tail K=4 /
  split K=20 / pure-`iv_dp8` K=32 — + whole-model mode 2 bit-identical incl. the
  biased head), and **M3c** (whole-model integer inference: < 1e-9 relerr vs the
  full-quant f64 forward, finite activation-quant delta vs the trained model,
  argmax agreement at all T).
- `tests/tentib.bcyr` — **the 0.5.0 benchmark harness** (drives `docs/benchmarks.md`):
  per-layer kernel sweep (branchy/branchless/SIMD-pack-once/rosnet-f64 at 128²–768²),
  prep-amortization costs, whole-model forward → tok/s (f64 / scalar-int / SIMD-int
  at a b1.58-shaped 267k-param config). B-series fairness; ratios reproduce, absolute
  ns vary with boost. `tests/tentib.fcyr` — fuzz stub (no-op).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math
- **rosnet** 0.2.0 (latent f64 weights + `linear_fwd`/`linear_bwd`) — M1, active
- **tyche** 0.1.1 (PRNG; rosnet's `t_randn` → `rng_normal`) — M1, active
- **akshara** 0.1.0 (tokenizer; `corpus_set`/`gd_ld` byte-vocab path) — M2, active

## Consumers

_None yet._ (Eventual: hoosh / murti serving the ternary model.)

## Next

**v0.5.0 cut (benchmarks — `docs/benchmarks.md` + the real bench harness).** The
remaining SemVer path to 1.0: **0.6.0** alloc-clean + API freeze + `docs/api.md` ·
**0.7.0** consumer readiness (the pack-once deployment inference entry point —
load → quantize + pack once → integer inference; §2/§3 of benchmarks.md motivate
it) · **0.8.0** security audit · **1.0.0**. Full per-version scope + acceptance in
[`roadmap.md`](roadmap.md).
