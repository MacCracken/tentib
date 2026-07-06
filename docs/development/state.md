# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.8.0** — **security/hardening audit + CHANGELOG completeness** (the LAST v1.0
criterion — all six now green), cut 2026-07-06 (user tags). Audit record
`docs/audit/2026-07-06-audit.md` (re-derived from source): **6 findings
fixed/guarded** — the γ=0 quantizer sign bug FIXED (all-zero weights quantized to
all-“−1” via Inf→INT_MIN→clip; the M1 chain was already safe) + fail-loud
`guard()` on non-ternary pack input (`tpack2`/`tsimd_pack_w`), the SIMD K ≤ 2²²
exactness bound, and packed-store misuse (`tx_pack`/`tx_fwd_packed` ordering) —
**5 paths verified sound** (incl. `_tanh`, immune to the ganita overflow class by
its stable form). Cold paths only; packed serving unchanged (13,675 tok/s
guarded run). SECURITY.md rewritten to the audited posture. Suite **101/101**.
Prior: **0.7.0** (pack-once serving `tx_pack_init`/`tx_pack`/`tx_fwd_packed`,
bit-identical, 5.7× f64 whole-model; self-checking `examples/quickstart.cyr`),
**0.6.0** (alloc-clean + API freeze — `docs/api.md`). Prior: **0.5.0** (benchmarks — `docs/benchmarks.md` + the real
bcyr harness; SIMD advantage grows 5.0×→18.4× over f64-SIMD with layer size;
whole-model 3,549 vs 2,307 tok/s; honest toy-scale quality delta vs f64 attn11
CE 0.006 vs 0.11), **0.4.1** (the integer-SIMD ternary kernel — the toolchain gate
lifted by cyrius 6.4.6/6.4.7 `iv_dp8`; bit-identical to scalar + beats the
f64-SIMD matmul; whole-model = `bl_forward_q` mode 2), **0.4.0** (M3 — whole-model
matmul-free integer inference, exact parity < 1e-9, argmax 10/10, 2-bit packed
32×), **0.3.0** (M2 — ternary transformer trains), **0.2.0** (M1 — BitLinear +
STE), **0.1.0** (M0 — ternary quantizer).

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
  attn11), `lm_train` (public; `lm_init`/`lm_eval_sweep`/`lm_correct` are its
  internals — 0.6.0 classification), and `bl_xent_loss` (the FD-gate's
  quantized-loss probe). Embedding (full-precision) → BitLinear head (ternary) →
  softmax-CE, SGD on the latent weight.
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
  exactness bounds documented in-file). **0.7.0 pack-once serving**:
  `tx_pack_init`/`tx_pack` (7-layer packed store, built once) + `tx_fwd_packed`
  (deployment forward; internal `bl_forward_pk`/`*_sublayer_pk` chain).
- `src/main.cyr` — demo driver (M0 quantizer · M1 BitLinear · M2 transformer-trains ·
  M3 matmul-free kernel · M3b branchless bench · M3c whole-model integer inference).

## Tests

- `tests/tentib.tcyr` — **101/101** green: rosnet smoke, M0 quantization, BitLinear
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
  biased head), **M3c** (whole-model integer inference: < 1e-9 relerr vs the
  full-quant f64 forward, finite activation-quant delta vs the trained model,
  argmax agreement at all T), the **0.7.0 pack-once gates** (packed serving
  bit-identical to the scalar integer path + reproduced on a second serve), the
  **0.8.0 hardening gates** (all-zero weights → γ=0 + all-zero codes, the
  ternary_one γ=0 fix, guard pass-path), and the **0.6.0 API-freeze gates**
  (`ref_dot` exact, `bl_forward_q` direct mode 0-vs-2 bit-identity + same γ, the
  manual zero/loss/bwd/sgd quartet descends).
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

**v0.8.0 cut (security/hardening audit — the last v1.0 criterion).** All six
v1.0 criteria are green; the remaining step is the **1.0.0 clean cut** (no code
change expected — the freeze + criteria close it, per the tarka/prajna/anukūlana
precedent). Post-1.0 levers live in [`roadmap.md`](roadmap.md) (GPU ternary
kernels via mabda; hoosh/murti serving consumption; rotation-PTQ pairing with
the Type-3 lane).
