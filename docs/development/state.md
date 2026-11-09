# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.1.2** (2026-11-09): the P(-1) hardening, optimization and repair sweep
([`docs/audit/2026-11-09-audit.md`](../audit/2026-11-09-audit.md)). 2 MEDIUM fixed: an
out-of-range token id made `tx_bwd` write past `tx_dtokemb`, and `bl_ste_mask_x` zeroed the
max element's gradient in ~14% of rows. 9 LOW fixed: target ids, dims vs the init sizing, NaN
through the integer path, the guard's stderr channel, `absmean(·, 0)`, test gaps, the test-stub
warnings, CI integrity, and bench history. No signature changed. A 333,281-value golden dump is
bit-identical to 1.1.1 except the mask fix's one value. Faster with the same bits: the f64
forward −33%, packing −53%, packed serving −6% (14.2k tok/s), a training step −19%. Suite
**139** (incl. 20 guard fail paths in forked children); a 5,000-round property fuzzer; both
also on aarch64 (qemu) in CI. `cyrius audit` clean. Posture:
[ADR 0002](../adr/0002-hardening-preserves-bits.md).

**1.1.1** (2026-11-09): toolchain 6.7.9 → 6.7.18 and the deps' latest releases. rosnet
1.2.2–1.4.1 made `linear_fwd` 2.4–2.9× faster, so the "vs f64" ratios shrank. The figures
below and in `docs/benchmarks.md`'s tables are the 2026-07-06 run of record; its dated notes
carry the 1.1.1 and 1.1.2 numbers, and a full re-baseline is still open.

Releases since 1.0.0: 1.0.1 (AGNOS GPU integer offload, `ternary_matmul_free_gpu`) ·
1.0.2 (pin 6.5.27, deps to their 1.x freezes) · 1.0.3 (pin 6.6.2) · 1.0.4 (pin 6.6.20) ·
1.1.0 (pin 6.7.9) · 1.1.1 (pin 6.7.18, latest deps) · 1.1.2 (P(-1) sweep).

**1.0.0 — STABLE**, cut 2026-07-06 (user-named; user tags). The clean cut
(tarka/prajna/anukūlana precedent): **no code change from 0.8.0** — all six v1.0
criteria green (BitLinear+STE FD-gated · ternary transformer trains ·
matmul-free integer kernel reproduces the model · benchmarks · API frozen ·
CHANGELOG + audit). The `docs/api.md` freeze is **in force for the 1.x series**
(minors add only). The one-day 0.4.1→1.0.0 arc: int-SIMD kernel (`iv_dp8`,
bit-identical, 5–18× the f64-SIMD matmul, growing with layer size) → benchmarks
(B-series fairness; honest quality delta vs f64 attn11) → API freeze +
alloc-audit → pack-once serving (**~13.5k tok/s vs ~2.3k f64 = 5.7×**
whole-model; `examples/quickstart.cyr`) → security audit (6 fixed/guarded incl.
the γ=0 quantizer sign bug, 5 verified sound). Suite **101/101**. Post-1.0 =
user-driven levers only (GPU/mabda, hoosh-murti serving consumption,
rotation-PTQ, NEON on cyrius Phase 5). Prior: **0.5.0** (benchmarks — `docs/benchmarks.md` + the real
bcyr harness; SIMD advantage grows 5.0×→18.4× over f64-SIMD with layer size;
whole-model 3,549 vs 2,307 tok/s; honest toy-scale quality delta vs f64 attn11
CE 0.006 vs 0.11), **0.4.1** (the integer-SIMD ternary kernel — the toolchain gate
lifted by cyrius 6.4.6/6.4.7 `iv_dp8`; bit-identical to scalar + beats the
f64-SIMD matmul; whole-model = `bl_forward_q` mode 2), **0.4.0** (M3 — whole-model
matmul-free integer inference, exact parity < 1e-9, argmax 10/10, 2-bit packed
32×), **0.3.0** (M2 — ternary transformer trains), **0.2.0** (M1 — BitLinear +
STE), **0.1.0** (M0 — ternary quantizer).

## Toolchain

- **Cyrius pin**: `6.7.18` (in `cyrius.cyml [package].cyrius`; bumped from 6.7.9
  at 1.1.1, 2026-11-09). `cyrius lib sync --full` re-vendored the whole 6.7.18 stdlib
  snapshot (113 files, byte-matching); `cyrius.lock` regenerated (3 commit pins, 116
  hashes, `--verify` and `--locked` clean); suite re-verified 101/101 at the new pin.
  tentib's source contains none of the constructs the 6.7.x language changes touch
  (newly reserved words, `!`, unary `-` / `~` on constants, typed `*T` pointers, `f32`).
  From 6.7.18, the pin, `lib/` and `cyrius.lock` must land in one commit: with a fully
  vendored `lib/`, cyrius refuses a moved pin until `lib sync --full` has run.
- Pin history (1.x): 6.4.72 → 6.5.27 (1.0.2) → 6.6.2 (1.0.3) → 6.6.20 (1.0.4) →
  6.7.9 (1.1.0) → 6.7.18 (1.1.1).

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
- `src/gpu.cyr` — `ternary_matmul_free_gpu`: the integer accumulate tiled 8×8×8 onto the
  AGNOS `#82` dispatch, bit-exact at any K; the host computes every tile itself (gated in
  the suite at ragged shapes since 1.1.2). Built by `programs/gpumm.cyr`.
- `src/main.cyr` — demo driver (M0 quantizer · M1 BitLinear · M2 transformer-trains ·
  M3 matmul-free kernel · M3b branchless bench · M3c whole-model integer inference).
- `src/test.cyr` — the `[build].test` stub (defines akshara's hooks).
- `scripts/bench-history.sh` — builds the bench harness once, runs it pinned (core 4)
  RUNS times, appends the medians to `bench-history.csv` (1.1.2).

## Tests

- `tests/tentib.tcyr` — **139/139** green (x86_64 and aarch64): rosnet smoke, M0 quantization, BitLinear
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
  manual zero/loss/bwd/sgd quartet descends), and the **1.1.2 sweep group** (the
  activation-mask fix, NaN propagation, `absmean(·, 0)`, the GPU fallback at four
  ragged shapes, prefix serving, and 20 guard fail paths in forked children with two
  exit-0 controls).
- `tests/tentib.bcyr` — **the 0.5.0 benchmark harness** (drives `docs/benchmarks.md`):
  per-layer kernel sweep (branchy/branchless/SIMD-pack-once/rosnet-f64 at 128²–768²),
  prep-amortization costs, whole-model forward → tok/s (f64 / scalar-int / SIMD-int
  at a b1.58-shaped 267k-param config) + packed serving + (1.1.2) a training step.
  B-series fairness; ratios reproduce, absolute ns vary with boost.
- `tests/tentib.fcyr` — **seeded property fuzzer** (1.1.2): 5,000 rounds, ~3 s,
  properties P1–P8 (kernel agreement, code range, pack round-trip, Weff = code·γ,
  integer vs f64 within the summation bound, NaN rows, mask identity, whole-model
  modes + prefix serving).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math, fnptr
- **rosnet** 1.4.1 (latent f64 weights + `linear_fwd`/`linear_bwd`) — M1, active
- **tyche** 1.2.2 (PRNG; rosnet's `t_randn` → `rng_normal`) — M1, active
- **akshara** 1.1.2 (tokenizer; `corpus_set`/`gd_ld` byte-vocab path) — M2, active

All three are the latest releases, built at cyrius 6.7.18. rosnet 1.4.1 pins tyche 1.2.2.
akshara 1.1.2 still pins rosnet 1.2.2, and tentib's root pin of 1.4.1 wins (the resolver
notes it). That is safe: akshara calls only `t_alloc` / `t_zero`, and rosnet's API is unchanged.

## Consumers

_None yet._ (Eventual: hoosh / murti serving the ternary model.)

## Next

tentib is stable (1.x, API frozen); the open work is in [`roadmap.md`](roadmap.md),
proposed in order: **1.2.0** packed-model serialization (the first untrusted-input
surface) · **1.3.x** a distlib bundle + the first consumer (hoosh / murti) · **1.4.x**
serving from 2-bit / trit packed storage · **1.5.x** the Pi 5 edge measurement (NEON
`iv_dp8` already runs on aarch64) · **1.6.x** GPU kernels via mabda · later 1.x minors
(a matched-recipe quality ablation, the token-mixer research watch) · a **2.0.0
backlog** (model handles instead of module globals, among others). Maintenance rides
any patch: the benchmark re-baseline (maintainer's call), the akshara → rosnet
realign, re-proving `/bin/gputern` on iron, pin sweeps (latest: 6.7.18 at 1.1.1).
The latest P(-1) sweep is 1.1.2.
