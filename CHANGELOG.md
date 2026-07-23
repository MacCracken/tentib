# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.1] — 2026-07-23

**AGNOS GPU offload (the 1.54.x GPU crown / C6, integer path): tentib's ternary integer matmul can run on the
AMD gfx90c shader cores.** Additive — `ternary_matmul_free_gpu` is a new peer of `ternary_matmul_free`; the CLI
and the frozen 1.x public API are unchanged (kernel internals are explicitly not frozen). This is the SECOND
GPU-crown consumer after rupantara's f64 path (`#83`), and the stronger one: integer accumulate is associative,
so it is **bit-exact at ANY K**, not just K≤8.

### Added
- **`src/gpu.cyr` — `ternary_matmul_free_gpu(qx, sx, Wq, gamma, b, y, M, K, N)`**, a bit-exact peer of
  `ternary_matmul_free` that tiles the integer accumulate `SUM_k qx·Wq` into 8×8×8 blocks and dispatches each
  on the gfx90c shader cores via the AGNOS ring-3 syscall **`#82`** (`gpu_dispatch`, i32 8×8 matmul), summing
  the tile results in i64 CPU-side; the f64 dequant (`gamma·sx·acc + b`) stays byte-identical to the CPU
  reference. **Bit-exact at any K** — integer addition is associative, so cross-tile i64 accumulation equals
  the reference's sequential sum exactly. **Signed-correct**: qx is int8-range and Wq is {−1,0,+1} so acc goes
  negative; `#82`'s MAC is `v_mul_lo_u32`+`v_add_u32` (two's-complement mod 2³², sign-transparent), and the
  low-32 result is sign-extended on readback. On non-AGNOS / no-GPU, each tile is computed directly from the
  i64 arrays — identical result. Returns the GPU tile count.
- **`programs/gpumm.cyr` — the crown proof.** Runs a real ternary projection `ternary_matmul_free(qx, sx, Wq,
  γ, 0, y, 8, 16, 32)` twice — CPU vs `ternary_matmul_free_gpu` — on **negative** activations + real ternary
  weights, and byte-compares the 2048-byte outputs. **K=16 = two k-tiles**, so the cross-tile accumulation the
  any-K claim rests on is exercised. Exit code: **95** = byte-identical AND all 8 tiles on the GPU (crown);
  **96** = byte-identical, 0 GPU tiles (host/QEMU — tiling + signed logic proven, no GPU); **90** = mismatch.

### Verified
- **tentib builds `--agnos` for the first time** (the crown-proof subset — kernel + rosnet/tyche/akshara; the
  file-streaming symbols are stubbed and unreachable).
- **Tiling + signed logic bit-identical on host** (2026-07-23): `build/gpumm` exits **96** —
  `ternary_matmul_free_gpu` (CPU-tiled, negative data, cross-tile K=16) is byte-for-byte equal to the
  production `ternary_matmul_free`.
- **CROWN PROVEN ON IRON** (2026-07-23): `run /bin/gpumm` → **`exit 95`** on archaemenid (AMD Cezanne) —
  byte-identical outputs AND all 8 tiles on the GPU, so a real ternary BitLinear projection's integer
  accumulate ran on the gfx90c shader cores via `#82`, **bit-exact at K=16 (cross-tile i64 accumulation)** —
  the any-K integer advantage over the f64 `#83` path. This is the AGNOS 1.54.x GPU crown (C6), second
  consumer (after rupantara/f64). **Bonus:** first iron confirmation of `#82` on NEGATIVE integers (its boot
  self-test only used positives; the signed two's-complement path is now proven). Tracked in agnosticos
  `iron-nuc-zen-log.md#tracker-156x-ml-crown-tentib`.

## [1.0.0] — 2026-07-06

**v1.0 — STABLE. The clean cut** (tarka/prajna/anukūlana precedent): **no code
change from 0.8.0** — the freeze and the criteria close it. tentib is the
sovereign integer-native / ternary (1.58-bit) ML reference: BitNet-b1.58 ternary
weights, STE training, and a matmul-free integer inference path that is
**bit-identical to its f64-latent reference and faster than the f64-SIMD matmul**
— trained, packed, and served entirely in Cyrius with no BLAS / libc / autodiff.

### All six v1.0 criteria green
- **BitLinear + STE, FD-gated** (0.2.0) — prove the surrogate, not the
  discontinuity; γ-cancellation + falsifiers.
- **A ternary transformer trains from scratch** (0.3.0) — RMSNorm + causal attn +
  GELU-MLP, all linears ternary, on akshara-tokenized text; every level FD-gated.
- **The matmul-free integer kernel reproduces the model** (0.4.0) — whole-model,
  exact parity < 1e-9, argmax 10/10, 2-bit packed weights (32×); + the 0.4.1
  integer-SIMD kernel (`iv_dp8`), bit-identical and 5–18× the f64-SIMD matmul.
- **Benchmarks** (0.5.0) — `docs/benchmarks.md`, B-series fairness; pack-once
  serving **~13.5k tok/s vs ~2.3k f64 (5.7×)** whole-model (0.7.0); honest
  toy-scale quality delta vs a matched-size f64 attn11, b1.58 ≥3B parity cited.
- **Public API frozen** (0.6.0) — `docs/api.md`, every symbol documented +
  directly gated; allocation audit (inference paths allocation-free). **The
  freeze is now in force for the 1.x series**: signatures/semantics stable,
  minors add only.
- **CHANGELOG complete + security/hardening audit pass** (0.8.0) —
  `docs/audit/2026-07-06-audit.md`; 6 findings fixed/guarded, 5 paths verified
  sound; fail-loud guards on cold paths only.

### Changed (docs only)
- Version-bearing docs swept to 1.0.0 (README status, api.md freeze declaration,
  SECURITY.md maturity language). Suite **101/101**; demo + bench + quickstart
  re-verified at the cut.

### Post-1.0 (user-driven levers, roadmap § out-of-scope)
- GPU ternary kernels (mabda) · hoosh/murti serving consumption of
  `tx_fwd_packed` · rotation-PTQ (QuaRot/SpinQuant-class) pairing with the
  Type-3 import lane · aarch64 NEON when cyrius Phase 5 lands.

## [0.8.0] — 2026-07-06

**Security/hardening audit + CHANGELOG completeness** — the last v1.0 criterion.
Audit record: [`docs/audit/2026-07-06-audit.md`](docs/audit/2026-07-06-audit.md)
(re-derived from source, comments not trusted). Posture = the tarka precedent:
fail-loud `guard()` on public preconditions whose violation means silent
corruption or silently-wrong numerics — **cold paths only, hot loops carry zero
checks** (packed serving throughput unchanged: 13,675 tok/s this run). Suite
**97 → 101**.

### Fixed — found by the audit
- **`ternary_one(w, gamma=0)` quantized all-zero weight vectors to all-“−1”**:
  w/0 → Inf/NaN → `f64_to` → INT_MIN → clipped to −1 (silent sign garbage through
  the public M0 pair; the M1+ chain was already safe — `bl_weight_effective`'s
  γ=0 branch verified real). Now γ==0 → 0 (b1.58: γ==0 ⟺ all-zero tensor).
  Regression-gated.

### Added — fail-loud guards (`guard(cond, msg)`, new public primitive)
- **`tpack2` / `tsimd_pack_w`**: a non-ternary weight no longer silently packs as
  0 / store8-truncates (the latter also voided the SIMD kernel's no-saturation
  exactness bound) — pack-time validation, fail-loud.
- **`ternary_matmul_free_simd`**: K ≤ 2²² enforced once per call (the i32
  accumulate is exact only for K ≲ 8.4M — the bit-identity contract's bound).
- **`tx_pack`**: fails loud before writing through null pointer tables if
  `tx_pack_init` / `tx_init` / `tx_int_init` haven't run (was silent heap
  corruption); **`tx_fwd_packed`**: fails loud before `tx_pack` (was silent
  garbage logits).

### Verified sound (audited, no change)
- `_tanh` (stable `e^(−2|x|)` form — the ganita `f64_tanh` overflow class does
  not apply, ±inf saturates to ±1) · `act_quant_int` zero-row + ±127 clamps ·
  the M1 degenerate-γ chain · `tx_int_init`/`tx_pack_init` sizing formulas vs
  all 7 BitLinear shapes · scalar kernels' i64 accumulate (no overflow at any
  allocatable K).

### Docs
- **CHANGELOG verified complete + gap-free 0.1.0 → 0.8.0**; all relative doc
  cross-links resolve (scripted check). `SECURITY.md`: stale version literal
  dropped, hardening-posture section added, the "0.8.0 pending" maturity note
  replaced with the audit record. `api.md`: `guard` added to the frozen surface
  + a guarded-preconditions convention note. `benchmarks.md`: run-of-record
  refreshed with guards live (pack ~4 → ~8 ns/weight — the only measurable cost,
  paid once per weight matrix; serving unchanged).

## [0.7.0] — 2026-07-06

**Downstream-consumer readiness — the pack-once deployment serving path.** The
piece the 0.5.0 benchmarks motivated: `tx_fwd_q` re-quantizes (mode 2: + re-packs)
every forward by reference-fairness design; a deployment packs **once** and serves.
All additions are **additive to the 0.6.0 frozen surface**. Suite **95 → 97**.

### Added — pack-once serving (`src/kernel.cyr`, public per `docs/api.md` §0.7.0)
- **`tx_pack_init(V, T, C, F)`** — allocate the packed store (7 per-layer
  transposed-i8 weight buffers + `wsum` tables + γs; attn Q/K/V/O, MLP up/down,
  head).
- **`tx_pack(V, T, C, F)`** — quantize + pack all 7 BitLinears from the live
  `tx_*` latent weights, once (~4 ns/weight; 6.4 ms at the bench config,
  amortized after ~3 forwards).
- **`tx_fwd_packed(tokens, V, T, C, F)`** — the serving forward: per call only the
  per-token activation quant runs, weights are never touched; logits → `ki_logits`.
  **Bit-identical to `tx_fwd_q` modes 0/2** (same quantize → same pack → same
  kernel, hoisted out of the call). Internal chain (`bl_forward_pk`,
  `*_sublayer_pk`) stays out of the freeze like its `_q` siblings.

### Benchmarks (`docs/benchmarks.md` §3 updated — same host/core/config)
- **PACKED serving = 2,375,236 ns/fwd → 13,472 tok/s = 5.8× the f64 path**
  (2,316 tok/s) and **3.8× mode 2** (3,580) — the whole-model number now reflects
  the §1 layer-sweep advantage instead of per-call prep; the residual 2.4 ms is
  dominated by the shared f64 interludes (RMSNorm/attention/GELU, kept f64 per
  b1.58) + the T×V head.

### Added — `examples/quickstart.cyr` (the worked consumer example)
- Train a tiny ternary LM → `tx_int_init`/`tx_pack_init`/`tx_pack` once →
  `tx_fwd_packed` serving, **public API only** (no internal-symbol reach-in),
  **self-checking** (packed argmax must equal the scalar integer kernel's at every
  position; non-zero exit on mismatch — running the example IS a test). Verified:
  builds + runs + PASS. (Placed at repo-root `examples/` per the tarka/anukūlana
  house convention; the roadmap's `docs/examples/` phrasing predates it.)

### Tests (95 → 97)
- Packed serving logits **bit-identical** to the scalar integer path on the
  trained M3c model, and **reproduced on a second serve** from the same store
  (serve-time read-only).

## [0.6.0] — 2026-07-06

**Allocation-clean + API freeze** (`docs/api.md`, the "API frozen" v1.0 criterion).
No behavior change to any shipped path — the cut settles and documents the surface,
closes the last indirect-only test coverage, and records the allocation audit.
Suite **90 → 95**.

### Added — `docs/api.md` (the frozen public surface)
- Every public symbol documented with its contract + "tested by" driver, in the
  tarka api.md house pattern: freeze policy (stable through 1.x from the 1.0.0 cut;
  minor versions add only), the **not-frozen internal mechanism** list (STE masks,
  FD probes, sublayer chains, `lm_train`-local helpers, scratch globals), caller
  conventions (everything-is-i64, no bounds checks, `b = 0` = no bias), and the
  documented output cells (`tx_logits`, `ki_logits`, `mlp_out`).
- **Allocation & lifetime contract** (the audit result): allocation happens ONLY
  in `blk_init`/`tx_init`/`tx_int_init` + the one-shot trainers (`lm_train`,
  `tx_train`); **every forward/backward/kernel path is allocation-free** — audited
  by walking all 32 `t_alloc`/`alloc` sites outside the demo driver. Repeated
  inference costs zero allocation; re-calling an `*_init` re-points its scratch
  family (bump allocator — old buffers unreachable, not freed).

### Tests (90 → 95) — the API-freeze gates
- **`ref_dot`** exact-value gate (the M0 baseline was demo-only).
- **`bl_forward_q` direct** (was reachable only through `tx_fwd_q`): mode 0 vs
  mode 2 — same γ returned + outputs bit-identical through the dispatcher.
- **The manual training-loop quartet** (`tx_zero_grads → tx_loss → tx_bwd →
  tx_sgd`) composed by the caller from a fresh init: 5 steps descend the loss,
  finite — gating the public surface `tx_train` wraps (`tx_sgd` was previously
  exercised only inside `tx_train`).

### Classified internal (not in the freeze)
- `lm_init`/`lm_eval_sweep`/`lm_correct` (operate on `lm_train`-local buffers),
  the FD probes + STE masks, the sublayer chains (`*_sublayer_*`, `block_q`),
  numeric helpers. Readable + FD-gated; may change between minors.

## [0.5.0] — 2026-07-06

**Benchmarks — the honest numbers behind the claims** (`docs/benchmarks.md`, a v1.0
criterion). The `tests/tentib.bcyr` stub is now the real harness driving every
number; B-series fairness throughout (identical shapes, warm cache, single pinned
CPU, every path pays its per-call prep, ratios-not-nanoseconds reproduction).
Run of record: cyrius 6.4.10, AMD Ryzen 7 5800H (Zen 3), 2026-07-06.

### Added — `docs/benchmarks.md` + the real bench harness
- **§1 per-layer kernel sweep** (M=1, pack-once SIMD): the SIMD advantage **grows
  with size** — vs the rosnet f64-SIMD matmul **5.0× at 128×128 → 18.4× at
  768×768** (3,124 → 29,361 ns/call; vs branchless scalar 29.5× → 123.1×). Two
  compounding mechanisms documented: 16 int8 lanes vs 2-wide SSE2, and i8 weights
  staying L2-resident where f64 spills (590 KB vs 4.7 MB at 768²).
- **§2 prep amortization**: `tsimd_pack_w` ~4 ns/weight, paid once — amortizes
  after ~81 calls at 768²; `act_quant_int` 8.3 µs/row.
- **§3 whole-model tok/s** (b1.58-shaped F=4C, 267k params, T=32): f64 2,307 →
  scalar-int 456 → **SIMD-int 3,549 tok/s (1.5× over f64)** — with the honest
  note that mode 2 re-quantizes AND re-packs per call (fairness mirror of the f64
  path's per-call quantization), so the pack-once deployment entry point (the
  0.7.0 deliverable) is bounded by §1, not §3.
- **§4 memory story measured**: 1,835,008 B f64 → 57,344 B 2-bit packed (32×) at
  the bench config; zero inner-loop multiplies.
- **§5 quality delta (honest)**: matched-size, same-corpus, same-steps f64 sibling
  — **attn11 1.13.0 (324 params) final CE 0.006 vs tentib ternary (252 params)
  CE 0.11**; both solve the task, ternary plateaus ~20× higher at toy scale;
  differences beyond quantization enumerated (bundled gap, not an ablation);
  b1.58 paper (arXiv:2402.17764, parity from ~3B) + bitnet.cpp (~2–6× over fp16)
  **cited, not re-demonstrated**.

### Changed
- `tests/tentib.bcyr`: noop stub → the benchmark harness (layer sweep ×4 shapes,
  prep costs, whole-model tok/s; `bench_new`/`bench_run`/`bench_avg_ns` +
  fnptr-dispatched kernels; random tokens use the masked `rng_u64` idiom).
- `cyrius.cyml` stdlib gains `fnptr` (the bench harness dispatches via `fncall0`).
- Suite unchanged at **90/90** (re-validated with the manifest change).

## [0.4.1] — 2026-07-06

**The integer-SIMD ternary kernel — "multiply-free is also FASTER."** The 0.4.0 gate
lifted: cyrius **6.4.6/6.4.7** (SIMD arc Phase 3) shipped integer SIMD, including
`iv_dp8(a, b, n)` — the widening u8·i8 → i32 dot (`pmaddubsw` → `pmaddwd` → `paddd`,
16 int8 lanes/instruction) that the filed proposal
([`docs/development/proposals/2026-06-23-cyrius-integer-simd.md`](docs/development/proposals/2026-06-23-cyrius-integer-simd.md))
asked for. The ternary kernel now **beats the f64-SIMD matmul it was ~5× behind**,
while staying **bit-identical** to the scalar kernel. Suite **86 → 90**.

### Added — the SIMD kernel (`src/kernel.cyr`)
- **`tsimd_pack_w(Wq, K, N, w8t, wsum)`** — the "quantize + pack once" deployment
  step: transposes `Wq` [K,N] (i64 ternary) to **`w8t` [N,K] i8 bytes** (each
  output's weights contiguous, as `iv_dp8` needs) + per-output **`wsum`** over the
  SIMD span (the u8-offset correction term). The i8 sibling of `tpack2`.
- **`ternary_matmul_free_simd(qx, sx, w8t, wsum, γ, b, y, M, K, N, xu8)`** — same
  contract + **bit-identical output** vs `ternary_matmul_free`. `pmaddubsw` is
  u8 × i8, so the signed int8 activation codes are offset to u8 (`+128`, exact
  algebra: `dot(x+128, w) = dot(x, w) + 128·Σw`, corrected by `wsum`); the `K % 16`
  tail runs the M3b branchless mask arithmetic on the same `w8t` bytes. Exactness
  argument documented in-file: ternary weights bound each `pmaddubsw` pair-sum by
  510 ≪ 32767 (the i16 stage never saturates), i32 accumulate safe to K ~ 2²⁴.
- **`bl_forward_q` mode 2** — whole-model SIMD inference through all 7 BitLinears
  (re-packs per call, matching mode 0's re-quantize-per-call fairness); new scratch
  `ki_w8t`/`ki_wsum`/`ki_xu8` in `tx_int_init`.

### Benchmarks (128×128 layer, M=1, 100 iters/kernel, x86-64 host, cyrius 6.4.10)
- **SIMD int8 kernel = 1,946 ns/call** vs **rosnet SIMD-f64 `linear_fwd` = 14,670 ns**
  — **~7.5× faster than the f64 baseline** (the 0.4.1 acceptance: the *"multiply-free
  is also faster"* demonstration). vs branchless scalar (M3b) 87,254 ns = **~45×**;
  vs branchy 102,652 ns = **~53×**. The 0.4.0 standings (scalar ternary ~5× *slower*
  than f64-SIMD) are inverted.

### Tests (86 → 90, all green)
- **M3d bit-identity** across the three K regimes: K=4 (K16=0, pure branchless
  tail — the whole-model C=4/F=8 shape), K=20 (SIMD span + tail, correction split),
  K=32 (pure `iv_dp8`, two lane-groups).
- **M3d whole-model**: `tx_fwd_q(mode 2)` logits **bit-identical** to the scalar
  integer path (mode 0) through all 7 BitLinears incl. the biased head.

### Changed
- **Cyrius pin 6.2.37 → 6.4.10** (`cyrius lib sync --full`; the 86/86 baseline
  re-validated green at the new pin before the kernel work — the bump crosses the
  6.3.13 array-locals-to-stack boundary, no fallout).
- Demo driver (`src/main.cyr`) M3/M3c sections print the SIMD parity + timing rows;
  the stale "toolchain does not yet expose integer SIMD" gate-note replaced.

## [0.4.0] — 2026-06-23

**M3 — the matmul-free integer inference kernel, whole-model.** The payoff: ternary
weights {−1, 0, +1} × int8 activations run as integer signed-accumulate (add / sub /
skip — *no multiply*) through **every BitLinear of the trained transformer**,
reproducing the f64 forward's logits. The scalar throughput lever (branch elimination)
lands here; the SIMD lever is gated on toolchain integer-SIMD (→ 0.4.1). Suite **86/86**.

### Added — M3a: the matmul-free integer kernel
- `src/kernel.cyr` — the payoff. `ternary_matmul_free(qx, sx, Wq, γ, b, y, M, K, N)`:
  ternary weights + int8 activations, **integer signed-accumulate** in the inner loop
  (`+qx / −qx / skip` over i64 — *no multiply*), then one dequant per output (`γ·sx`) +
  optional bias. `act_quant_int` produces the int8 codes (same quantization as M1's
  `bl_act_quant`, kept integer). The M0 `ternary_dot` generalized to a full layer +
  carried to genuine integers.
- **Logit parity** (gated): the integer kernel reproduces `bl_forward_full`'s f64
  logits to **< 1e-9** (the integer accumulate is exact; the f64 path is what rounds)
  — a 128×128 demo shows `max|int−f64| < 1e-12`.
- **2-bit packed ternary storage** (gated round-trip): `tpack2`/`tunpack2` pack
  `{−1,0,+1}` to 2 bits/weight — **32× smaller** than an f64 slot.

### Added — M3b: branchless scalar kernel (the achievable throughput lever)
- `ternary_matmul_free_bl` — the same matmul-free accumulate, but the data-dependent
  per-weight branch (unpredictable on random ±1/0 weights) is replaced by mask
  arithmetic (`pm = −(w==+1)`, `nm = −(w==−1)`, `acc += (qv & pm) − (qv & nm)`). Still
  genuinely multiply-free (AND / ADD / SUB only). **Bit-identical** to the branchy
  kernel (gated).
- **Measured (128×128 layer):** branchy 112.6 µs → branchless **84.5 µs** (~25%
  faster). rosnet's SIMD-f64 matmul is 16.5 µs — branch elimination helps but cannot
  beat SIMD.
- **The int-SIMD gate (honest):** the b1.58 CPU throughput win over SIMD-f64 needs
  INTEGER-SIMD lanes; this Cyrius has f64-only SIMD (2-wide SSE2, `f64v_*` →
  `movupd`/`mulpd`/`addpd`). Filed as a cyrius proposal; the SIMD kernel is **0.4.1**,
  gated on it. Today's wins: multiply-free + branch elimination + 32× memory.

### Added — M3c: whole-model integer inference
- `tx_fwd_q` (+ `bl_forward_q` / `attn_sublayer_q` / `mlp_sublayer_q` / `block_q` /
  `tx_int_init` / `logits_argmax`, `src/kernel.cyr`) — the matmul-free kernel runs
  through **every BitLinear of the trained transformer** (attention Q/K/V/O + MLP
  up/down + the head with bias); RMSNorm / GELU / attention stay f64 (b1.58 ternarizes
  only the linear layers). One mode-parameterized chain serves both the integer payoff
  (mode 0) and a full-quant-f64 reference (mode 1).
- **Whole-model parity** (gated): the integer logits reproduce the full-quant f64
  forward to **< 1e-9 relerr** across all 7 BitLinears (the M3a per-layer parity,
  composed); a trained "hello world" model shows `max|int−f64| < 1e-12`.
- **Honest activation-quant delta:** vs the weight-only **trained** model (which never
  saw int8-quantized activations), `max|int−trained| ≈ 0.01` — and the integer model's
  **next-token argmax matches the trained f64 model at 10/10 positions** (the small
  logit delta doesn't flip the rank). The honest small-scale gap, reported not hidden.
- **Latency:** the integer whole-model forward is 24 µs vs 16.5 µs weight-only f64
  (slower without integer-SIMD — the 0.4.1 gate).
- Suite **82 → 86**.

### Fixed
- `ternary_matmul_free` gained a bias param (8 → 9) for the M3c head; three stale
  8-arg call sites compiled clean (Cyrius does not check call arity) but failed the M3
  parity test at runtime — call sites corrected.

### Filed (toolchain)
- **cyrius issue** — call-site arity mismatch silently accepted (no arg-count check):
  compiles `OK`, binds garbage. The M3c bias-param regression above is the trigger.
  [`docs/development/issues/2026-06-23-cyrius-call-arity-no-check.md`](docs/development/issues/2026-06-23-cyrius-call-arity-no-check.md).
- **cyrius proposal** — integer SIMD (typed `iNxM` vectors + int8/16/32 ops), the
  quantized-ML throughput floor; unblocks the 0.4.1 int-SIMD kernel.
  [`docs/development/proposals/2026-06-23-cyrius-integer-simd.md`](docs/development/proposals/2026-06-23-cyrius-integer-simd.md).

## [0.3.0] — 2026-06-23

**M2 — a ternary transformer trains from scratch.** The full milestone: BitLinear
assembled into a pre-norm transformer block (RMSNorm + causal attention + GELU-MLP,
all six linear projections ternary), trained end-to-end via the STE on real
akshara-tokenized text. Every component *and* the full assembly is finite-difference
gated; the model trains (CE → ~0.1). The ternary sibling of attn11's first loss
curve, built assembly-up in sovereign Cyrius on rosnet — no BLAS / libc / autodiff.

### Added — M2: a ternary transformer trains from scratch
- `src/model.cyr` — softmax cross-entropy (`softmax_xent_fwd`/`_bwd`, ported from attn11; rosnet has no softmax) + a minimal trainable ternary LM: token id → **full-precision** embedding row → **BitLinear head** (ternary, weight-only) → softmax-CE. SGD walks the *latent* head weight (quantized only in the forward — the BitNet shadow-weight loop) + the full-precision embedding/bias.
- **Trains from scratch** on a synthetic successor-bigram task (`x → (x+1) mod V`): CE descends from the uniform `ln 8 ≈ 2.08` baseline to **~0.57** (2000 steps), memorizing **7/8** successors by argmax — the ternary sibling of attn11's first loss curve, at honest tiny scale (the persistent 1/8 mis-rank is the ternary capacity limit, not hidden).
- **FD-gates** (suite 34 → **57**): softmax-CE `dlogits` == central FD; the **end-to-end** head `dW` == FD of the linearized surrogate *with the real softmax-CE gradient frozen* (the M1 STE composes with the loss); saturated entries mask to 0 *through the loss*; the **embedding gradient** == FD of the live loss (the full chain); falsifier (FD-through-the-quantized-loss diverges > 0.5).
- Embedding stays full-precision (b1.58 ternarizes only linear layers).
- `src/layers.cyr` — differentiable transformer layers, each FD-gated standalone (suite 57 → **63**):
  - **RMSNorm** (pre-norm, the BitNet/Llama choice — no mean-subtraction; backward = LayerNorm minus the centering term, cross-checked vs attn11's `ln_bwd`).
  - **GELU** (tanh approximation, ported from attn11; `tanh` implemented from `f64_exp` since `f64_tanh` is not a builtin in this toolchain).
  - **Causal scaled-dot-product self-attention** (`attn_fwd`/`attn_bwd`) — single-head, `score = Q·K/√d`, causal softmax, value-sum; the full backward (`dQ`/`dK`/`dV` incl. the softmax-attention `P·(dP − ΣP·dP)` term) cross-checked vs attn11's `attn_core_bwd`, all three FD-gated.
- `src/block.cyr` — the **full ternary transformer block, assembled and end-to-end FD-gated** (module-global scratch + caches; suite 63 → **70**). Each level's dx gate (`dx == central FD` of `½‖out‖²`, exact since quantizers freeze under an x-perturbation) validates the entire fwd+bwd chain, catching any wiring bug:
  - **E1 — attention sublayer**: `RMSNorm → BitLinear Q/K/V → causal attention → BitLinear O → +residual` (whole-sequence M=T projections; the three Q/K/V `dx` sum; residual fold inside the sublayer bwd).
  - **E2 — MLP sublayer**: `RMSNorm → BitLinear up(C→F) → GELU → BitLinear down(F→C) → +residual`.
  - **E3 — full pre-norm block** = E1 ∘ E2; the per-sublayer residual fold makes `block_bwd` just thread `dout → dxmid → dx` (the rolling residual gradient). 6 BitLinear projections + 2 RMSNorms + GELU + attention + 2 residuals, all gradient-verified.
- **bite-E4: the ternary transformer LM trains from scratch — the "first loss curve."** token-embed + learned absolute pos-embed (full-precision) → the block → final RMSNorm → a *separate* ternary BitLinear head → per-position next-token softmax-CE; SGD on the latent weights + full-precision embeds/gains.
  - **E4a** — the full LM forward/backward, **end-to-end FD-gated**: `d_tokemb`/`d_posemb == FD` of the CE loss through the entire chain (embed scatter → block → final RMSNorm → head → softmax-CE).
  - **E4b** — **trains**: on a synthetic next-token sequence, CE descends `ln 6 ≈ 1.79 → ~0.04` (1500 steps) — the ternary sibling of attn11's first loss curve. Suite → **76**.
- **bite-F: trains on a REAL akshara-tokenized corpus.** Wired `[deps.akshara]` 0.1.0 (the shared sovereign tokenizer); `corpus_set(text,len)` byte-tokenizes `"hello world"` → V=8, `gd_ld(i)` reads the token ids, and the ternary transformer trains next-token on it: CE `ln 8 ≈ 2.08 → 0.11` (2000 steps). One tokenizer behind attn11, tarka, and tentib.

## [0.2.0] — 2026-06-23

**M1 — BitLinear + the straight-through estimator, finite-difference-gated.** The
matmul-free thesis becomes a *trainable* layer: ternary weights + int8 activations
on rosnet's matmul, the STE on the backward, and the FD-gate proving the surrogate.

### Added — M1: BitLinear + the straight-through estimator
- `src/bitlinear.cyr` — **BitLinear**, a BitNet b1.58 quantization wrapper around rosnet's f64 matmul:
  - **Forward**: per-tensor absmean ternary weight quant (`bl_weight_effective` → `Wq ∈ {−1,0,+1}`, `Weff = γ·Wq`) + per-token absmax **int8 activation quant** (`bl_act_quant`). Both scales bake into the dequantized operands, so the matmul (rosnet `linear_fwd`) needs no trailing rescale. `bl_forward_wonly` (weight-only) / `bl_forward_full` (both quantizers). `γ==0` and `absmax==0` guards.
  - **Backward (STE)**: `bl_backward_wonly` + `bl_ste_mask_w` / `bl_ste_mask_x`. `dW` straight-through to the latent weight with the clip-mask (`|W/γ|<1`) and **no γ factor** — the γ from `Weff=γ·Wq` and the 1/γ from `W/γ` cancel; `dx` through the real forward weight `Weff`; `db` identity. One `linear_bwd` call carries it; the masks re-route the quantizers' gradients.
- `[deps.rosnet]` 0.2.0 + `[deps.tyche]` 0.1.1 wired (rosnet's `t_randn` → tyche's `rng_normal`).
- **Finite-difference gate** (`tests/tentib.tcyr`) — the headline correctness story, *prove the surrogate, not the discontinuity*: analytic STE `dW` == FD of the linearized surrogate on unsaturated entries; `dx` == FD through `Weff`; **γ-cancellation exercised at γ=3** (a spurious γ factor would fail by ×γ); **falsifiers** — FD through the *quantized* forward diverges (relerr > 0.5), and a masked (saturated) entry's surrogate FD is nonzero; STE-SGD reduces the quantized loss. Suite **34/34**.
- `src/main.cyr` demos a BitLinear layer alongside the M0 quantizer.

### Notes
- **Honest scale**: b1.58 "matches fp16" is a ≥3B result; at the tiny from-scratch scale the ternary error is large (visible in the demo) — reported, not hidden.
- The packed-ternary **matmul-free inference kernel** (M3) and a ternary transformer **trained from scratch** over an akshara corpus (M2) remain ahead.

## [0.1.0] — 2026-06-23

**M0 — scaffold + the ternary quantizer.** `tentib` (`bitnet` reversed) opens as
the sovereign **integer-native / ternary (1.58-bit)** ML reference — attn11's
everything-is-i64 thesis applied to the weights: weights collapse to {−1, 0, +1},
so the weight·activation path is **add / subtract / skip**, no multiply.

### Added
- `cyrius init` scaffold (pin 6.2.37), GPL-3.0-only; doc-tree per first-party docs.
- `src/ternary.cyr` — the M0 core, self-contained on raw f64 (no external deps):
  - `absmean` — the b1.58 per-tensor scale γ = mean(|wᵢ|).
  - `ternary_one` / `ternary_quantize` — quantize weights to {−1, 0, +1} via
    `clip(round(w/γ), −1, +1)`.
  - `ternary_dot` — the **matmul-free** signed-accumulate dot (`+x / −x / skip`,
    then ·γ), against `ref_dot` (full-precision baseline).
- `src/main.cyr` — demo driver: quantizes an 8-vector, prints the ternary weights,
  γ, and the matmul-free dot vs the full-precision dot with the error.
- `tests/tentib.tcyr` — correctness suite (**6/6**): absmean γ, the four ternary
  values incl. the zero state, and matmul-free dot == γ·(signed sum).
- Scope + naming ADR (`docs/adr/0001-tentib-scope.md`), roadmap, state.

### CI / Release
- `ci.yml` declares `on: workflow_call:` so `release.yml`'s `uses: ./.github/workflows/ci.yml`
  gate resolves (the generated scaffold omitted it — a tag push would otherwise fail the CI
  gate before building).
- Both workflows install the toolchain via the upstream `scripts/install.sh` (lays out
  `~/.cyrius/versions/<v>/`) instead of the generated flat `curl+cp`-into-`~/.cyrius/{bin,lib}`
  form — the flat layout fails `cyrius deps`'s pin-check / `cyrius lib sync` once the
  rosnet/tyche/akshara git deps land at M1. Matches tarka (same dep profile) / patra.

### Notes
- Sibling reference to attn11 (Transformer) and tarka (RL/reasoning); stands on
  the same f64 substrate (rosnet / tyche / akshara) — those deps wire in at M1+.
