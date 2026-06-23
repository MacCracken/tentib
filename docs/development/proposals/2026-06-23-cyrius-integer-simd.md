# Integer SIMD — typed `iNxM` vectors + int8/16/32 vector ops (the quantized-ML throughput floor)

**Filed:** 2026-06-23 (by a tentib consumer — tentib 0.4.x M3b, the matmul-free
ternary inference kernel)
**Severity:** Language/stdlib gap — Cyrius SIMD is **f64-only**. `lib/simd.cyr`
exposes `f64v2`/`f64v4` over the `f64v_*` builtins; there are **no integer vector
types or ops**. This caps the throughput of every integer / quantized / bit kernel
at scalar speed.
**Affects:** tentib (ternary b1.58 inference — the headline), attn11 / tarka (int
paths), sankoch (compression), any future int8/int16 DSP, image (rasa / ranga),
bulk hash/crypto, and the kernel's `memcpy`/`memset`-class ops. Also every edge /
Pi-ARM tok/s story (seema).
**Target slot:** a v6.x language/stdlib feature — maintainer direction. **Not a
release blocker:** tentib ships **0.4.0** with the scalar matmul-free kernel; this
unblocks tentib **0.4.1** (the int-SIMD kernel) and the b1.58 *"multiply-free is also
faster"* claim.
**Template:** the existing typed-SIMD **f64** arc (v5.10.x ABI Phases 1–11) is the
precedent — same value/pointer dual-form, lane extractors, SysV/aarch64 reg
conventions, `bench`-gated correctness.

## Trigger

tentib M3b — the matmul-free ternary kernel. Ternary weights `{-1, 0, +1}` × int8
activations collapse `weight·activation` to integer **add / subtract / skip** — no
multiply. On paper that should beat an f64 matmul. Measured on a 128×128 layer, this
toolchain (cyrius 6.2.37, host x86-64):

| kernel | ns/call | note |
|--------|--------:|------|
| branchy scalar (M3a) | 112 600 | data-dependent branch per weight |
| **branchless scalar (M3b)** | **84 484** | mask arithmetic, no branch — 25% faster |
| rosnet SIMD-f64 matmul | **16 516** | `f64v_fmadd`, the baseline to beat |

**The scalar integer kernel is ~5× SLOWER than the f64-SIMD matmul.** The multiply
is gone; the throughput is bounded by **ops per instruction**, and there is no
integer SIMD to widen the integer path. Branch elimination is the only achievable
scalar lever, and it does not close the gap.

The asymmetry is even sharper than "f64v4" suggests. Inspecting the 6.2.37 backend
(`src/backend/x86/float.cyr`), `f64v_add`/`mul`/`fmadd`/`dot` lower to **SSE2
packed-double** (`movupd` / `mulpd` / `addpd`) and step the loop counter by **2** per
iteration regardless of the requested width — so `f64v_fmadd(..., 4)` is *two* 2-wide
SSE2 iterations, and the "FMA" is a separate `mul` + `add`, not a fused
`vfmaddpd`. The baseline that beats the ternary path ~6× is **merely 2-wide SSE2**.

## The gap

b1.58's CPU speedup (e.g. **bitnet.cpp**'s `TL1` / `TL2` / `I2_S` kernels — ~2–6×
over fp16) comes entirely from **integer SIMD**: pack 16–32 int8 lanes into one
vector register and use sign-select / shuffle-lookup / VNNI dot-product
(`vpdpbusd`, `vpmaddubsw` + `vpmaddwd`, `pshufb`). The *defining* property of low-bit
inference — that low-precision integers pack more lanes per register — is
**inexpressible** in Cyrius today. Ternary on Cyrius currently buys:

- ✅ multiply-free accumulate (the arithmetic-floor thesis — proven),
- ✅ 32× weight memory (2-bit packed ternary vs f64 — proven),
- ❌ **no wall-clock throughput win** (blocked here).

## Proposed surface (sketch — maintainer to shape)

Mirror the f64 typed-SIMD design, but for integers. Two tiers:

**1. Typed integer vector values + lane-wise ops** (the f64v2/f64v4 analog):

```
i8v16  / i8v32     i16v8 / i16v16     i32v4 / i32v8     i64v2 / i64v4
```

with value- and pointer-form wrappers (the `&IDENT → _ptr` dispatch already exists)
over `iNv_*` builtins: lane-wise `add` / `sub` / `and` / `or` / `xor` / `shl` /
`shr`, `cmpeq`/`cmpgt` → mask, `blend`/`select`, `broadcast`, lane extractors.

**2. The quantized-ML primitives** (what actually closes the throughput gap):

- **widening multiply-add**: `i8 × i8 → i16` accumulate, `i16 → i32` accumulate
  (the `vpmaddubsw` + `vpmaddwd` / VNNI `vpdpbusd` shape).
- **horizontal reduce** (`i16`/`i32` lane-sum).
- for the **ternary** path specifically: an int8 load, **sign-select by a ternary
  mask** (or split positive/negative masks), int16/int32 accumulate, horizontal sum
  — i.e. `acc += select(w>0, x, 0) - select(w<0, x, 0)` vectorized 16–32 lanes wide.

The ternary kernel never needs a general int×int multiply — only sign-select +
widening accumulate — so a minimal first cut (int8 load, mask/blend, int16 add,
hreduce) already unblocks tentib 0.4.1 without the full VNNI surface.

**Portability:** like the f64 arc, define the builtins arch-neutrally and lower per
target (SSE2/AVX2/AVX-512-VNNI on x86-64; NEON `sdot`/`smlal` on aarch64). This is
also the path off the current x86-64-only float story.

## What it unblocks

- **tentib 0.4.1** — the int-SIMD ternary kernel: the b1.58 *"multiply-free is also
  faster"* demonstration, benchmarked vs a named reference (bitnet.cpp / the b1.58
  paper) under the B-series fairness rules. The headline integer-native sovereignty
  flex (a real LLM tok/s on the 1.5x Pi-ARM line under tight power) depends on it.
- **Ecosystem int throughput** — compression (sankoch), image (rasa/ranga), DSP,
  bulk hash, and the kernel's byte-move ops all currently run scalar.

## Honest scope note

This is the throughput **floor** for quantized ML on Cyrius. Without it, tentib's
matmul-free claim stays *correct* (the multiply is genuinely eliminated) but its
*speed* claim is bounded by f64-SIMD parity-at-best (today, 2-wide-SSE2-parity-at-
best). With it, ternary realizes the lane-packing advantage that is the entire point
of low-bit inference — the difference between "no multiply" as a curiosity and as a
deployment win.

## References

- **b1.58** — Ma et al., *The Era of 1-bit LLMs* (2024), arXiv:2402.17764.
- **bitnet.cpp** (Microsoft) — `TL1`/`TL2`/`I2_S` CPU kernels; the converged
  integer-SIMD packing/lookup shape.
- Intel **VNNI** `vpdpbusd`, AVX2 `vpmaddubsw` + `vpmaddwd`, `pshufb`; aarch64 NEON
  `sdot` / `smlal`.
- The cyrius **typed-simd f64 arc** (v5.10.x ABI Phases 1–11, `lib/simd.cyr`) — the
  structural template for the value/pointer dual-form + the ABI work.

---
*Mirrored consumer-side at `tentib/docs/development/proposals/2026-06-23-integer-simd.md`.*
