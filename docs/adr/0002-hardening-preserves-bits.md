# 0002 — Hardening and optimization preserve bits; guards sit at call entry, never in kernel loops

**Status**: Accepted
**Date**: 2026-11-09

## Context

tentib's claims are exact ones: the integer kernels are bit-identical to each
other, the whole-model integer path reproduces the full-quant f64 reference, and
packed serving reproduces both. The 1.1.2 P(-1) sweep changed hot code in the
quantizer, the packer and the activation quant for speed, and added input checks
for defects that corrupted the heap (out-of-range token ids) or returned wrong
numbers. Two questions needed a rule rather than a case-by-case call: what may an
optimization change, and where may a check go?

The 0.8.0 audit had already set a posture for the second (fail-loud `guard()` on
public-API preconditions, cold paths only, no bounds checks on caller buffers).
1.1.2 extended the guarded set to data — token and target ids — and to tentib's
own init-owned state (the dims each call passes against what its `*_init`
allocated), and it measured one check that would have sat inside a kernel loop.

## Decision

**An optimization or refactor must leave every output bit-identical on valid
input, proven by a golden dump** — raw bit patterns of every public function at
fixed seeds, recorded before the change and compared byte for byte after. A
change that alters any value is a behavior change and is recorded as a fix, not
an optimization. Reassociating f64 arithmetic, replacing a divide by a reciprocal
multiply, or changing a comparison's boundary is therefore out; hoisting address
arithmetic, inlining, reordering integer sums, and precomputing the same f64
operation once instead of per element are in.

**Guards go at call entry or in cold passes, never inside a kernel's loops.**
Guarded: preconditions whose violation is silent heap corruption or silently-wrong
numerics and that the caller cannot detect downstream — non-ternary weights at pack
time, K past the SIMD exactness bound, call order and dims against the `*_init`
sizing, and ids that index a table (O(T) or O(M) once per call). Not guarded:
caller-supplied buffer sizes, element accessors, and per-element ranges inside a
kernel (documented as caller contracts instead).

## Consequences

- **Positive** — every speedup is provably free of numerical drift, which is what
  lets the suite's bit-identity gates stay exact; the golden dump also catches
  drift from toolchain and dependency bumps. The heap-corruption and
  wrong-gradient defects of 1.1.2 cannot recur silently.
- **Negative** — some speedups are off the table (e.g. `|W| < γ` for the STE mask,
  or a reciprocal in the quantizer); and a kernel called directly with codes
  outside [−128, +127] is still silently wrong, by contract.
- **Neutral** — the golden-dump tool lives outside the repo (the audit describes
  it); each sweep rebuilds it for the surfaces it touches.

## Alternatives considered

- **Tolerance-based equivalence for optimizations** (e.g. < 1e-12): rejected — it
  would hide a real reordering in the integer path, where the claim is bit-exact.
- **A per-row activation-code check inside `ternary_matmul_free_simd`**: measured
  at 4–10% of the kernel at M = 1 and rejected; every in-repo caller passes
  `act_quant_int`'s codes, which the fuzz harness checks (property P2).
- **Bounds-checking caller buffers**: rejected again (as at 0.8.0) — the rosnet
  substrate contract is caller-guarantees, and the checks would sit in hot loops.
