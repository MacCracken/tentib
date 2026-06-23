# 0001 — tentib scope, the integer-native thesis, and the name

**Status**: Accepted
**Date**: 2026-06-23

## Context

[attn11](https://github.com/MacCracken/attn11) proved that gradient-based learning
of a transformer is expressible in Cyrius's "everything-is-i64" model — hand-written
forward + backprop + Adam on raw `f64` arrays, no BLAS / libc / autodiff, gradients
gated by finite-difference checks. But attn11's arithmetic is still **f64 bit
patterns simulating real-valued math**. The honest next step in the *sovereignty*
direction is not a bigger model or a new modality; it is to make **the model's own
arithmetic integer-native** — and the cleanest expression of that is **BitNet
b1.58**: ternary {−1, 0, +1} weights, which make the weight·activation product
multiply-free (−x / nothing / +x), with int8 activations so accumulation is integer.

For a project whose founding claim is *"the sovereign machine is integer all the
way down,"* a **matmul-free ternary-weight LLM** is the most literal possible
statement of the thesis. It is also the path to a real LLM on the 1.5x Pi-ARM line
under tight power, feeding the seema edge-fleet endgame. The
[integer-native-ml.md](https://github.com/MacCracken/agnosticos/blob/main/docs/development/planning/integer-native-ml.md)
forward-design map (in agnosticos) lays out the full plan; this repo is its home.

## Decision

Stand up **tentib** as the sovereign **integer-native / ternary (1.58-bit) ML
reference**. Scope: weight ternarization (BitLinear), the straight-through
estimator (STE) for training through the non-differentiable quantizer, a ternary
transformer trained from scratch, and the packed-ternary + int8 **matmul-free
inference kernel** that makes "no multiply" a measured tok/s.

**The name.** `tentib` = **`bitnet` spelled backwards** — the English-wordplay /
trickster naming lane (per AGNOS naming conventions, English names are wordplay;
non-English are direct-semantic). It reads as a reference *binary* name, a sibling
to `attn11` and `tarka`, not a system-lib.

**Structural choice — sibling, not chain.** tentib does **not** depend on
attn11-the-binary. Both (and tarka) are consumers of the same substrate:
- **rosnet** — f64 tensor storage + matmul + gradient. The **latent** (full-precision)
  weights the optimizer updates live here; BitLinear quantizes them in the forward
  pass only. (Wires in at M1.)
- **tyche** — deterministic PRNG (weight init, optional stochastic rounding). (M1.)
- **akshara** — the shared tokenizer for the from-scratch training corpus. (M2.)

**M0 (v0.1.0) is deliberately dependency-free** — a self-contained ternary
quantizer + matmul-free dot on raw f64 (`src/ternary.cyr`), so the thesis is
demonstrable and tested before the substrate is wired.

## Consequences

- **Positive.** The most AGNOS-distinctive model-making lane gets a home: the i64
  thesis applied to the weights, a multiply-free hot path, an edge/Pi sovereignty
  demo. The **STE becomes the canonical finite-difference-gate case study** — you
  cannot FD-check through `round`/`clip`, so the discipline shifts to *proving the
  surrogate* (the STE path matches the non-quantized surrogate's gradient) — a
  reusable template for every future quantized / discrete op.
- **Negative / owned.** A packed-ternary storage format + an integer accumulate
  kernel are new surface to maintain. Quantization-aware training is finickier than
  full-precision (scale calibration, activation clipping).
- **Honest scope.** The b1.58 "matches fp16" result is a **scale** result (≈3B+
  params). At the tiny scale a sovereign reference trains from scratch, expect a
  measurable quality gap vs a same-size f64 attn11 — report it like attn11's MTP
  honest-negative, not as a loss. The *proof* is that ternary training converges
  and the integer kernel reproduces the model's logits; the *scaling claim* is
  cited, not re-demonstrated at 3B.

## Alternatives considered

- **Bury ternary inside attn11 as a flag.** Rejected — same reasoning as the
  tarka split (ADR pattern): it would bolt a second thesis onto attn11's single
  teachable one ("learning from the assembly up"). attn11 stays the Transformer.
- **Binary {−1,+1} (original BitNet) instead of ternary.** Rejected as the lead —
  b1.58's zero state buys feature-filtering and is the modern result; binary is a
  historical footnote, not the reference.
- **Post-hoc quantize an imported checkpoint (rotation-PTQ: QuaRot/SpinQuant).**
  A *different* technique (quantize-the-import), paired with the Type-3
  weight-import reference — kept separate from tentib's QAT-from-scratch scope.
- **Name `bitnet` / a Sanskrit system-lib name.** Rejected — `bitnet` is the prior
  art's name (not sovereign); a Sanskrit name is the system-lib lane, but tentib is
  a reference binary. `bitnet`-reversed keeps the lineage legible while staying in
  the English-wordplay binary lane.
