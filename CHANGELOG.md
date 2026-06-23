# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
