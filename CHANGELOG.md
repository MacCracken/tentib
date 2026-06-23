# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

### Notes
- Sibling reference to attn11 (Transformer) and tarka (RL/reasoning); stands on
  the same f64 substrate (rosnet / tyche / akshara) — those deps wire in at M1+.
