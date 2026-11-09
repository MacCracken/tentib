# Contributing to tentib

## Development

1. Install the Cyrius toolchain at the version pinned in `cyrius.cyml`
   (`[package].cyrius`). The pin is the single source of truth — never hardcode
   a version elsewhere.
2. `cyrius deps` — resolve stdlib + the sibling git deps (rosnet / tyche /
   akshara) into `lib/`.
3. `cyrius build src/main.cyr build/tentib` — compile the demo driver
   (M0 quantizer · M1 BitLinear · M2 transformer-trains · M3 matmul-free kernel).
4. `./build/tentib` — run the demo.
5. `cyrius test` — run the suite (`tests/tentib.tcyr`: the FD gates, the
   bit-identity gates, and every guard's fail path in a forked child) plus the
   `[build].test` entry (`src/test.cyr`).
6. `cyrius build tests/tentib.bcyr build/bench && taskset -c 4 ./build/bench` —
   the benchmark harness behind `docs/benchmarks.md`. For a before/after
   comparison run `./scripts/bench-history.sh` before and after the change: it
   appends the median of three pinned runs to `bench-history.csv`.
7. `CYRIUS_DCE=1 cyrius build tests/tentib.fcyr build/fuzz && ./build/fuzz` —
   seeded property fuzzing (5,000 rounds, a few seconds; exit 1 names the failed
   property and round).

See [`CLAUDE.md`](CLAUDE.md) for the full development loop and
[`docs/development/roadmap.md`](docs/development/roadmap.md) for the open work (1.2.x – 1.x.x and
the 2.0.0 backlog).

## Finite-difference checks are the contract

tentib has **no autodiff** — every backward is hand-derived, and a hand-derived
gradient is the part most likely to be subtly wrong. Every backward function MUST
be verified against central finite differences in `tests/tentib.tcyr` before it
lands. A new op without a passing FD check is incomplete.

The discontinuous quantizers are the subtlety: you **cannot** finite-difference
through `round` / `clip`, so the rule is **prove the surrogate, not the
discontinuity** — the straight-through-estimator path must match the gradient of
the non-quantized linearized surrogate (with the γ-cancellation exercised and a
falsifier that diverges if the FD is taken through the *quantized* forward
instead). The end-to-end block / LM gates verify the whole fwd+bwd chain by
checking `dx == central FD` of a scalar loss, which freezes the quantizers under
the perturbation.

## Numeric rules

- Cyrius has no float type — an `f64` is its bit pattern in an i64. Use the
  `f64_*` builtins, never `+` / `*` on float values.
- Build precise constants from integer ratios or runtime math; long-digit float
  literals mis-parse.
- Latent weights stay full-precision f64; quantization happens only inside the
  forward pass (the BitNet shadow-weight loop). The optimizer updates the latent
  weights; the forward re-quantizes ternary each step. No allocation inside the
  training loop.

## Process

- One change at a time. Never bundle unrelated changes in a single PR.
- Test after every change; FD-check after every backward-touching change;
  benchmark after every perf-touching change.
- Run the suite after a signature change. cyrius rejects a call with the wrong
  number of arguments (every toolchain from 6.6.0 on; it did not at 6.2.37, see
  `docs/development/issues/2026-06-23-cyrius-call-arity-no-check.md`), but a
  call with the right count and swapped arguments still compiles clean.
- Optimizations must be bit-identical on valid input, and guards go at call
  entry, never inside a kernel loop
  ([ADR 0002](docs/adr/0002-hardening-preserves-bits.md)).
- Performance claims must include numbers — `before → after` with the bench name.
- Honest-scale rule: b1.58's "matches fp16" is a ≥3B *scale* result; at tiny
  from-scratch scale, **report the quality gap**, never hide it.
- Breaking changes get a `Breaking` section in [`CHANGELOG.md`](CHANGELOG.md)
  with a migration paragraph.
- Do not commit/push or use `gh` — the maintainer handles git operations.

## License

GPL-3.0-only.
