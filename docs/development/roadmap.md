# tentib — Roadmap

> **Self-contained SemVer plan.** An agent can pick up the next version from this repo
> alone. Volatile state (current version, test count, deps) is in [`state.md`](state.md).
> What already shipped (M0 → 1.0.0, all six v1.0 criteria met at 1.0.0, then 1.0.1–1.1.2)
> is in [`../../CHANGELOG.md`](../../CHANGELOG.md); this file lists **only open work**.
> The ecosystem map at agnosticos `docs/development/planning/integer-native-ml.md` is
> background, not required.

## What tentib is

**tentib** = `bitnet` reversed: the sovereign **integer-native / ternary (1.58-bit)** ML
reference. BitNet b1.58 weights collapse to **{−1, 0, +1}**, so `weight · activation`
becomes **add / subtract / skip, no multiply**. attn11 proved learning is expressible in
an everything-is-i64 language; tentib takes the i64 claim to the **weights themselves**.

Discipline (non-negotiable): **Cyrius-native, no BLAS / libc / autodiff;
finite-difference-gated** (prove the *surrogate*, not the discontinuity); **benchmarked
vs a named reference**; **honest-scale** (b1.58's "matches fp16" is a ≥3B result, so
report the small-scale quality gap, never hide it).

## What it stands on

- **rosnet** (f64 latent tensors + `linear_fwd` / `linear_bwd`), **tyche** (PRNG),
  **akshara** (the byte-vocab tokenizer).
- Latent weights stay f64; only the linear layers are ternary, quantized inside the
  forward. Embeddings, RMSNorm, GELU and the attention core stay f64.

## How a version is cut

- **SemVer under the 1.x freeze** ([`../api.md`](../api.md)): a minor **adds** symbols and
  never changes a frozen signature or semantics; a patch is fixes, docs and pin sweeps.
  Anything that breaks the freeze waits for **2.0.0** (backlog below).
- **Every change** follows [ADR 0002](../adr/0002-hardening-preserves-bits.md): an
  optimization is bit-identical on valid input (golden-dump proven), and guards sit at
  call entry, never inside a kernel loop. New backward code is FD-gated; new kernels are
  bit-identity-gated in the suite and the fuzz harness.
- Bump `VERSION` + the CHANGELOG section, all gates green (`cyrius audit` clean, suite
  and fuzz on both targets), then **the user commits and tags**.
- The ordering below is proposed. Each version is an independently cuttable bite; adjust
  granularity with the maintainer.

## 1.2.x — maintenance on the shipped serialization

1.2.0 shipped packed-model serialization (`tx_pack_serialize` / `tx_pack_deserialize`,
[`../architecture/001-packed-model-format.md`](../architecture/001-packed-model-format.md));
see the CHANGELOG. What remains in the 1.2.x line is maintenance.

### Maintenance — 1.2.x patches and onward

- **Re-baseline the benchmarks** (maintainer's call): `docs/benchmarks.md`'s tables, the
  README figures and `api.md`'s performance contract are the 2026-07-06 run of record.
  The dated 1.1.1 note (rosnet's faster f64 path) and 1.1.2 note (tentib's faster
  quantizer / packer) carry the current numbers.
- **Re-prove the GPU tile path on hardware** at the current pin: `/bin/gputern` last
  returned exit 95 (all tiles on the gfx90c, bit-exact) on 2026-07-23. CI covers the host
  fallback and the `--agnos` build, not the device.
- Toolchain / dep pin sweeps, per the established procedure in `state.md`.

## 1.3.x — the first consumer (hoosh / murti)

1.3.0 shipped the library bundle (`dist/tentib.cyr`, release assets, a CI drift check),
`tx_pack_dims`, and the CI-gated consumer example `examples/serve.cyr`; see the
CHANGELOG and [`../guides/consuming-tentib.md`](../guides/consuming-tentib.md).

**Open.** hoosh / murti declare `[deps.tentib]` and serve a model file through
`tx_fwd_packed`. That work lands in their repos. tentib's part is to answer what the
integration finds: a missing accessor, a contract that needs a sentence, or a bundle
that does not build under their pin (hoosh pins 6.7.18, as tentib does). Any additive
API it needs is a 1.3.x minor here.

## 1.4.x — serve from packed ternary storage

**Goal.** Make the memory story true at serve time. Serving now reads an i8 working
copy (1 byte per weight, 8× smaller than f64); the 2-bit form (`tpack2`, 32×) is storage
only.

**Scope.** Additive: a kernel that serves from 2-bit codes, or from 5 trits per byte
(≈1.6 bits, a `tpack3` / `tunpack3` codec), unpacking in-register or through lookup
tables in the bitnet.cpp TL1 / TL2 / I2_S shape (ported, never copied); a packed-store
variant of `tx_pack` / `tx_fwd_packed` that uses it. With it, a **serving-only
allocation for loaded models**: `tx_pack_deserialize` runs the full `tx_init` today, so a
load also allocates the latent weights, their gradients and the training scratch
(12,192,728 bytes for the 357,504-byte bench artifact). The artifact format can stay at
version 1; only what a load allocates changes.

**Gate.** Bit-identical to the scalar kernel (suite + fuzz property P1 extended to the
new kernel); benchmarked against the i8 path at the four layer sizes and whole-model,
reporting the memory/speed trade honestly if it is slower; resident weight memory
measured.

## 1.5.x — the edge target (Pi 5, for seema)

**Goal.** The Pi-class tok/s result the integer-native plan is aimed at. The SIMD kernel
already runs NEON on aarch64 (cyrius SIMD Phase 5, 6.4.32, lowers `iv_dp8` natively), and
the suite and fuzz harness pass under qemu in CI. Nothing has been measured on real ARM
hardware.

**Scope.** Benchmark on a Pi 5 (an aarch64 path for `scripts/bench-history.sh`), and size
a b1.58-shaped model the Pi can serve (with 1.2.0's format to ship it and 1.4.x's storage
to fit it).

**Gate.** `docs/benchmarks.md` gains an aarch64 section: host, pin, model size, tok/s, and
the f64 path on the same board for comparison.

## 1.6.x — GPU ternary kernels via mabda

**Goal.** A portable GPU path. Today `ternary_matmul_free_gpu` tiles the integer
accumulate onto the AGNOS `#82` dispatch (8×8 i32 tiles), which is AGNOS-only.

**Scope.** The ternary kernel through mabda (rosnet's GPU bundle is the precedent; also
ai-hwaccel), then whole-model offload of the seven BitLinears. Integer accumulation is
associative, so the result must be bit-identical to the CPU kernel on every device.

**Gate.** Device-verified bit-identity at ragged shapes; the no-device fallback contract
tested on a host (rosnet's `gpu_host` pattern).

## 1.x.x — later, unscheduled

- **A matched-recipe quality ablation.** `docs/benchmarks.md` §5 bounds a *bundled* gap:
  attn11 trains with Adam, an LR schedule and a 4C MLP; tentib with plain SGD and a 2C
  MLP. An optimizer (Adam) and schedule as additive `tx_*` entry points, then a re-run,
  would make the ternary-vs-f64 delta single-variable.
- **Research watch: the matmul-free token mixer** (Zhu et al., *Scalable MatMul-free
  Language Modeling*, 2024): a ternary recurrence in place of attention. A sibling
  experiment, not committed.

## 2.0.0 backlog — breaking changes (collected, not scheduled)

- **Model handles instead of module-global state.** Every family (`tx_*`, `blk_*`,
  `ki_*`, `pk_*`) is module-global, so a process holds one model. Every call re-passes
  V, T, C, F, which the 1.1.2 dims guards have to police. Re-calling an `*_init` leaks the
  previous buffers (the bump allocator never frees). Outputs land in globals
  (`tx_logits`, `ki_logits`). An arena-owned, freeable handle fixes all four and changes
  every `tx_*` signature.
- **Compute the STE weight mask in the forward.** `bl_backward_wonly` divides `W/γ` again
  for every weight, and the forward already computed it. Keeping it changes the frozen
  signature (a new entry point in a 1.x minor is the non-breaking alternative).
- **Full-precision numeric constants.** The GELU constant is `0.7978845608`, 10 digits of
  √(2/π) = 0.79788456080286…; every output's bits depend on it, so the exact double
  changes every output.

## Not on this roadmap (scope boundaries, [ADR 0001](../adr/0001-tentib-scope.md))

- **Rotation-PTQ of imported checkpoints** (QuaRot / SpinQuant): quantize-the-import is a
  different technique; it belongs with the Type-3 weight-import reference.
- **Scale claims**: b1.58's parity with fp16 at ≥3B params is cited, not re-demonstrated.

## References (for the open work)

- **b1.58**: Ma et al., *The Era of 1-bit LLMs* (2024), arXiv:2402.17764 (the recipe and the
  scaling claims to cite).
- **bitnet.cpp**: Microsoft's CPU inference kernels (TL1 / TL2 / I2_S), the packing and
  lookup shape for 1.4.x.
- **STE**: Bengio, Léonard, Courville (2013), arXiv:1308.3432, the surrogate the FD gates
  validate.
- **Scalable MatMul-free LM**: Zhu et al. (2024), arXiv:2406.02528 (the research watch).
- In-repo history: [`proposals/2026-06-23-cyrius-integer-simd.md`](proposals/2026-06-23-cyrius-integer-simd.md)
  (the 0.4.1 design, shipped).
