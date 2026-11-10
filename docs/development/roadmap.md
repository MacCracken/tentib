# tentib — Roadmap

> **Self-contained SemVer plan.** An agent can pick up the next version from this repo
> alone. Volatile state (current version, test count, deps) is in [`state.md`](state.md).
> What already shipped (M0 → 1.0.0, all six v1.0 criteria met at 1.0.0, then 1.0.1–1.4.0)
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

**hoosh: adopted** in hoosh 2.11.0 (1.3.1 records it). hoosh serves a model file in its
own process through `tx_fwd_packed`, buffered and streamed. The integration needed no new
accessor; what it found became the consuming guide's §4 (one caller at a time, `guard()`
ends the process, the consumer owns the vocabulary, a load keeps its memory). See the
CHANGELOG.

**murti: open, waiting for its port.** murti is still a Rust tree, and tentib is consumed
from Cyrius only (no FFI). When murti has a Cyrius port, it declares `[deps.tentib]` the
way hoosh did, and tentib answers what that integration finds, as here.

## 1.4.x — what the first consumer asked for

**Shipped in 1.4.0.** Incremental decode
(`tx_dec_reset` / `tx_dec_step`, bit-identical to the full forward's row), a load that
reuses a loaded model's buffers when the dims match, and the byte vocabulary as flags
bit 0 (with `tx_pack_peek` to read it before a load). See the CHANGELOG,
[`../architecture/002-incremental-decode.md`](../architecture/002-incremental-decode.md)
and [`../adr/0003-a-load-reuses-only-a-loaded-models-buffers.md`](../adr/0003-a-load-reuses-only-a-loaded-models-buffers.md).
**hoosh adopted them in 2.11.1**: decode with the cache (one cache shared by its
interleaved replies), `tx_pack_peek` instead of reading offset 24, the declared vocabulary
logged, and same-shape reloads that keep no memory.

What remains, additive, so a 1.x minor; proposed, not scheduled:

- **A token table in the artifact.** The byte vocabulary covers byte-level models only. A
  model trained on akshara's BPE needs its tokenizer beside the file today. A new flags
  bit would mark a section that carries one: either akshara's own serialized tables,
  opaque to tentib and checked only for length, or a plain id → bytes table (decode
  only; encode needs the merges). Which one is a question for the first consumer that
  serves a BPE model. **Gate:** the format doc specifies it, the decoder bounds and checks
  it like every other field, and P10 covers it.

(No batched prefill is planned. A prompt fed through `tx_dec_step` already costs one full
forward over it: 2.35 ms for 32 steps against 2.33 ms at the bench config.)

## 1.5.x — serve from packed ternary storage

**Goal.** Make the memory story true at serve time. Serving now reads an i8 working
copy (1 byte per weight, 8× smaller than f64); the 2-bit form (`tpack2`, 32×) is storage
only.

**Scope.** Additive: a kernel that serves from 2-bit codes, or from 5 trits per byte
(≈1.6 bits, a `tpack3` / `tunpack3` codec), unpacking in-register or through lookup
tables in the bitnet.cpp TL1 / TL2 / I2_S shape (ported, never copied); a packed-store
variant of `tx_pack` / `tx_fwd_packed` (and `tx_dec_step`) that uses it. With it, a
**serving-only allocation for loaded models**: a fresh `tx_pack_deserialize` runs the full
`tx_init`, so a load also allocates the latent weights, their gradients and the training
scratch (12,192,728 bytes for the 357,504-byte bench artifact; a same-dims reload has
allocated nothing since 1.4.0). The artifact format can stay at version 1; only what a
load allocates changes.

**Gate.** Bit-identical to the scalar kernel (suite + fuzz property P1 extended to the
new kernel); benchmarked against the i8 path at the four layer sizes and whole-model,
reporting the memory/speed trade honestly if it is slower; resident weight memory
measured.

## 1.6.x — the edge target (Raspberry Pi 4, for seema)

**Goal.** The Pi-class tok/s result the integer-native plan is aimed at, measured on the board
at hand: a **Raspberry Pi 4 Model B** (Cortex-A72, 4 × 1.5 GHz, 2 GB, `ssh pi`). It was
retargeted from a Pi 5 after 1.4.0. The Pi 5 came from the agnosticos plan's target figure,
not from anything tentib needs.

**Why a Pi 4 is enough.**
- **The kernel needs only base NEON.** cyrius lowers `iv_dp8` on aarch64 (since 6.4.30) to
  base ARMv8.0 NEON: `uxtl` / `sxtl` + `smlal`, with no `SDOT`. The A72 has no dot-product
  extension (its CPU features are `fp asimd`, no `asimddp`), and it needs none.
- **It already runs.** On 2026-11-09 the 1.4.0 suite passed 280/280 on the board. That was
  tentib's first run on real ARM hardware; CI runs aarch64 under qemu.

**Scope.**
1. **Measure: done (unreleased, after 1.4.0).**
   - `scripts/bench-history.sh` runs on the board with `BENCH_HOST=pi`.
   - `docs/benchmarks.md` §6 has the results: packed serving 1,241 tok/s, against 434 for
     f64 on the board (2.85×); a cached decode step 1,249 tok/s, against 80 with a full
     forward per token.
   - The fuzz harness also held there, and an x86_64 model file served with its x86_64
     digest.
   - This meets the gate below for the bench config.
2. **Size a b1.58-shaped model the Pi 4 can serve.** With 2 GB of RAM, a model near the
   plan's sizes needs 1.5.x's packed storage: 2B weights at 1 byte each is the whole memory,
   and at 2 bits it is 0.5 GB. Ship it with 1.2.0's format.

**A Pi 5** (Cortex-A76, which has `SDOT`) is a second data point whenever one is available.
It is also the only reason to add an `SDOT` path, which would keep the current one as the
A72's fallback. **Power** is still unmeasured, since there is no meter on the board.

**Check the plan's figure before citing it.** The figure is "a 2B model at ~5–7 tok/s on a
Pi 5 under ~15 W" (agnosticos `integer-native-ml.md`), and the plan gives no source for it.

**Gate.** `docs/benchmarks.md` gains an aarch64 section: the board (model, CPU, clock, RAM),
the pin, the model size, tok/s, and the f64 path on the same board for comparison.

## 1.7.x — GPU ternary kernels via mabda

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
  previous buffers (the bump allocator never frees); a server that reloads models keeps
  every one of another shape it has loaded (hoosh 2.11.0: 802,376 bytes per load at T = 30,
  C = 16; a same-shaped reload has kept nothing since 1.4.0), holds one decode cache, and
  must serialize every call. Outputs land in globals (`tx_logits`, `ki_logits`). An
  arena-owned, freeable handle fixes all four, allows more than one model per process, and
  changes every `tx_*` signature.
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
  lookup shape for 1.5.x.
- **STE**: Bengio, Léonard, Courville (2013), arXiv:1308.3432, the surrogate the FD gates
  validate.
- **Scalable MatMul-free LM**: Zhu et al. (2024), arXiv:2406.02528 (the research watch).
- In-repo history: [`proposals/2026-06-23-cyrius-integer-simd.md`](proposals/2026-06-23-cyrius-integer-simd.md)
  (the 0.4.1 design, shipped).
