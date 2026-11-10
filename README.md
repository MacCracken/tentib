# tentib

**Sovereign integer-native / ternary (1.58-bit) machine-learning reference, in [Cyrius](https://github.com/MacCracken/cyrius).**

`tentib` is `bitnet` spelled backwards. It is the AGNOS reference that takes
[attn11](https://github.com/MacCracken/attn11)'s thesis — *gradient-based learning
is expressible in everything-is-i64 Cyrius* — and applies it to **the weights
themselves**: weights quantized to **ternary {−1, 0, +1}** (log₂3 ≈ **1.58
bits/weight**), so the weight·activation path has **no multiply left** — only
**add / subtract / skip** over integers (the BitNet b1.58 idea, sovereign and
from the metal up).

It is a **sibling reference** to attn11 (the Transformer) and
[tarka](https://github.com/MacCracken/tarka) (RL/reasoning), standing on the same
f64 substrate ([rosnet](https://github.com/MacCracken/rosnet) tensors + grad,
[tyche](https://github.com/MacCracken/tyche) PRNG,
[akshara](https://github.com/MacCracken/akshara) tokenizer). No BLAS, no libc, no
autodiff; every hand-derived gradient is finite-difference-gated.

> Forward-design map: [`agnosticos/docs/development/planning/integer-native-ml.md`](https://github.com/MacCracken/agnosticos/blob/main/docs/development/planning/integer-native-ml.md).

## Status — 1.x STABLE (1.5.0: 346 assertions + a property fuzzer, x86_64 and aarch64, on a Raspberry Pi 4 too; API frozen for the 1.x series)

- **M0 (v0.1.0)** — ternary quantizer + matmul-free dot ([`src/ternary.cyr`](src/ternary.cyr)).
- **M1 (v0.2.0)** — **BitLinear**: ternary weights + int8 activations over rosnet's
  matmul ([`src/bitlinear.cyr`](src/bitlinear.cyr)), trainable via the
  **straight-through estimator**, every gradient FD-gated — *prove the surrogate,
  not the discontinuity* (incl. the γ-cancellation + a falsifier).
- **M2 (v0.3.0)** — **a ternary transformer trains from scratch.** BitLinear assembled
  into a pre-norm block — RMSNorm + causal attention + GELU-MLP, all six linear
  projections ternary ([`src/layers.cyr`](src/layers.cyr), [`src/block.cyr`](src/block.cyr))
  — trained end-to-end via the STE on real **akshara**-tokenized text. *Every*
  component and the full assembly is finite-difference-gated; the model trains
  (CE → ~0.1). The ternary sibling of attn11's first loss curve.
- **M3 (v0.4.0)** — **the matmul-free integer inference kernel, whole-model.** The
  kernel runs through every BitLinear of the trained transformer
  ([`src/kernel.cyr`](src/kernel.cyr)): ternary weights × int8 activations →
  integer signed-accumulate (add / subtract / skip, *no multiply*) → **exact parity
  < 1e-9** vs the f64 forward, next-token **argmax 10/10**, plus a branchless scalar
  kernel (~25% over branchy) and 2-bit packed weights (**32×** smaller). The SIMD
  throughput lever was deferred to **0.4.1**, gated on cyrius integer SIMD.
- **M3-SIMD (v0.4.1)** — **the integer-SIMD kernel: multiply-free is also *faster*.**
  The gate lifted (cyrius 6.4.6/6.4.7 shipped integer SIMD incl. `iv_dp8`, the
  widening u8·i8 → i32 dot, 16 lanes/instr): `ternary_matmul_free_simd` packs the
  ternary weights to transposed i8 rows once, offsets the int8 activation codes to
  u8 (exact `+128·Σw` correction), and runs one `iv_dp8` per 16 weights —
  **bit-identical** to the scalar kernel, and on a 128×128 layer **~7.5× faster
  than rosnet's f64-SIMD matmul** (1.9 µs vs 14.7 µs; ~45× over the branchless
  scalar). The 0.4.0 standings — scalar ternary ~5× *slower* than f64 — are inverted.
- **Benchmarks (v0.5.0)** — [`docs/benchmarks.md`](docs/benchmarks.md), driven by the
  real `tests/tentib.bcyr` harness under B-series fairness: the SIMD advantage
  **grows with size** (5.0× over f64 at 128×128 → **18.4× at 768×768** — lane width
  + i8 cache residency), whole-model **3,549 tok/s vs 2,307 f64** (per-call re-pack
  included; pack-once deployment is the 0.7.0 entry point), the 32× memory story
  measured, and the **honest quality delta** vs a matched-size f64 attn11 (CE 0.006
  vs 0.11 at toy scale; b1.58 ≥3B parity cited, not re-demonstrated).
- **API freeze (v0.6.0)** — [`docs/api.md`](docs/api.md): every public symbol
  documented + directly gated (suite 90 → 95); the allocation audit recorded —
  allocation only in `*_init` + the one-shot trainers, **every inference path
  allocation-free**; internals (FD probes, STE masks, sublayer chains) explicitly
  out of the freeze.
- **Consumer readiness (v0.7.0)** — the **pack-once deployment serving path**
  (additive): `tx_pack_init`/`tx_pack` quantize + pack all 7 BitLinears once,
  `tx_fwd_packed` serves — bit-identical to the reference kernels and **13,675
  tok/s vs 2,378 f64 (5.7×)** at the bench config. Worked, self-checking consumer
  example: [`examples/quickstart.cyr`](examples/quickstart.cyr) (train → pack →
  serve, public API only).
- **Hardening audit (v0.8.0)** — [`docs/audit/2026-07-06-audit.md`](docs/audit/2026-07-06-audit.md):
  re-derived from source; **6 findings fixed/guarded** (the γ=0 quantizer sign
  bug; fail-loud `guard()` on non-ternary pack input, the SIMD K-bound, and
  packed-store misuse) + **5 paths verified sound** (incl. `_tanh` — immune to
  the ganita overflow class by construction). Hot loops carry zero checks;
  packed serving throughput unchanged. CHANGELOG verified complete 0.1.0→now.
- **v1.0.0 STABLE** — the clean cut: all six v1.0 criteria green, no code change
  from 0.8.0; the [`docs/api.md`](docs/api.md) freeze is in force for the 1.x
  series (signatures/semantics stable, minors add only).
- **P(-1) sweep (v1.1.2)** — [`docs/audit/2026-11-09-audit.md`](docs/audit/2026-11-09-audit.md):
  out-of-range token / target ids and mismatched dims now fail loud (an id ≥ V
  used to write past a gradient buffer in `tx_bwd`); the activation STE mask no
  longer zeroes a row's max-element gradient; NaN propagates through the integer
  path; the quantizer, packer and activation quant are faster with the same bits
  (golden-dump verified); every guard's fail path is tested, and a real property
  fuzzer runs on both targets in CI.
- **Packed-model serialization (v1.2.0)** — `tx_pack_serialize` writes a trained,
  packed model as bytes (2-bit ternary weights + the f64 embeddings, gains and bias);
  `tx_pack_deserialize` loads them in another process and `tx_fwd_packed` serves the
  same logits bit for bit. The decoder treats the bytes as hostile: every field is
  checked before anything is allocated, bad data is a status code, the caller caps the
  memory. Format: [`docs/architecture/001-packed-model-format.md`](docs/architecture/001-packed-model-format.md).
- **A library for consumers (v1.3.0)** — `dist/tentib.cyr` is a distlib bundle: one
  `[deps.tentib]` entry brings it with rosnet + tyche (no akshara, no hooks to define).
  [`examples/serve.cyr`](examples/serve.cyr) is the consumer shape, built from the bundles
  alone: one process saves a model file, another loads and serves it bit for bit. How-to:
  [`docs/guides/consuming-tentib.md`](docs/guides/consuming-tentib.md). **First consumer
  (v1.3.1):** hoosh 2.11.0 serves tentib models in its own gateway process; the guide's §4
  is what a long-running server has to handle.
- **What the first consumer asked for (v1.4.0)** — **incremental decode**: `tx_dec_step`
  runs one new position against a key/value cache, bit-identical to the full forward's
  row, so a generated token no longer costs a forward over the whole window. A **reload of
  a same-shaped model allocates nothing** (it reuses the loaded model's buffers). The model
  file can declare the **byte vocabulary**, and `tx_pack_peek` reads a file's header before
  a load. Constraints: [`docs/architecture/002-incremental-decode.md`](docs/architecture/002-incremental-decode.md).
- **A load for serving only (v1.5.0)** — `tx_pack_deserialize_serving` serves exactly what
  `tx_pack_deserialize` does, from 8.5× less memory (1.4 MB instead of 12.2 MB at the bench
  config): no latent weights, gradients or training scratch. Measured on a **Raspberry Pi 4**
  (Cortex-A72): packed serving 1,241 tok/s, 2.85× the board's f64 path, and a cached decode
  step 1,249 tok/s ([`docs/benchmarks.md`](docs/benchmarks.md) §6).

```
M0  ternary w = [ -1 -1 -1 -1 0 1 1 1 ]   gamma = 0.39
    matmul-free dot 1.09  vs  full-precision 1.20   (|err| 0.10)

M1  BitLinear (ternary weights, int8 activations, matmul-free):
    ternary W = [ 1 -1 0 1 ]  gamma = 0.80   BitLinear y = [ 0.80 -1.60 ]

M2  ternary transformer trains on akshara-tokenized text "hello world":
    V = 8 tokens, T = 10 positions   initial CE = 2.02   final CE = 0.11

M3  matmul-free integer inference through the trained transformer:
    exact parity vs f64 < 1e-12   next-token argmax 10/10   (weights 32x smaller)

M3-SIMD  the 0.4.1 kernel (iv_dp8, 16 int8 lanes/instr), 128x128 layer:
    bit-identical to scalar   1946 ns/call  vs  rosnet SIMD-f64 14670 ns  (~7.5x)

0.7.0  pack once -> serve (267k-param model, T=32; 0.8.0-guarded run):
    PACKED serving 13675 tok/s  vs  f64 2378 tok/s  (5.7x, bit-identical)
```

Deps: rosnet 1.4.1 + tyche 1.2.2 + akshara 1.3.1 (the shared sovereign tokenizer, test scope);
the current pins live in `cyrius.cyml` and [`docs/development/state.md`](docs/development/state.md).

See [`docs/development/roadmap.md`](docs/development/roadmap.md) for the open work
(1.2.x – 1.x.x, and the 2.0.0 backlog), [`CHANGELOG.md`](CHANGELOG.md) for what shipped, and
[`docs/adr/0001-tentib-scope.md`](docs/adr/0001-tentib-scope.md) for the scope +
naming rationale.

## Build

```sh
cyrius deps                               # resolve stdlib (no external deps at M0)
cyrius build src/main.cyr build/tentib    # compile
./build/tentib                            # run the demo
cyrius test                               # ternary-quantization correctness suite
```

## License

GPL-3.0-only
