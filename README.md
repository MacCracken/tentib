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

## Status — v0.4.1: the integer-SIMD kernel — multiply-free is also *faster* (90/90 gated)

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
```

Deps: rosnet 0.2.0 + tyche 0.1.1 + akshara 0.1.0 (the shared sovereign tokenizer).

See [`docs/development/roadmap.md`](docs/development/roadmap.md) for the M0→v1.0
plan (M3 the packed int8 matmul-free inference kernel) and
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
