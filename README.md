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

## Status — v0.2.0 shipped; M2 in progress (57/57 gated)

- **M0 (v0.1.0)** — ternary quantizer + matmul-free dot ([`src/ternary.cyr`](src/ternary.cyr)).
- **M1 (v0.2.0)** — **BitLinear**: ternary weights + int8 activations over rosnet's
  matmul ([`src/bitlinear.cyr`](src/bitlinear.cyr)), trainable via the
  **straight-through estimator**, every gradient FD-gated — *prove the surrogate,
  not the discontinuity* (incl. the γ-cancellation + a falsifier).
- **M2 in progress** — a minimal **ternary LM trains from scratch**
  ([`src/model.cyr`](src/model.cyr)): embedding → BitLinear head → softmax-CE, SGD
  on the latent weight. CE descends `ln 8 ≈ 2.08 → ~0.57` on a synthetic successor
  bigram (7/8 memorized) — the ternary sibling of attn11's first loss curve. The
  end-to-end gradient is FD-gated (the STE composes with the real loss). Next:
  RMSNorm → MLP → attention → full block → akshara corpus (v0.3.0).

```
M0  ternary w = [ -1 -1 -1 -1 0 1 1 1 ]   gamma = 0.39
    matmul-free dot 1.09  vs  full-precision 1.20   (|err| 0.10)

M1  BitLinear (ternary weights, int8 activations, matmul-free):
    ternary W = [ 1 -1 0 1 ]  gamma = 0.80   BitLinear y = [ 0.80 -1.60 ]

M2  ternary LM trains from scratch (embedding -> BitLinear head -> softmax-CE):
    initial CE = 2.07   final CE = 0.57   memorized = 7 / 8 successor tokens
```

Deps: rosnet 0.2.0 + tyche 0.1.1 (akshara lands later in M2).

See [`docs/development/roadmap.md`](docs/development/roadmap.md) for the M0→v1.0
plan (M2 a ternary transformer trained from scratch over an akshara corpus, M3 the
packed int8 matmul-free inference kernel) and
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
