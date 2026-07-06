# tentib benchmarks

The honest numbers behind the claims (the 0.5.0 v1.0 criterion). Everything here
is reproduced by the committed harness — no hand-curated numbers.

## Methodology (B-series fairness)

- **Harness:** [`tests/tentib.bcyr`](../tests/tentib.bcyr) —
  `cyrius build tests/tentib.bcyr build/bench && taskset -c 4 ./build/bench`.
- **Fairness:** identical shapes for every path; warm cache (warmup runs dropped);
  **single thread, one pinned CPU** (`taskset`); every whole-model path pays its
  per-call prep (see the honesty notes below); no path is hand-tuned differently.
- **Run of record:** 2026-07-06 · **cyrius 6.4.10** (pin in `cyrius.cyml`) ·
  **AMD Ryzen 7 5800H** (Zen 3, 8c/16t, x86-64 Linux) · core 4 (an idle physical
  core — a busy SMT sibling inflates everything ~1.6×). **Ratios are stable
  run-to-run; absolute ns vary with boost state** — reproduce ratios, not nanoseconds.
- **Quality run:** attn11 **1.13.0** (`build/attn11`, same host + core), command below.

## 1. Per-layer kernel sweep (M=1 row, K×N; ns/call)

The same quantized BitLinear, four compute paths. The SIMD row is **pack-once**
(the deployment shape — weights packed to transposed i8 ahead of time):

| K×N | branchy scalar | branchless (M3b) | **SIMD `iv_dp8` (0.4.1)** | rosnet SIMD-f64 `linear_fwd` | SIMD vs f64 | SIMD vs branchless |
|-----|---------------:|-----------------:|--------------------------:|-----------------------------:|------------:|-------------------:|
| 128×128 | 117,922 | 92,268 | **3,124** | 15,786 | **5.0×** | 29.5× |
| 256×256 | 567,076 | 359,524 | **6,128** | 60,974 | **9.9×** | 58.6× |
| 512×512 | 2,474,408 | 1,565,194 | **15,170** | 246,689 | **16.2×** | 103.1× |
| 768×768 | 5,765,001 | 3,615,121 | **29,361** | 541,591 | **18.4×** | 123.1× |

**The advantage grows with size.** Two compounding effects: (1) lane width — the
f64 path is 2-wide SSE2 packed-double, `iv_dp8` is 16 int8 lanes per instruction;
(2) cache — at 768×768 the f64 weights are 4.7 MB (spilling L2) while the i8
weights are 590 KB (L2-resident). Low-bit inference's defining property — more
lanes AND more weights per cache line — is exactly what the scalar kernels
couldn't express (0.4.0 shipped them ~5× *slower* than the f64 matmul at 128×128).

Bit-identity is not traded for the speed: the SIMD kernel's output is
**bit-identical** to the scalar kernel (suite gates M3d, 90/90).

## 2. Prep amortization (768×768)

| step | cost | paid |
|------|-----:|------|
| `tsimd_pack_w` (quantized i64 → transposed i8 + `wsum`) | 2,377,450 ns (~4 ns/weight) | **once per weight matrix** |
| `act_quant_int` (one f64 row → int8 codes) | 8,277 ns | once per token row |

Pack-once amortizes after ~81 matmul calls at this shape (2.38 ms ÷ 29.4 µs);
per-call re-packing would dominate the matmul ~80:1. That is why the whole-model
number below (which re-packs per call, by fairness design) understates the
deployment ceiling the sweep above measures.

## 3. Whole-model forward → tok/s

A b1.58-shaped transformer (F = 4C), random-init, **266,880 params**
(V=256 · T=32 · C=128 · F=512), T=32 tokens per forward, 30 measured forwards:

| path | ns/forward | tok/s | vs f64 |
|------|-----------:|------:|-------:|
| f64 weight-only (`tx_fwd` — rosnet matmul) | 13,866,101 | 2,307 | 1.0× |
| integer scalar (`tx_fwd_q` mode 0) | 70,169,395 | 456 | 0.20× |
| **integer SIMD (`tx_fwd_q` mode 2, 0.4.1)** | **9,015,473** | **3,549** | **1.5×** |

**Honesty notes.** All three paths re-quantize the weights every forward (the f64
path's `bl_forward_wonly` does too — matched per-call cost by design); mode 2
*additionally* re-packs to i8 every forward, so most of its 9.0 ms is prep, not
matmul. A pack-once whole-model inference entry point is the **0.7.0 deliverable**
(load → quantize + pack once → integer inference); the layer sweep above bounds
what it can reach. The non-linear f64 interludes (RMSNorm, attention core, GELU)
are shared by all paths and stay f64 per b1.58.

## 4. The memory story (measured at the whole-model config)

- Ternary linear weights: **1,835,008 B as f64 → 57,344 B 2-bit packed = 32×**
  (`tpack2`; the SIMD kernel's working format is 1-byte i8 = 8× — the 2-bit form
  is the storage format).
- The inner loop performs **zero multiplies** (`pmaddubsw` against {−1,0,+1} only
  sign-selects); the only f64 multiplies are the M·N dequant scalings vs M·K·N in
  a real matmul.

## 5. Quality delta (the honest small-scale gap)

Same corpus (`hello world`, byte-tokenized, V=8, 10 next-token positions), same
step budget (2000), matched size — the f64 sibling vs the ternary reference:

| model | params | final CE | task solved? |
|-------|-------:|---------:|--------------|
| attn11 1.13.0, f64 (`--d-model 4 --ctx 10 --heads 1 --layers 1 --steps 2000 --eval`) | 324 | **0.006** | yes (memorizes to ~zero) |
| tentib ternary transformer (C=4, F=8, T=10, 2000 steps — the bite-F demo) | 252 | **0.11** | yes (integer inference argmax 10/10) |

Both models solve the task; the ternary model **plateaus ~20× higher in CE** at
toy scale. Caveats, stated rather than hidden: the two references differ beyond
quantization (attn11: 4C MLP, learned positions, Adam + LR schedule; tentib:
F=2C MLP, RMSNorm, plain SGD 0.1) — this bounds the *bundled* small-model gap,
it is not a single-variable ablation. That is consistent with the named
references, **cited, not re-demonstrated**:

- **b1.58 paper** (Ma et al., *The Era of 1-bit LLMs*, arXiv:2402.17764): ternary
  matches fp16 quality **from ~3B params up** — small models pay a quality gap.
- **bitnet.cpp** (Microsoft): ~2–6× CPU speedup over fp16 via integer-SIMD
  kernels — the same lane-packing effect measured in §1 (our 5–18× is vs an
  SSE2-2-wide f64 baseline, not fp16 AVX2, so the numbers are not directly
  comparable; the *mechanism* is).

## Reproduction

```sh
cyrius build tests/tentib.bcyr build/bench
taskset -c <idle-core> ./build/bench          # §1–§4
cyrius build src/main.cyr build/tentib && ./build/tentib   # the bite-F ternary CE (§5)
# the f64 sibling (attn11 ≥1.13.0):
printf 'hello world' > /tmp/hello.txt
attn11 --corpus /tmp/hello.txt --d-model 4 --ctx 10 --heads 1 --layers 1 --steps 2000 --eval
```

Seeds are fixed in the harness/demo; CE values are deterministic, timings vary
with boost state (ratios hold).
