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

> **1.1.1 (2026-11-09): the f64 baseline moved; tentib's kernels did not.** rosnet 1.2.2 made
> `linear_fwd` run one packed `f64v_axpy` per row step, and 1.4.1 runs that step 4 lanes wide on an
> AVX2 CPU. The results are the same bits, and tentib's suite and demo output are unchanged. This is
> the same harness and host, `taskset -c 4`, at cyrius 6.7.18 / rosnet 1.4.1:
>
> | | run of record | 1.1.1 |
> |---|---:|---:|
> | rosnet f64 `linear_fwd`, 128×128 / 768×768 | 15,786 / 541,591 ns | 6,581 / 189,228 ns |
> | SIMD `iv_dp8`, 128×128 / 768×768 | 3,124 / 29,361 ns | 1,733 / 25,876 ns |
> | per-layer SIMD vs f64, 128² → 768² | 5.0× → 18.4× | **3.7× → 7.3×** |
> | whole-model f64 | 2,378 tok/s | 3,166 tok/s |
> | integer SIMD, mode 2, vs f64 | 1.3× | **0.9×** |
> | packed serving | 13,675 tok/s (5.7×) | 13,382 tok/s (**4.2×**) |
>
> The tables below are the 2026-07-06 run of record, left as measured. Wherever they compare against
> f64, they measured the older rosnet, and §1's lane-width reasoning ("2-wide SSE2") no longer
> describes rosnet on an AVX2 CPU. The tables, the README and `docs/api.md` have not been
> re-baselined yet.
>
> **1.1.2 (2026-11-09): tentib's own paths got faster, with the same bits.** The P(-1)
> sweep sped up the quantizer, the packer and the activation quant (the kernels were not
> touched; the per-layer rows did not move). Same host, `taskset -c 4`, median of three
> runs (`scripts/bench-history.sh`, `bench-history.csv`): f64 forward 11.16 → 7.44 ms;
> integer SIMD mode 2 10.84 → 6.30 ms; packed serving 2.40 → 2.25 ms (13.3k → 14.2k tok/s);
> one-time `tx_pack` 8.28 → 3.93 ms; a training step 27.85 → 22.43 ms. Because the f64
> reference path got faster too, packed serving is 3.3× the f64 path at this config.
>
> **1.4.0 (2026-11-09): decoding one token at a time, and reloading.** New rows, all at the
> whole-model config (V = 256, T = 32, C = 128, F = 512), same host, `taskset -c 4`, median of
> three runs (`bench-history.csv`):
>
> | | per token / per load |
> |---|---:|
> | decode a 32-token window, a full `tx_fwd_packed` per token (the 1.3.x way) | 1.14 ms |
> | decode the same window, `tx_dec_step` per token | **73.5 µs (15.5×)** |
> | `tx_pack_deserialize`, fresh (allocates 12,192,728 bytes) | 6.73 ms |
> | `tx_pack_deserialize` over a loaded model of the same dims (allocates 0) | **4.22 ms** |
>
> 32 steps take 2.35 ms, the same as one 32-position forward (2.33 ms): stepping a prompt in
> costs what one forward over it does. The same-dims load still runs every validation pass,
> the byte-at-a-time FNV checksum among them. An interleaved A/B against the 1.3.1 sources
> (five runs each, alternating, median) puts every whole-model row within ±1.3% (packed
> serving −1.2%), so the attention row's refactor costs nothing. Against 1.2.0's recorded
> run every row, the unchanged per-layer kernels too, is 4–12% slower: that is this host's
> state on the day, which is why only same-session rows compare.
>
> **1.5.0 (2026-11-09): loading for serving only.** `tx_pack_deserialize_serving` allocates what
> serving reads (1,434,536 bytes for the bench artifact) instead of the whole model (12,192,728).
> Median of three runs, both machines the same day:
>
> | | x86_64 (5800H, core 4) | Raspberry Pi 4 (core 3) |
> |---|---:|---:|
> | `tx_pack_deserialize`, fresh | 6.25 ms | 81.58 ms |
> | `tx_pack_deserialize_serving`, fresh | **4.24 ms** | **36.92 ms** |
> | the same over a serving-only model (allocates 0) | 3.94 ms | 30.69 ms |
> | packed serving from the serving-only model | 2.22 ms | 25.78 ms |
> | packed serving from a fully loaded model | 2.20 ms | 25.83 ms |
>
> Serving is the same kernel, and its speed does not move. A fresh serving load is a third
> faster on x86_64 and more than twice as fast on the Pi, where allocating and first touching
> 10.8 MB costs more. Every row whose code did not change sits within −1.5% to +1.1% of 1.4.0's
> run on the Pi, which was idle both times. On x86_64 the same rows read a median 5.9% faster,
> because the machine was quieter.

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
| `tsimd_pack_w` (quantized i64 → transposed i8 + `wsum`) | 4,632,914 ns (~8 ns/weight, incl. the 0.8.0 ternary-validity guard) | **once per weight matrix** |
| `act_quant_int` (one f64 row → int8 codes) | 7,561 ns | once per token row |

Pack-once amortizes after ~160 matmul calls at this shape (4.63 ms ÷ 29.4 µs);
per-call re-packing would dominate the matmul ~160:1. That is why the mode-2
whole-model number below (which re-packs per call, by fairness design) understates
the deployment path — and why the 0.7.0 **packed serving** row, which hoists the
pack, is ~4.2× faster than mode 2 on the same model.

## 3. Whole-model forward → tok/s

A b1.58-shaped transformer (F = 4C), random-init, **266,880 params**
(V=256 · T=32 · C=128 · F=512), T=32 tokens per forward, 30 measured forwards
(run of record 2026-07-06 with the 0.8.0 guards live, same host/core):

| path | ns/forward | tok/s | vs f64 |
|------|-----------:|------:|-------:|
| f64 weight-only (`tx_fwd` — rosnet matmul) | 13,451,628 | 2,378 | 1.0× |
| integer scalar (`tx_fwd_q` mode 0) | 69,640,126 | 459 | 0.19× |
| integer SIMD (`tx_fwd_q` mode 2, 0.4.1) | 9,920,976 | 3,225 | 1.3× |
| **PACKED serving (`tx_fwd_packed`, 0.7.0)** | **2,339,894** | **13,675** | **5.7×** |

**Honesty notes.** The first three paths re-quantize the weights every forward
(the f64 path's `bl_forward_wonly` does too — matched per-call cost by design);
mode 2 *additionally* re-packs to i8 every forward, so most of its 8.9 ms is
prep, not matmul — that is the reference-fairness surface. **The packed row is
the deployment surface** (0.7.0: `tx_pack_init`/`tx_pack` once — 7.3 ms at this
config incl. the 0.8.0 pack-time guards, amortized after ~3 forwards — then
`tx_fwd_packed` serves with only the per-token activation quant per call),
bit-identical to modes 0/2 (suite-gated).
Its remaining 2.4 ms/forward is dominated by the shared non-linear f64 interludes
(RMSNorm, attention core, GELU — kept f64 per b1.58) plus the T×V head; the
linear layers themselves run at the §1 sweep rates.

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

## 6. On a Raspberry Pi 4 (aarch64, the 1.6.x edge target)

The same harness, built `--aarch64` and run on the board (`BENCH_HOST=pi BENCH_LABEL=rpi4
./scripts/bench-history.sh`), median of 3. The rows are `rpi4: …` in `bench-history.csv`.

- **The board:** Raspberry Pi 4 Model B Rev 1.2. BCM2711, 4 × Cortex-A72 at 1.5 GHz,
  2 GB, Linux 7.0 (raspi). The `ondemand` governor held 1,500 MHz before and after the
  runs, at 51–56 °C (the firmware's throttle flags are not readable without `/dev/vcio`).
  Pinned to core 3, the other cores idle; the A72 has no SMT.
- **Build:** tentib 1.4.0 (`b29ae7c`), cyrius 6.7.18, 2026-11-09. The x86_64 column is
  the same day's run of record, on the Ryzen 7 5800H, core 4.
- **The kernel needs only base NEON.** The A72 has no dot-product extension
  (`fp asimd`, no `asimddp`). cyrius lowers `iv_dp8` to base ARMv8.0 NEON (`uxtl` /
  `sxtl` + `smlal`), so the SIMD rows run the same kernel CI checks under qemu.
- **Correctness on the board, the same day:**
  - the suite passed 280/280;
  - the fuzz harness held P1–P12 over 5,000 rounds (33 s);
  - a model file written on x86_64 loaded and served with the x86_64 digest.

**Whole model** (V = 256, T = 32, C = 128, F = 512; 266,880 params; T tokens per forward):

| path | Pi 4 | tok/s | x86_64 | tok/s |
|------|-----:|------:|-------:|------:|
| f64 weight-only (`tx_fwd`) | 73.58 ms | 434 | 7.50 ms | 4,265 |
| integer scalar (`tx_fwd_q` mode 0) | 472.5 ms | 67 | 67.45 ms | 474 |
| integer SIMD (`tx_fwd_q` mode 2) | 69.17 ms | 462 | 6.58 ms | 4,866 |
| **packed serving** (`tx_fwd_packed`) | **25.78 ms** | **1,241** | 2.33 ms | 13,726 |
| a training step (f64) | 229.6 ms | | 23.57 ms | |

**Packed serving is 2.85× the f64 path on the Pi** (3.2× on x86_64), and the whole board
runs about 10–11× slower than the 5800H core on every whole-model row.

**Decoding a 32-token window**, one token at a time (1.4.0):

| | Pi 4 | per second | x86_64 |
|---|---:|---:|---:|
| a full `tx_fwd_packed` per token | 12.57 ms | 80 tok/s | 1.14 ms |
| `tx_dec_step`, cached | **800.7 µs** | **1,249 tok/s** | 73.5 µs |

The cached step is 15.7× faster per token. Stepping the 32 tokens in takes 25.6 ms, the
same as one 32-position forward (25.8 ms), as on x86_64.

**Per layer** (M = 1, ns/call):

| K×N | branchy | branchless | SIMD (pack-once) | rosnet f64 | SIMD vs f64 (Pi / x86_64) |
|-----|--------:|-----------:|-----------------:|-----------:|--------------------------:|
| 128×128 | 886,124 | 825,030 | 19,437 | 84,618 | 4.4× / 4.1× |
| 256×256 | 3,963,468 | 3,693,084 | 55,218 | 257,504 | 4.7× / 5.8× |
| 512×512 | 18,277,112 | 17,494,516 | 175,732 | 1,060,017 | 6.0× / 7.0× |
| 768×768 | 88,444,097 | 86,160,542 | 357,850 | 2,352,412 | 6.6× / 7.5× |

**Load and serialization:**
- a fresh `tx_pack_deserialize` takes 81.6 ms, and a fresh `tx_pack_deserialize_serving`
  (1.5.0, 1.4 MB instead of 12.2 MB) 36.9 ms;
- a same-dims reload takes 31.2 ms;
- serialization takes 27.1 ms;
- the one-time `tx_pack` takes 43.3 ms.

**What the board shows that x86_64 did not:**
- **The branchless scalar kernel buys almost nothing on the A72.** It runs at 0.93–0.97 of
  branchy's time; on the 5800H it was 0.61–0.83. It is a reference kernel, not the serving
  path, so nothing changes. M3b's premise, that unpredictable branches dominate, holds
  on Zen 3 but much less on this core.
- **The scalar kernels fall off a cliff at 768×768:** 15–25× the x86_64 time, against
  about 10× elsewhere. Their weights are i64, 4.7 MB at 768², more than the A72's 1 MB L2,
  and the board has no L3 (the 5800H has 16 MB). The SIMD kernel's i8 copy is 0.6 MB and
  stays at 12.5×. That is consistent with the memory story in §4, though it has not been
  profiled.

**Not measured yet:**
- **Power.** No meter is on the board, so the integer-native plan's "under ~15 W" is
  untested.
- **A model sized for the board.** These are the bench's toy dims. A b1.58-shaped model
  near the plan's sizes needs 1.5.x's packed weight storage to fit 2 GB. Since 1.5.0 a
  serving load no longer allocates the training buffers (1.4 MB here instead of 12.2 MB).

## Reproduction

```sh
cyrius build tests/tentib.bcyr build/bench
taskset -c <idle-core> ./build/bench          # §1–§4
BENCH_HOST=pi BENCH_LABEL=rpi4 ./scripts/bench-history.sh   # §6, on an aarch64 board over ssh
cyrius build src/main.cyr build/tentib && ./build/tentib   # the bite-F ternary CE (§5)
# the f64 sibling (attn11 ≥1.13.0):
printf 'hello world' > /tmp/hello.txt
attn11 --corpus /tmp/hello.txt --d-model 4 --ctx 10 --heads 1 --layers 1 --steps 2000 --eval
```

Seeds are fixed in the harness/demo; CE values are deterministic, timings vary
with boost state (ratios hold).
