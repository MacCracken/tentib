# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.1.2] — 2026-11-09

The P(-1) hardening, optimization and repair sweep, recorded in
[`docs/audit/2026-11-09-audit.md`](docs/audit/2026-11-09-audit.md), the second tentib audit. No
public signature changed. On valid input every output is bit-identical to 1.1.1, except the
activation STE mask's (fixed below). A golden dump of 333,281 raw output values over every public
function differs from 1.1.1 in exactly that one place. Toolchain and deps are unchanged (cyrius
6.7.18; tyche 1.2.2, rosnet 1.4.1, akshara 1.1.2).

### Fixed

- **An out-of-range token id corrupted the heap** (MEDIUM). `tx_bwd`'s embedding scatter wrote C
  f64s past `tx_dtokemb` for a token ≥ V or < 0. `tx_fwd`, `tx_fwd_q` and `tx_fwd_packed` read past
  the embedding table for the same ids. Ids are checked once per call now, in one shared embed
  (`_tx_embed`), and again in `tx_bwd` before the scatter. A bad id fails loud.
- **`bl_ste_mask_x` zeroed the max element's gradient in about 14% of token rows** (MEDIUM). It
  masked where `|x/s| > 127`. At the row's max element, `a/(a/127)` rounds to `127.00000000000001`
  for 13.65% of absmax values (a = 1.1, 2.2, 0.01, …), but the forward never clamps that element.
  An element now counts as saturated only when the forward's clamp would change its code, which is
  `|x/s| ≥ 127.5`. With `bl_act_quant`'s scale the mask is the identity, as documented. tentib's own
  trainers use weight-only BitLinears, so the bug reached only a consumer training with activation
  quant.
- **Target ids are checked** in `softmax_xent_fwd` / `_bwd`. An out-of-range target used to read
  past the probs row, giving a wrong loss, or drop the one-hot term, giving a wrong gradient.
- **Dims are checked against what was allocated.** Each `*_init` records its dims. The `tx_*` calls,
  `block_fwd`/`_bwd`, `bl_forward_q`, `tx_fwd_q`, `tx_pack` and `tx_fwd_packed` refuse a V, C or F
  other than the init's, a T outside [1, init T], or a call before the init. Any of those used to
  read and write past the buffers. A shorter T is legal: serving a prefix returns exactly the
  prefix's rows of the full serve, bit for bit.
- **The integer path no longer turns NaN into a number.** `act_quant_int`'s absmax skipped NaN, so
  `f64_to(NaN)` became a valid-looking code (−127 on x86_64, 0 on aarch64) and the row gave a
  plausible logit, while the f64 reference returned NaN. The absmax now compares `f64_abs` bit
  patterns as integers. That gives the same bits on every NaN-free row, but a NaN sorts above
  +Inf, so its row comes out NaN in every kernel. `bl_act_quant` uses the same rule.
- `guard()` writes to **stderr** directly. A library consumer no longer has to define `puts` for
  tentib. The exit status is still 1.
- `absmean(w, 0)` returns 0, where it returned 0/0 = NaN.
- `src/test.cyr` defines akshara's four hooks, so the `[build].test` unit builds without
  "undefined function" warnings.

### Changed — faster, same bits

Measured on one binary per run, pinned to core 4, median of three runs (`bench-history.csv`):

- **`bl_weight_effective`**: `ternary_one` is inlined, and `Weff` comes from a 3-entry table of
  `{−γ, 0·γ, +γ}`. The f64 forward drops from 11.16 to 7.44 ms (−33%).
- **`tsimd_pack_w`** walks `Wq` row by row, with one ternary check per row: 4.82 → 3.28 ms (−32%).
  With the change above, the one-time `tx_pack` drops from 8.28 to 3.93 ms (−53%) and the
  integer-SIMD whole-model mode from 10.84 to 6.30 ms (−42%).
- **`act_quant_int`** (the integer absmax, plus per-row base pointers): 6.81 → 4.41 µs (−35%).
  **Packed serving** drops from 2.40 to 2.25 ms (−6%), 13.3k → 14.2k tok/s.
- **Training**: the six bias-free projections pass `db = 0` to `linear_bwd`, instead of
  accumulating into scratch that nothing read, which removes `blk_db` / `mlp_db`. A training step
  drops from 27.85 to 22.43 ms (−19%).
- The kernels themselves were not changed, and the per-layer rows did not move (±3%). The f64
  reference path got faster too, so the packed-vs-f64 ratio at the bench config falls from 4.2× to
  3.3× even though serving itself is faster.
- **Measured and not adopted:** a per-row code check inside `ternary_matmul_free_simd`. It cost
  4–10% of the kernel at M = 1, and hot loops carry no checks. The kernel's contract (codes in
  [−128, +127]) is now stated precisely instead. See
  [ADR 0002](docs/adr/0002-hardening-preserves-bits.md).

### Added

- **Suite 101 → 139 assertions.** The new group covers:
  - the mask fix, the NaN propagation and `absmean(·, 0)`;
  - `ternary_matmul_free_gpu`'s host fallback at four shapes, ragged against the 8×8×8 tile
    (the suite never ran it before, and `gpumm`'s one shape never reached those branches);
  - prefix serving;
  - **20 guard fail paths**, each run in a forked child with stderr silenced, plus 2 exit-0
    controls.
- **A real fuzz harness** (`tests/tentib.fcyr`; it was a no-op): 5,000 seeded rounds, ~3 s,
  checking eight properties:
  - the four kernels agree bit for bit;
  - the activation codes stay in range;
  - the 2-bit packing round-trips;
  - `Weff` = code·γ;
  - the integer output stays within the f64 summation error bound;
  - NaN rows propagate;
  - the activation mask is the identity;
  - on small whole models, modes 0/2 and packed serving agree, and served prefixes match.
- **Mutation testing**: the suite killed 9 of 9 injected defects, and the fuzz harness killed the 7
  that change outputs.
- **`scripts/bench-history.sh`** and `bench-history.csv`, with the before and after rows, plus a
  training-step benchmark row.
- `.gitattributes` (`lib/** -text`): `cyrius.lock` records `lib/`'s bytes, so a CRLF checkout
  would fail `cyrius deps --verify`.
- [ADR 0002](docs/adr/0002-hardening-preserves-bits.md): an optimization must preserve bits,
  proven by a golden dump, and guards go at call entry, never inside kernel loops.
- Every public function has a doc comment. `cyrius doc --check` reports 0 undocumented, down from
  67, and `cyrius audit` passes (it exited 1 before).

### Security / CI

- Workflows use a read-only token by default. Only the release job gets `contents: write`, so the
  CI gate it calls no longer inherits it.
- Checkouts set `persist-credentials: false`.
- Actions are pinned by SHA to checkout v7.0.1 and action-gh-release v3.0.3, verified against
  their tags.
- The release tag filter is `X.Y.Z*`.
- CI gates on fmt, on lint (any warning fails; all 53 line-length warnings are fixed), vet and
  doc. It also runs the demo, the suite, the example, `gpumm` (exit 96 on the host) plus an AGNOS
  build, the fuzz harness and a bench smoke run. Further jobs cover aarch64 (the suite and fuzz
  under qemu), a security scan and docs/version consistency.
- The release ships a DCE build, a source archive and `SHA256SUMS`.

### Documentation

- The SIMD kernel's comment and `docs/api.md` overstated the i32 exactness bound ("~64·K, safe to
  K ~ 2²⁴"). iv_dp8 sums its lanes in i32, so the limit is 255·K < 2³¹ (K < ~8.4M). The `K ≤ 2²²`
  guard was always right. Also corrected: a stale "no integer SIMD" comment, and the mask
  comment's "measure-zero".
- `SECURITY.md` is rewritten for the library surface a consumer drives with token ids, and for
  both audits. `docs/api.md` lists the 1.1.2 contracts.
- `CONTRIBUTING.md` no longer calls the bench and fuzz harnesses stubs. It also no longer warns
  about a missing call-arity check: every toolchain from 6.6.0 to 6.7.18 rejects an arity
  mismatch, and the 2026-06-23 issue is marked resolved upstream.
- `CLAUDE.md` gives the scope of `var buf[N]`: N bytes at function scope, N i64 slots at module
  scope. The ADR index now lists 0001, which was missing.

## [1.1.1] — 2026-11-09

This release moves tentib and its deps to their latest releases. Nothing in `src/` changed, the
frozen 1.x public API is untouched, and every value tentib computes is the same as in 1.1.0.
**tentib's f64 comparison baseline is faster, though,** so the headline speedup ratios are smaller
(see *Performance*).

### Changed

- **Toolchain `6.7.9` → `6.7.18`** (through 6.7.10–6.7.17). `cyrius lib sync --full` re-vendored
  the stdlib. `lib/` byte-matches the 6.7.18 snapshot: the same 113 files as 6.7.9, 66 of them with
  new content, and nothing was added or removed. `cyrius fmt --check` passes on every source. Lint
  gives the same 53 warnings it gave at 6.7.9, all of them line length; the lint rules that 6.7.11
  and 6.7.12 added (array overruns, write lengths) fire on nothing. 6.7.17 refuses an include line
  with code after the file name, and tentib's include lines carry only a trailing `#` comment, which
  is allowed. 6.7.17 also refuses a statement that is only a value builtin, and every target still
  compiles. **From 6.7.18, the pin, `lib/` and `cyrius.lock` must land in one commit.** With a
  fully vendored `lib/`, cyrius refuses a moved pin until `lib sync --full` has run, so CI's
  `cyrius deps` step fails on a commit that moves the pin alone.

- **Deps moved to their latest releases, all built against 6.7.18: `tyche` 1.2.0 → 1.2.2, `rosnet`
  1.2.0 → 1.4.1, `akshara` 1.1.0 → 1.1.2.** No public signature changed in any of them. Vendored
  `lib/tyche.cyr`, `lib/rosnet.cyr` and `lib/akshara.cyr` byte-match the tagged `dist/` bundles.
  These changes reach tentib:
  - **rosnet** 1.2.2–1.4.1 speeds up `linear_fwd` and `linear_bwd`, and 1.4.1 adds an AVX2 path
    for them. rosnet reports that the CPU results are bit-identical to 1.2.0, and tentib confirms it
    for its own outputs: the demo prints exactly what it printed at 6.7.9 (39 lines compared, timing
    lines excluded). `t_alloc(0)` now returns an empty tensor instead of exiting, and rosnet's errors
    go to stderr. The 1.4.1 bundle begins with `include "lib/simd.cyr"`, which resolves because
    tentib vendors the whole snapshot.
  - **tyche** 1.2.2 gives the same bits for every state its API can reach. Its one behavior change
    is that writing 0 to `_rng_state` no longer hangs `rng_normal`. tentib never writes `_rng_state`.
  - **akshara** 1.1.2 is a hardening sweep. It validates widths and adds error codes, sends fatal
    messages to stderr, and no longer needs the consumer to define `puts`. Its outputs on valid input
    are bit-identical to 1.1.1. tentib calls `corpus_set(text, 11)` at the default width. tentib
    keeps its own `puts`, since it prints through it, and the name does not collide with anything.
  - **akshara 1.1.2 still pins rosnet 1.2.2.** tentib's own pin, 1.4.1, wins, and the resolver says
    so ("rosnet 1.2.2 (wanted by akshara) not used"). This is safe: akshara calls only `t_alloc` and
    `t_zero`, and rosnet's API did not change. akshara's next release is expected to close the gap.

- **`cyrius.lock` regenerated from scratch**, as in 1.0.4 and 1.1.0. It has one commit pin per dep
  (`cdb74e2` tyche, `af2bb1a` rosnet, `b8b3f68` akshara), 116 file hashes, and the trailer
  `cyrius 6.7.18`. `cyrius deps --verify` and `cyrius deps --locked` both pass.

### Performance — the f64 baseline moved

tentib's kernels did not get faster or slower, but rosnet's f64 `linear_fwd`, which every "vs f64"
figure divides by, is now 2.4–2.9× faster per layer. These are the same harness and host as before,
run with `taskset -c 4`:

| | run of record (2026-07-06) | 1.1.1 |
|---|---:|---:|
| rosnet f64 `linear_fwd`, 768×768 | 541,591 ns | 189,228 ns |
| per-layer SIMD vs f64, 128² → 768² | 5.0× → 18.4× | **3.7× → 7.3×** |
| whole-model f64 | 2,378 tok/s | 3,166 tok/s |
| integer SIMD, mode 2 (re-packs per call) vs f64 | 1.3× | **0.9×** |
| packed serving | 13,675 tok/s | 13,382 tok/s |
| packed serving vs f64 | 5.7× | **4.2×** |

Pack-once serving is still 4.2× faster than f64 at the bench config. The per-layer integer-SIMD
kernel still beats f64 at every size, and its lead still grows with the layer. Mode 2, which re-packs
the weights on every call, is now slightly slower than f64; it is the reference-fairness path, not
the deployment one. `docs/benchmarks.md` gains a dated note with these figures. Its tables, the
README and `docs/api.md` still quote the 2026-07-06 run of record, which measured against the older
rosnet.

Re-verified at the new pin:
- The build passes, and the suite passes **101/101** (unchanged).
- The demo prints the same values as at 6.7.9.
- The `examples/quickstart.cyr` self-check passes.
- `programs/gpumm.cyr` exits **96** on the host, the expected result without a GPU, and still builds
  with `--agnos`.
- Fuzz passes 1/1.

The binary grows from 252,592 to 290,232 B, all of it unreachable stdlib code that DCE would drop
(532 → 682 unreachable fns).

## [1.1.0] — 2026-11-09

This release moves tentib onto the cyrius 6.7 series, as tyche 1.2.0, rosnet 1.2.0 and akshara 1.1.0
did. Nothing in `src/` changed, and the frozen 1.x public API is untouched.

### Changed

- **Toolchain `6.6.20` → `6.7.9`** (through 6.7.0–6.7.8). `cyrius lib sync --full` re-vendored the
  stdlib. `lib/` byte-matches the 6.7.9 snapshot: the same 113 files as 6.6.20, 67 of them with new
  content. Nothing was added or removed. `cyrius fmt --check` passes on every source, and lint gives
  the same 19 line-length warnings it gave at 6.6.2 and 6.6.20.
  I checked the 6.7.x language changes against tentib's own code (`src/`, `programs/`, `examples/`,
  `tests/`, with comments and strings excluded):
  - **6.7.0–6.7.8** reserve `trait`, `const`, `true`, `false`, `do` and `dyn`. tentib uses none of
    them as names.
  - **6.7.3** makes `!` a real operator; before that, a lone `!x` compiled as `x`. tentib has no `!`
    except in `!=`.
  - **6.7.9** makes unary `-` and `~` on a constant bind tighter. tentib has no unary `-` and no `~`.
    It writes negatives as `0 - n`.
  - **6.7.8–6.7.9** change how typed `*T` pointers step and dereference, and **6.7.6** rounds every
    write into an `f32`. tentib has no typed pointers and no `f32`. Its tensors are untyped addresses
    read with `load8` / `load64`.
  - **6.7.6** deprecates `cyrius tests`, and 6.7.12 removes it. CI and the docs already use bare
    `cyrius test`.
  - tentib's four `>>` shifts read a sign bit or a 2-bit code under an `& 1` / `& 3` mask, so they
    give the same result whether `>>` is logical or arithmetic.

- **Substrate deps moved to the tags each cut at its own 6.7.9 pin: `tyche` 1.1.1 → 1.2.0,
  `rosnet` 1.1.3 → 1.2.0, `akshara` 1.0.4 → 1.1.0.** The pins agree: rosnet 1.2.0 pins tyche 1.2.0,
  and akshara 1.1.0 pins rosnet 1.2.0. Each of the three reports no change to its source, API or
  output. Each checked that by comparing a dump of its outputs between 6.6.20 and 6.7.9, and found it
  byte-identical. Vendored `lib/tyche.cyr`, `lib/rosnet.cyr` and `lib/akshara.cyr` byte-match the
  tagged `dist/` bundles.

- **`cyrius.lock` regenerated from scratch**, as in 1.0.4. It has one commit pin per dep (`2c393fe`
  tyche, `f065c58` rosnet, `09ed724` akshara), 116 file hashes, and the trailer `cyrius 6.7.9`.
  `cyrius deps --verify` passes, and so does `cyrius deps --locked`, which is new in 6.7.6.

Re-verified at the new pin:
- The build passes, and the suite passes **101/101** (unchanged).
- The demo prints the same values as at 6.6.20: its output matches line for line apart from the
  timing lines (39 lines compared).
- The `examples/quickstart.cyr` self-check passes.
- `programs/gpumm.cyr` exits **96** on the host, the expected result without a GPU, and still builds
  with `--agnos`.
- Fuzz passes 1/1.
- The bench ratios reproduce: packed serving runs at 13,387 tok/s against 2,265 for f64 (5.9×). The
  per-layer SIMD speedup over f64 grows from 8.2× at 128² to 19.0× at 768².

## [1.0.4] — 2026-11-09

A toolchain and dependency realign. No source change; the frozen 1.x public API is untouched.

### Changed

- **Toolchain `6.6.2` → `6.6.20`.** `cyrius lib sync --full` re-vendored the whole version-matched
  stdlib snapshot: `lib/` byte-matches the 6.6.20 snapshot, 113 files, five of them new (`alloc_cx`,
  `boxed`, `hashseed`, `poison`, `tls_hostid`). The snapshot dropped nothing, so nothing was pruned.
  `cyrius fmt --check` is clean on every source, so the new pin brought no format drift.

- **Substrate deps realigned to the tags each cut at its own 6.6.20 pin: `tyche` 1.0.1 → 1.1.1,
  `rosnet` 1.1.1 → 1.1.3, `akshara` 1.0.2 → 1.0.4.** The chain is self-consistent: rosnet 1.1.3 pins
  tyche 1.1.1, and akshara 1.0.4 pins rosnet 1.1.3. Every symbol tentib calls is inside their frozen
  1.x sets. Vendored `lib/tyche.cyr`, `lib/rosnet.cyr` and `lib/akshara.cyr` byte-match the tagged
  `dist/` bundles. One upstream change reaches tentib: tyche 1.1.0 moved `rng_normal` to a portable
  logarithm, which shifts 0.011% of x86_64 draws by 1–3 ulp. rosnet's `t_randn` inherits that, and
  tentib's weight init draws through it. The suite does not compare drawn values, and it passes unchanged.
  rosnet 1.1.3 and akshara 1.0.4 also narrowed their `dist/*.deps` sidecars to the stdlib leaves each
  bundle needs. tentib already declares all of those leaves in its own `[deps] stdlib`, so this
  change does not affect it.

- **`cyrius.lock` regenerated from scratch.** `cyrius deps` added the new commit pins but kept the
  superseded tyche 1.0.1 / rosnet 1.1.1 / akshara 1.0.2 pins beside them. The regenerated lock has
  one pin per dep (`68bb5db` tyche, `5aa4ad7` rosnet, `60f56f5` akshara), 116 file hashes, and the
  `cyrius 6.6.20` toolchain line the 6.6.20 format records. `cyrius deps --verify` is clean.

Re-verified at the new pin: build; suite **101/101** (unchanged); the demo; the `examples/quickstart.cyr`
self-check (PASS); `programs/gpumm.cyr` on host exits **96** (byte-identical, no GPU, which is the
expected host result) and still cross-builds with `--agnos`; fuzz 1/1. Bench ratios reproduce: packed
serving runs at 13,032 tok/s against 2,171 for f64 (6.0×), and the per-layer SIMD speedup over f64 is
8.5× → 19.7× (128² → 768²).

## [1.0.3] - 2026-09-11

### Changed

- **Toolchain `6.5.27` → `6.6.2`.** No source change; the value form needed none.
  Build, tests, and any bench/fuzz/distlib target the repo ships re-verified at the new pin.


## [1.0.2] — 2026-08-18

### Changed

- **Cyrius pin `6.4.72` -> `6.5.27`** (2026-08-17, ecosystem-wide ML/AI-arc realign ahead of
  the arc reopening). `cyrius lib sync --full` re-vendored the whole version-matched stdlib
  snapshot, clearing the toolchain-drift and `./lib/ shadows version-pinned` warnings.
  Suite **101/101**, identical to the pre-bump baseline; the frozen 1.x public API is untouched.

### Changed

- **The crown proof stages on AGNOS as `/bin/gputern`, not `/bin/gpumm`** (agnos 1.56.32). No change
  to this repo — `programs/gpumm.cyr` keeps its name, and so does the local `build/gpumm` you get
  from a host build. Only the *rootfs* name moved.

- **Substrate dependency set brought current: `tyche` 0.1.1 -> 1.0.1, `rosnet` 0.2.0 -> 1.1.1,
  `akshara` 0.1.0 -> 1.0.2.** All three cut **1.0.0** in early July and froze their public surfaces;
  no consumer had moved off the pre-1.0 tags, so the freeze was real and its propagation was not.
  tentib sits in the last release wave, downstream of all three. This is a tag realign with no behavior change — every symbol
  tentib calls is inside the frozen sets. Verified each bump took rather than merely built: vendored
  `lib/tyche.cyr` at `1.0.1`, `lib/rosnet.cyr` at `1.1.1`, `lib/akshara.cyr` at `1.0.2`.
  Suite **101/101**, unchanged.

  Why: agnos's `stage-tools.sh` was staging this program and the standalone **gpumm** repo's seam
  probe under the same `/bin/gpumm`. tentib's row ran later and won the copy, so the 36,816 B seam
  probe never reached the image while both rows reported success. `/bin/gpumm` goes back to the
  gpumm repo (its own name, and the reference consumer of the `#82`/`#83` seam); this proof is named
  for the ternary integer path it demonstrates, alongside `gpulayer` (rupantara) and `gpuattn`
  (attn11 — which builds a `programs/gpumm.cyr` of its own and has always staged under a distinct
  name). agnos's `stage_one` also gained a duplicate-name guard so the next collision fails loudly.

  ⚠ **The 1.0.1 entry below says `run /bin/gpumm`** — accurate when written on 2026-07-23, and left
  as written. To re-run the iron crown today, `run /bin/gputern`.

## [1.0.1] — 2026-07-23

**AGNOS GPU offload (the 1.54.x GPU crown / C6, integer path): tentib's ternary integer matmul can run on the
AMD gfx90c shader cores.** Additive — `ternary_matmul_free_gpu` is a new peer of `ternary_matmul_free`; the CLI
and the frozen 1.x public API are unchanged (kernel internals are explicitly not frozen). This is the SECOND
GPU-crown consumer after rupantara's f64 path (`#83`), and the stronger one: integer accumulate is associative,
so it is **bit-exact at ANY K**, not just K≤8.

### Added
- **`src/gpu.cyr` — `ternary_matmul_free_gpu(qx, sx, Wq, gamma, b, y, M, K, N)`**, a bit-exact peer of
  `ternary_matmul_free` that tiles the integer accumulate `SUM_k qx·Wq` into 8×8×8 blocks and dispatches each
  on the gfx90c shader cores via the AGNOS ring-3 syscall **`#82`** (`gpu_dispatch`, i32 8×8 matmul), summing
  the tile results in i64 CPU-side; the f64 dequant (`gamma·sx·acc + b`) stays byte-identical to the CPU
  reference. **Bit-exact at any K** — integer addition is associative, so cross-tile i64 accumulation equals
  the reference's sequential sum exactly. **Signed-correct**: qx is int8-range and Wq is {−1,0,+1} so acc goes
  negative; `#82`'s MAC is `v_mul_lo_u32`+`v_add_u32` (two's-complement mod 2³², sign-transparent), and the
  low-32 result is sign-extended on readback. On non-AGNOS / no-GPU, each tile is computed directly from the
  i64 arrays — identical result. Returns the GPU tile count.
- **`programs/gpumm.cyr` — the crown proof.** Runs a real ternary projection `ternary_matmul_free(qx, sx, Wq,
  γ, 0, y, 8, 16, 32)` twice — CPU vs `ternary_matmul_free_gpu` — on **negative** activations + real ternary
  weights, and byte-compares the 2048-byte outputs. **K=16 = two k-tiles**, so the cross-tile accumulation the
  any-K claim rests on is exercised. Exit code: **95** = byte-identical AND all 8 tiles on the GPU (crown);
  **96** = byte-identical, 0 GPU tiles (host/QEMU — tiling + signed logic proven, no GPU); **90** = mismatch.

### Verified
- **tentib builds `--agnos` for the first time** (the crown-proof subset — kernel + rosnet/tyche/akshara; the
  file-streaming symbols are stubbed and unreachable).
- **Tiling + signed logic bit-identical on host** (2026-07-23): `build/gpumm` exits **96** —
  `ternary_matmul_free_gpu` (CPU-tiled, negative data, cross-tile K=16) is byte-for-byte equal to the
  production `ternary_matmul_free`.
- **CROWN PROVEN ON IRON** (2026-07-23): `run /bin/gpumm` → **`exit 95`** on archaemenid (AMD Cezanne) —
  byte-identical outputs AND all 8 tiles on the GPU, so a real ternary BitLinear projection's integer
  accumulate ran on the gfx90c shader cores via `#82`, **bit-exact at K=16 (cross-tile i64 accumulation)** —
  the any-K integer advantage over the f64 `#83` path. This is the AGNOS 1.54.x GPU crown (C6), second
  consumer (after rupantara/f64). **Bonus:** first iron confirmation of `#82` on NEGATIVE integers (its boot
  self-test only used positives; the signed two's-complement path is now proven). Tracked in agnosticos
  `iron-nuc-zen-log.md#tracker-156x-ml-crown-tentib`.

## [1.0.0] — 2026-07-06

**v1.0 — STABLE. The clean cut** (tarka/prajna/anukūlana precedent): **no code
change from 0.8.0** — the freeze and the criteria close it. tentib is the
sovereign integer-native / ternary (1.58-bit) ML reference: BitNet-b1.58 ternary
weights, STE training, and a matmul-free integer inference path that is
**bit-identical to its f64-latent reference and faster than the f64-SIMD matmul**
— trained, packed, and served entirely in Cyrius with no BLAS / libc / autodiff.

### All six v1.0 criteria green
- **BitLinear + STE, FD-gated** (0.2.0) — prove the surrogate, not the
  discontinuity; γ-cancellation + falsifiers.
- **A ternary transformer trains from scratch** (0.3.0) — RMSNorm + causal attn +
  GELU-MLP, all linears ternary, on akshara-tokenized text; every level FD-gated.
- **The matmul-free integer kernel reproduces the model** (0.4.0) — whole-model,
  exact parity < 1e-9, argmax 10/10, 2-bit packed weights (32×); + the 0.4.1
  integer-SIMD kernel (`iv_dp8`), bit-identical and 5–18× the f64-SIMD matmul.
- **Benchmarks** (0.5.0) — `docs/benchmarks.md`, B-series fairness; pack-once
  serving **~13.5k tok/s vs ~2.3k f64 (5.7×)** whole-model (0.7.0); honest
  toy-scale quality delta vs a matched-size f64 attn11, b1.58 ≥3B parity cited.
- **Public API frozen** (0.6.0) — `docs/api.md`, every symbol documented +
  directly gated; allocation audit (inference paths allocation-free). **The
  freeze is now in force for the 1.x series**: signatures/semantics stable,
  minors add only.
- **CHANGELOG complete + security/hardening audit pass** (0.8.0) —
  `docs/audit/2026-07-06-audit.md`; 6 findings fixed/guarded, 5 paths verified
  sound; fail-loud guards on cold paths only.

### Changed (docs only)
- Version-bearing docs swept to 1.0.0 (README status, api.md freeze declaration,
  SECURITY.md maturity language). Suite **101/101**; demo + bench + quickstart
  re-verified at the cut.

### Post-1.0 (user-driven levers, roadmap § out-of-scope)
- GPU ternary kernels (mabda) · hoosh/murti serving consumption of
  `tx_fwd_packed` · rotation-PTQ (QuaRot/SpinQuant-class) pairing with the
  Type-3 import lane · aarch64 NEON when cyrius Phase 5 lands.

## [0.8.0] — 2026-07-06

**Security/hardening audit + CHANGELOG completeness** — the last v1.0 criterion.
Audit record: [`docs/audit/2026-07-06-audit.md`](docs/audit/2026-07-06-audit.md)
(re-derived from source, comments not trusted). Posture = the tarka precedent:
fail-loud `guard()` on public preconditions whose violation means silent
corruption or silently-wrong numerics — **cold paths only, hot loops carry zero
checks** (packed serving throughput unchanged: 13,675 tok/s this run). Suite
**97 → 101**.

### Fixed — found by the audit
- **`ternary_one(w, gamma=0)` quantized all-zero weight vectors to all-“−1”**:
  w/0 → Inf/NaN → `f64_to` → INT_MIN → clipped to −1 (silent sign garbage through
  the public M0 pair; the M1+ chain was already safe — `bl_weight_effective`'s
  γ=0 branch verified real). Now γ==0 → 0 (b1.58: γ==0 ⟺ all-zero tensor).
  Regression-gated.

### Added — fail-loud guards (`guard(cond, msg)`, new public primitive)
- **`tpack2` / `tsimd_pack_w`**: a non-ternary weight no longer silently packs as
  0 / store8-truncates (the latter also voided the SIMD kernel's no-saturation
  exactness bound) — pack-time validation, fail-loud.
- **`ternary_matmul_free_simd`**: K ≤ 2²² enforced once per call (the i32
  accumulate is exact only for K ≲ 8.4M — the bit-identity contract's bound).
- **`tx_pack`**: fails loud before writing through null pointer tables if
  `tx_pack_init` / `tx_init` / `tx_int_init` haven't run (was silent heap
  corruption); **`tx_fwd_packed`**: fails loud before `tx_pack` (was silent
  garbage logits).

### Verified sound (audited, no change)
- `_tanh` (stable `e^(−2|x|)` form — the ganita `f64_tanh` overflow class does
  not apply, ±inf saturates to ±1) · `act_quant_int` zero-row + ±127 clamps ·
  the M1 degenerate-γ chain · `tx_int_init`/`tx_pack_init` sizing formulas vs
  all 7 BitLinear shapes · scalar kernels' i64 accumulate (no overflow at any
  allocatable K).

### Docs
- **CHANGELOG verified complete + gap-free 0.1.0 → 0.8.0**; all relative doc
  cross-links resolve (scripted check). `SECURITY.md`: stale version literal
  dropped, hardening-posture section added, the "0.8.0 pending" maturity note
  replaced with the audit record. `api.md`: `guard` added to the frozen surface
  + a guarded-preconditions convention note. `benchmarks.md`: run-of-record
  refreshed with guards live (pack ~4 → ~8 ns/weight — the only measurable cost,
  paid once per weight matrix; serving unchanged).

## [0.7.0] — 2026-07-06

**Downstream-consumer readiness — the pack-once deployment serving path.** The
piece the 0.5.0 benchmarks motivated: `tx_fwd_q` re-quantizes (mode 2: + re-packs)
every forward by reference-fairness design; a deployment packs **once** and serves.
All additions are **additive to the 0.6.0 frozen surface**. Suite **95 → 97**.

### Added — pack-once serving (`src/kernel.cyr`, public per `docs/api.md` §0.7.0)
- **`tx_pack_init(V, T, C, F)`** — allocate the packed store (7 per-layer
  transposed-i8 weight buffers + `wsum` tables + γs; attn Q/K/V/O, MLP up/down,
  head).
- **`tx_pack(V, T, C, F)`** — quantize + pack all 7 BitLinears from the live
  `tx_*` latent weights, once (~4 ns/weight; 6.4 ms at the bench config,
  amortized after ~3 forwards).
- **`tx_fwd_packed(tokens, V, T, C, F)`** — the serving forward: per call only the
  per-token activation quant runs, weights are never touched; logits → `ki_logits`.
  **Bit-identical to `tx_fwd_q` modes 0/2** (same quantize → same pack → same
  kernel, hoisted out of the call). Internal chain (`bl_forward_pk`,
  `*_sublayer_pk`) stays out of the freeze like its `_q` siblings.

### Benchmarks (`docs/benchmarks.md` §3 updated — same host/core/config)
- **PACKED serving = 2,375,236 ns/fwd → 13,472 tok/s = 5.8× the f64 path**
  (2,316 tok/s) and **3.8× mode 2** (3,580) — the whole-model number now reflects
  the §1 layer-sweep advantage instead of per-call prep; the residual 2.4 ms is
  dominated by the shared f64 interludes (RMSNorm/attention/GELU, kept f64 per
  b1.58) + the T×V head.

### Added — `examples/quickstart.cyr` (the worked consumer example)
- Train a tiny ternary LM → `tx_int_init`/`tx_pack_init`/`tx_pack` once →
  `tx_fwd_packed` serving, **public API only** (no internal-symbol reach-in),
  **self-checking** (packed argmax must equal the scalar integer kernel's at every
  position; non-zero exit on mismatch — running the example IS a test). Verified:
  builds + runs + PASS. (Placed at repo-root `examples/` per the tarka/anukūlana
  house convention; the roadmap's `docs/examples/` phrasing predates it.)

### Tests (95 → 97)
- Packed serving logits **bit-identical** to the scalar integer path on the
  trained M3c model, and **reproduced on a second serve** from the same store
  (serve-time read-only).

## [0.6.0] — 2026-07-06

**Allocation-clean + API freeze** (`docs/api.md`, the "API frozen" v1.0 criterion).
No behavior change to any shipped path — the cut settles and documents the surface,
closes the last indirect-only test coverage, and records the allocation audit.
Suite **90 → 95**.

### Added — `docs/api.md` (the frozen public surface)
- Every public symbol documented with its contract + "tested by" driver, in the
  tarka api.md house pattern: freeze policy (stable through 1.x from the 1.0.0 cut;
  minor versions add only), the **not-frozen internal mechanism** list (STE masks,
  FD probes, sublayer chains, `lm_train`-local helpers, scratch globals), caller
  conventions (everything-is-i64, no bounds checks, `b = 0` = no bias), and the
  documented output cells (`tx_logits`, `ki_logits`, `mlp_out`).
- **Allocation & lifetime contract** (the audit result): allocation happens ONLY
  in `blk_init`/`tx_init`/`tx_int_init` + the one-shot trainers (`lm_train`,
  `tx_train`); **every forward/backward/kernel path is allocation-free** — audited
  by walking all 32 `t_alloc`/`alloc` sites outside the demo driver. Repeated
  inference costs zero allocation; re-calling an `*_init` re-points its scratch
  family (bump allocator — old buffers unreachable, not freed).

### Tests (90 → 95) — the API-freeze gates
- **`ref_dot`** exact-value gate (the M0 baseline was demo-only).
- **`bl_forward_q` direct** (was reachable only through `tx_fwd_q`): mode 0 vs
  mode 2 — same γ returned + outputs bit-identical through the dispatcher.
- **The manual training-loop quartet** (`tx_zero_grads → tx_loss → tx_bwd →
  tx_sgd`) composed by the caller from a fresh init: 5 steps descend the loss,
  finite — gating the public surface `tx_train` wraps (`tx_sgd` was previously
  exercised only inside `tx_train`).

### Classified internal (not in the freeze)
- `lm_init`/`lm_eval_sweep`/`lm_correct` (operate on `lm_train`-local buffers),
  the FD probes + STE masks, the sublayer chains (`*_sublayer_*`, `block_q`),
  numeric helpers. Readable + FD-gated; may change between minors.

## [0.5.0] — 2026-07-06

**Benchmarks — the honest numbers behind the claims** (`docs/benchmarks.md`, a v1.0
criterion). The `tests/tentib.bcyr` stub is now the real harness driving every
number; B-series fairness throughout (identical shapes, warm cache, single pinned
CPU, every path pays its per-call prep, ratios-not-nanoseconds reproduction).
Run of record: cyrius 6.4.10, AMD Ryzen 7 5800H (Zen 3), 2026-07-06.

### Added — `docs/benchmarks.md` + the real bench harness
- **§1 per-layer kernel sweep** (M=1, pack-once SIMD): the SIMD advantage **grows
  with size** — vs the rosnet f64-SIMD matmul **5.0× at 128×128 → 18.4× at
  768×768** (3,124 → 29,361 ns/call; vs branchless scalar 29.5× → 123.1×). Two
  compounding mechanisms documented: 16 int8 lanes vs 2-wide SSE2, and i8 weights
  staying L2-resident where f64 spills (590 KB vs 4.7 MB at 768²).
- **§2 prep amortization**: `tsimd_pack_w` ~4 ns/weight, paid once — amortizes
  after ~81 calls at 768²; `act_quant_int` 8.3 µs/row.
- **§3 whole-model tok/s** (b1.58-shaped F=4C, 267k params, T=32): f64 2,307 →
  scalar-int 456 → **SIMD-int 3,549 tok/s (1.5× over f64)** — with the honest
  note that mode 2 re-quantizes AND re-packs per call (fairness mirror of the f64
  path's per-call quantization), so the pack-once deployment entry point (the
  0.7.0 deliverable) is bounded by §1, not §3.
- **§4 memory story measured**: 1,835,008 B f64 → 57,344 B 2-bit packed (32×) at
  the bench config; zero inner-loop multiplies.
- **§5 quality delta (honest)**: matched-size, same-corpus, same-steps f64 sibling
  — **attn11 1.13.0 (324 params) final CE 0.006 vs tentib ternary (252 params)
  CE 0.11**; both solve the task, ternary plateaus ~20× higher at toy scale;
  differences beyond quantization enumerated (bundled gap, not an ablation);
  b1.58 paper (arXiv:2402.17764, parity from ~3B) + bitnet.cpp (~2–6× over fp16)
  **cited, not re-demonstrated**.

### Changed
- `tests/tentib.bcyr`: noop stub → the benchmark harness (layer sweep ×4 shapes,
  prep costs, whole-model tok/s; `bench_new`/`bench_run`/`bench_avg_ns` +
  fnptr-dispatched kernels; random tokens use the masked `rng_u64` idiom).
- `cyrius.cyml` stdlib gains `fnptr` (the bench harness dispatches via `fncall0`).
- Suite unchanged at **90/90** (re-validated with the manifest change).

## [0.4.1] — 2026-07-06

**The integer-SIMD ternary kernel — "multiply-free is also FASTER."** The 0.4.0 gate
lifted: cyrius **6.4.6/6.4.7** (SIMD arc Phase 3) shipped integer SIMD, including
`iv_dp8(a, b, n)` — the widening u8·i8 → i32 dot (`pmaddubsw` → `pmaddwd` → `paddd`,
16 int8 lanes/instruction) that the filed proposal
([`docs/development/proposals/2026-06-23-cyrius-integer-simd.md`](docs/development/proposals/2026-06-23-cyrius-integer-simd.md))
asked for. The ternary kernel now **beats the f64-SIMD matmul it was ~5× behind**,
while staying **bit-identical** to the scalar kernel. Suite **86 → 90**.

### Added — the SIMD kernel (`src/kernel.cyr`)
- **`tsimd_pack_w(Wq, K, N, w8t, wsum)`** — the "quantize + pack once" deployment
  step: transposes `Wq` [K,N] (i64 ternary) to **`w8t` [N,K] i8 bytes** (each
  output's weights contiguous, as `iv_dp8` needs) + per-output **`wsum`** over the
  SIMD span (the u8-offset correction term). The i8 sibling of `tpack2`.
- **`ternary_matmul_free_simd(qx, sx, w8t, wsum, γ, b, y, M, K, N, xu8)`** — same
  contract + **bit-identical output** vs `ternary_matmul_free`. `pmaddubsw` is
  u8 × i8, so the signed int8 activation codes are offset to u8 (`+128`, exact
  algebra: `dot(x+128, w) = dot(x, w) + 128·Σw`, corrected by `wsum`); the `K % 16`
  tail runs the M3b branchless mask arithmetic on the same `w8t` bytes. Exactness
  argument documented in-file: ternary weights bound each `pmaddubsw` pair-sum by
  510 ≪ 32767 (the i16 stage never saturates), i32 accumulate safe to K ~ 2²⁴.
- **`bl_forward_q` mode 2** — whole-model SIMD inference through all 7 BitLinears
  (re-packs per call, matching mode 0's re-quantize-per-call fairness); new scratch
  `ki_w8t`/`ki_wsum`/`ki_xu8` in `tx_int_init`.

### Benchmarks (128×128 layer, M=1, 100 iters/kernel, x86-64 host, cyrius 6.4.10)
- **SIMD int8 kernel = 1,946 ns/call** vs **rosnet SIMD-f64 `linear_fwd` = 14,670 ns**
  — **~7.5× faster than the f64 baseline** (the 0.4.1 acceptance: the *"multiply-free
  is also faster"* demonstration). vs branchless scalar (M3b) 87,254 ns = **~45×**;
  vs branchy 102,652 ns = **~53×**. The 0.4.0 standings (scalar ternary ~5× *slower*
  than f64-SIMD) are inverted.

### Tests (86 → 90, all green)
- **M3d bit-identity** across the three K regimes: K=4 (K16=0, pure branchless
  tail — the whole-model C=4/F=8 shape), K=20 (SIMD span + tail, correction split),
  K=32 (pure `iv_dp8`, two lane-groups).
- **M3d whole-model**: `tx_fwd_q(mode 2)` logits **bit-identical** to the scalar
  integer path (mode 0) through all 7 BitLinears incl. the biased head.

### Changed
- **Cyrius pin 6.2.37 → 6.4.10** (`cyrius lib sync --full`; the 86/86 baseline
  re-validated green at the new pin before the kernel work — the bump crosses the
  6.3.13 array-locals-to-stack boundary, no fallout).
- Demo driver (`src/main.cyr`) M3/M3c sections print the SIMD parity + timing rows;
  the stale "toolchain does not yet expose integer SIMD" gate-note replaced.

## [0.4.0] — 2026-06-23

**M3 — the matmul-free integer inference kernel, whole-model.** The payoff: ternary
weights {−1, 0, +1} × int8 activations run as integer signed-accumulate (add / sub /
skip — *no multiply*) through **every BitLinear of the trained transformer**,
reproducing the f64 forward's logits. The scalar throughput lever (branch elimination)
lands here; the SIMD lever is gated on toolchain integer-SIMD (→ 0.4.1). Suite **86/86**.

### Added — M3a: the matmul-free integer kernel
- `src/kernel.cyr` — the payoff. `ternary_matmul_free(qx, sx, Wq, γ, b, y, M, K, N)`:
  ternary weights + int8 activations, **integer signed-accumulate** in the inner loop
  (`+qx / −qx / skip` over i64 — *no multiply*), then one dequant per output (`γ·sx`) +
  optional bias. `act_quant_int` produces the int8 codes (same quantization as M1's
  `bl_act_quant`, kept integer). The M0 `ternary_dot` generalized to a full layer +
  carried to genuine integers.
- **Logit parity** (gated): the integer kernel reproduces `bl_forward_full`'s f64
  logits to **< 1e-9** (the integer accumulate is exact; the f64 path is what rounds)
  — a 128×128 demo shows `max|int−f64| < 1e-12`.
- **2-bit packed ternary storage** (gated round-trip): `tpack2`/`tunpack2` pack
  `{−1,0,+1}` to 2 bits/weight — **32× smaller** than an f64 slot.

### Added — M3b: branchless scalar kernel (the achievable throughput lever)
- `ternary_matmul_free_bl` — the same matmul-free accumulate, but the data-dependent
  per-weight branch (unpredictable on random ±1/0 weights) is replaced by mask
  arithmetic (`pm = −(w==+1)`, `nm = −(w==−1)`, `acc += (qv & pm) − (qv & nm)`). Still
  genuinely multiply-free (AND / ADD / SUB only). **Bit-identical** to the branchy
  kernel (gated).
- **Measured (128×128 layer):** branchy 112.6 µs → branchless **84.5 µs** (~25%
  faster). rosnet's SIMD-f64 matmul is 16.5 µs — branch elimination helps but cannot
  beat SIMD.
- **The int-SIMD gate (honest):** the b1.58 CPU throughput win over SIMD-f64 needs
  INTEGER-SIMD lanes; this Cyrius has f64-only SIMD (2-wide SSE2, `f64v_*` →
  `movupd`/`mulpd`/`addpd`). Filed as a cyrius proposal; the SIMD kernel is **0.4.1**,
  gated on it. Today's wins: multiply-free + branch elimination + 32× memory.

### Added — M3c: whole-model integer inference
- `tx_fwd_q` (+ `bl_forward_q` / `attn_sublayer_q` / `mlp_sublayer_q` / `block_q` /
  `tx_int_init` / `logits_argmax`, `src/kernel.cyr`) — the matmul-free kernel runs
  through **every BitLinear of the trained transformer** (attention Q/K/V/O + MLP
  up/down + the head with bias); RMSNorm / GELU / attention stay f64 (b1.58 ternarizes
  only the linear layers). One mode-parameterized chain serves both the integer payoff
  (mode 0) and a full-quant-f64 reference (mode 1).
- **Whole-model parity** (gated): the integer logits reproduce the full-quant f64
  forward to **< 1e-9 relerr** across all 7 BitLinears (the M3a per-layer parity,
  composed); a trained "hello world" model shows `max|int−f64| < 1e-12`.
- **Honest activation-quant delta:** vs the weight-only **trained** model (which never
  saw int8-quantized activations), `max|int−trained| ≈ 0.01` — and the integer model's
  **next-token argmax matches the trained f64 model at 10/10 positions** (the small
  logit delta doesn't flip the rank). The honest small-scale gap, reported not hidden.
- **Latency:** the integer whole-model forward is 24 µs vs 16.5 µs weight-only f64
  (slower without integer-SIMD — the 0.4.1 gate).
- Suite **82 → 86**.

### Fixed
- `ternary_matmul_free` gained a bias param (8 → 9) for the M3c head; three stale
  8-arg call sites compiled clean (Cyrius does not check call arity) but failed the M3
  parity test at runtime — call sites corrected.

### Filed (toolchain)
- **cyrius issue** — call-site arity mismatch silently accepted (no arg-count check):
  compiles `OK`, binds garbage. The M3c bias-param regression above is the trigger.
  [`docs/development/issues/2026-06-23-cyrius-call-arity-no-check.md`](docs/development/issues/2026-06-23-cyrius-call-arity-no-check.md).
- **cyrius proposal** — integer SIMD (typed `iNxM` vectors + int8/16/32 ops), the
  quantized-ML throughput floor; unblocks the 0.4.1 int-SIMD kernel.
  [`docs/development/proposals/2026-06-23-cyrius-integer-simd.md`](docs/development/proposals/2026-06-23-cyrius-integer-simd.md).

## [0.3.0] — 2026-06-23

**M2 — a ternary transformer trains from scratch.** The full milestone: BitLinear
assembled into a pre-norm transformer block (RMSNorm + causal attention + GELU-MLP,
all six linear projections ternary), trained end-to-end via the STE on real
akshara-tokenized text. Every component *and* the full assembly is finite-difference
gated; the model trains (CE → ~0.1). The ternary sibling of attn11's first loss
curve, built assembly-up in sovereign Cyrius on rosnet — no BLAS / libc / autodiff.

### Added — M2: a ternary transformer trains from scratch
- `src/model.cyr` — softmax cross-entropy (`softmax_xent_fwd`/`_bwd`, ported from attn11; rosnet has no softmax) + a minimal trainable ternary LM: token id → **full-precision** embedding row → **BitLinear head** (ternary, weight-only) → softmax-CE. SGD walks the *latent* head weight (quantized only in the forward — the BitNet shadow-weight loop) + the full-precision embedding/bias.
- **Trains from scratch** on a synthetic successor-bigram task (`x → (x+1) mod V`): CE descends from the uniform `ln 8 ≈ 2.08` baseline to **~0.57** (2000 steps), memorizing **7/8** successors by argmax — the ternary sibling of attn11's first loss curve, at honest tiny scale (the persistent 1/8 mis-rank is the ternary capacity limit, not hidden).
- **FD-gates** (suite 34 → **57**): softmax-CE `dlogits` == central FD; the **end-to-end** head `dW` == FD of the linearized surrogate *with the real softmax-CE gradient frozen* (the M1 STE composes with the loss); saturated entries mask to 0 *through the loss*; the **embedding gradient** == FD of the live loss (the full chain); falsifier (FD-through-the-quantized-loss diverges > 0.5).
- Embedding stays full-precision (b1.58 ternarizes only linear layers).
- `src/layers.cyr` — differentiable transformer layers, each FD-gated standalone (suite 57 → **63**):
  - **RMSNorm** (pre-norm, the BitNet/Llama choice — no mean-subtraction; backward = LayerNorm minus the centering term, cross-checked vs attn11's `ln_bwd`).
  - **GELU** (tanh approximation, ported from attn11; `tanh` implemented from `f64_exp` since `f64_tanh` is not a builtin in this toolchain).
  - **Causal scaled-dot-product self-attention** (`attn_fwd`/`attn_bwd`) — single-head, `score = Q·K/√d`, causal softmax, value-sum; the full backward (`dQ`/`dK`/`dV` incl. the softmax-attention `P·(dP − ΣP·dP)` term) cross-checked vs attn11's `attn_core_bwd`, all three FD-gated.
- `src/block.cyr` — the **full ternary transformer block, assembled and end-to-end FD-gated** (module-global scratch + caches; suite 63 → **70**). Each level's dx gate (`dx == central FD` of `½‖out‖²`, exact since quantizers freeze under an x-perturbation) validates the entire fwd+bwd chain, catching any wiring bug:
  - **E1 — attention sublayer**: `RMSNorm → BitLinear Q/K/V → causal attention → BitLinear O → +residual` (whole-sequence M=T projections; the three Q/K/V `dx` sum; residual fold inside the sublayer bwd).
  - **E2 — MLP sublayer**: `RMSNorm → BitLinear up(C→F) → GELU → BitLinear down(F→C) → +residual`.
  - **E3 — full pre-norm block** = E1 ∘ E2; the per-sublayer residual fold makes `block_bwd` just thread `dout → dxmid → dx` (the rolling residual gradient). 6 BitLinear projections + 2 RMSNorms + GELU + attention + 2 residuals, all gradient-verified.
- **bite-E4: the ternary transformer LM trains from scratch — the "first loss curve."** token-embed + learned absolute pos-embed (full-precision) → the block → final RMSNorm → a *separate* ternary BitLinear head → per-position next-token softmax-CE; SGD on the latent weights + full-precision embeds/gains.
  - **E4a** — the full LM forward/backward, **end-to-end FD-gated**: `d_tokemb`/`d_posemb == FD` of the CE loss through the entire chain (embed scatter → block → final RMSNorm → head → softmax-CE).
  - **E4b** — **trains**: on a synthetic next-token sequence, CE descends `ln 6 ≈ 1.79 → ~0.04` (1500 steps) — the ternary sibling of attn11's first loss curve. Suite → **76**.
- **bite-F: trains on a REAL akshara-tokenized corpus.** Wired `[deps.akshara]` 0.1.0 (the shared sovereign tokenizer); `corpus_set(text,len)` byte-tokenizes `"hello world"` → V=8, `gd_ld(i)` reads the token ids, and the ternary transformer trains next-token on it: CE `ln 8 ≈ 2.08 → 0.11` (2000 steps). One tokenizer behind attn11, tarka, and tentib.

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
