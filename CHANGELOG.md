# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.5.0] — 2026-11-09

**A load for serving only, and tentib measured on a Raspberry Pi 4.**
`tx_pack_deserialize_serving` reads and checks a model file exactly as `tx_pack_deserialize`
does, and the model serves the same logits bit for bit. It allocates only what serving reads:
1,434,536 bytes at the bench config instead of 12,192,728, 8.5× less. That is the first half of
the 1.5.x band.

It is a new entry point, not a change to the old one
([ADR 0004](docs/adr/0004-serving-only-load-is-a-new-entry-point.md)): `tx_pack_deserialize`'s
allocation is documented, frozen semantics, and hoosh's suite asserts it. No frozen signature or
semantics changed. The 796,704-byte golden dump of the existing outputs is byte-identical to
1.3.1 and 1.4.0, and so is the demo's output.

The release also carries the work committed after 1.4.0:
- the Raspberry Pi 4 measurement, the 1.6.x band's first half;
- `scripts/bench-history.sh`'s remote mode;
- documentation of hoosh 2.11.1's adoption.

### Added

- **`tx_pack_deserialize_serving(buf, n, max_bytes)` and `tx_serve_alloc_bytes(V, T, C, F)`**
  (`src/serial.cyr`).
  - **One loader.** Both entry points run `_pk_load`: the same header and field checks in
    the same order, the same status codes, the same copy-in. Only the memory figure the
    `max_bytes` check uses, and what is allocated, differ.
  - **What a serving load allocates.** `_tx_serve_alloc` allocates the packed store, the
    embeddings, gains and bias, and the scratch one forward writes. It sets every other
    buffer pointer of the `tx_*`, `blk_*`, `mlp_*` and `ki_*` families to 0. At the bench
    config: the i8 weight copy 229,376 bytes, the f64 parameters 300,032, the rest scratch.
  - **What a serving-only model refuses.** Every call that needs a missing buffer fails
    loud:
    - `tx_fwd`, `tx_fwd_q`, `tx_bwd`, `tx_sgd` and `tx_pack`, as on any loaded model;
    - `tx_param_init` and `tx_zero_grads`, through `tx_serve_only`, a new flag;
    - `block_fwd` / `block_bwd`, because the block's dims stay 0;
    - `bl_forward_q`, because the integer scratch is sized with no K×N room.

    It serves, decodes, reports its dims and vocabulary, and serializes. `tx_init` makes
    an ordinary model again.
  - **Reuse.** A serving load over a model loaded for serving with the same dims overwrites
    its buffers and allocates nothing (ADR 0003's rule, for this family). A serving load and
    a full load never reuse each other's buffers.
  - **Faster to load.** A fresh serving load takes 4.24 ms against 6.25 ms on x86_64, and
    36.9 ms against 81.6 ms on the Pi 4. Serving speed is unchanged: 2.22 against 2.20 ms,
    and 25.78 against 25.83 ms on the Pi.
- **`examples/serve.cyr`** loads with `tx_pack_deserialize_serving` and prints both memory
  figures (13,368 against 50,024 bytes for its model). It still builds from the bundles alone,
  and the cross-process and cross-target digests are unchanged.
- **Bench rows:** `deserialize serving` (fresh), `serving, same dims` and `PACKED, serving-only`.
- **Tests:**
  - **Suite 280 → 346 assertions.** The 1.5.0 group covers:
    - serving loads at three shapes: exact allocation, bit-identical serving and prefix
      rows, incremental decode, canonical re-serialization;
    - the bench figure;
    - the memory bound against the serving figure, and status codes for refused buffers;
    - reuse between serving loads, and none across the two loaders;
    - nine refused calls and four controls, each in a forked child;
    - that a full load still allocates `tx_model_alloc_bytes`.
  - **Fuzz:** new property P13 loads P9's two models for serving only (exact allocation,
    bit-identical serving and decode, nothing allocated on a same-dims serving reload,
    canonical bytes). P10 loads half its mutated buffers for serving only, against that
    loader's own figure.
  - **Everywhere:** both pass on x86_64, under qemu, and on the Raspberry Pi 4.
- **Mutation run:** 11 mutants of the new code, and the suite kills 10. The survivor drops
  the serving reuse rule's `tx_serve_only` test. It is equivalent, because a fully loaded
  model also fails that rule's block-dims and K×N-scratch tests.
- **Benchmarks on a Raspberry Pi 4, the 1.6.x edge target's first half.**
  `docs/benchmarks.md` §6 has the results, and `bench-history.csv` the `rpi4: …` rows (median
  of 3).
  - **The board:** Cortex-A72, 4 × 1.5 GHz, 2 GB, held at 1,500 MHz and 51–56 °C, pinned to
    core 3.
  - **Packed serving:** 25.8 ms per 32-token forward, 1,241 tok/s, against 434 for the f64
    path on the same board (2.85×; 3.2× on x86_64).
  - **Decoding a 32-token window:** 800.7 µs per cached step (1,249 tok/s), against 12.57 ms
    with a full forward per token (15.7×).
  - **Every whole-model row** runs about 10–11× slower than the Ryzen 7 5800H core.
  - **Two board-specific findings:**
    - the branchless scalar kernel gains almost nothing on the A72 (0.93–0.97 of branchy's
      time, against 0.61–0.83 on Zen 3);
    - the i64-weight scalar kernels fall off a cliff at 768² (15–25×, with no L3 behind the
      1 MB L2), while the SIMD kernel's i8 weights stay at 12.5×.
  - **The same day on the board:**
    - the fuzz harness held P1–P12 over 5,000 rounds (33 s);
    - a model file written on x86_64 served with its x86_64 digest;
    - a model trained on the board matched the qemu-trained digest.
  - **Not measured:** power (no meter), and a model sized for the board's 2 GB (that needs
    1.5.x's packed storage).
- **`scripts/bench-history.sh` runs on another machine.** `BENCH_HOST=<ssh host>` builds the
  harness `--aarch64`, copies it to a temporary file on the host, runs it RUNS times there
  (pinned to `BENCH_CPU`, default the host's last CPU), and removes it. It records the rows as
  `<BENCH_LABEL>: <row>`, so different machines' rows never mix. It prints the host's model,
  governor, clock, temperature and load before and after the runs. The host name is checked
  to be a plain ssh name, not something ssh would read as an option.

### Documented

- **[ADR 0004](docs/adr/0004-serving-only-load-is-a-new-entry-point.md)** records why the
  serving-only load is a new entry point. `api.md` has the two new rows and the refused calls.
  The format doc gives the serving loader's memory check. The consuming guide recommends the
  serving load to servers, and the dependency example uses tag 1.5.0. `SECURITY.md` covers
  the new loader.
- **hoosh 2.11.1 adopts 1.4.0.** It decodes with the cache, reads the header with
  `tx_pack_peek`, logs the declared vocabulary, and reloads same-shaped models without
  keeping memory. It shares the one decode cache between its interleaved replies: the
  cache belongs to the reply that last reset it, and another reply takes it over by
  stepping its window in.
  - The consuming guide's §4 gains that pattern, and `state.md` lists the release.
  - The 1.4.0 docs said hoosh's adoption would be a minor. It shipped as hoosh's patch
    2.11.1, in the band its roadmap had planned for it.
- **tentib ran on real ARM hardware for the first time.** The 1.4.0 aarch64 suite passed
  280/280 on a Raspberry Pi 4 (Cortex-A72), whose NEON lacks the dot-product extension.
  cyrius lowers `iv_dp8` to base ARMv8.0 NEON (`uxtl` / `sxtl` + `smlal`), so nothing
  needs it.

### Roadmap

- **1.5.x:** the serving-only allocation has shipped. What remains is the 2-bit / trit kernel,
  whose saving is the weight copy alone: under 0.2 MB of the 1.4 MB at the bench config, and
  nearly all of the memory at b1.58 scale.
- **The edge band (1.6.x) targets the Raspberry Pi 4 at hand, not a Pi 5.** The Pi 5 came
  from the agnosticos plan's target figure, not from anything tentib needs.
  - Its measuring half needs nothing from 1.5.x and no new hardware.
  - Sizing a b1.58-shaped model for the board's 2 GB needs 1.5.x's packed storage.
  - A Pi 5 is a second data point when one is available.
  - The plan's figure ("a 2B model at ~5–7 tok/s on a Pi 5 under ~15 W") has no source
    there, and the band says to check it before citing it.

## [1.4.0] — 2026-11-09

**What the first consumer asked for.** The three additive items that hoosh 2.11.0's
integration put on the roadmap at 1.3.1, plus the one call a consumer needs to use the
third:

- incremental decode;
- a reload that reuses the model's buffers;
- a vocabulary in the artifact, and `tx_pack_peek` to read it before a load.

No frozen signature or semantics changed, and on valid input every existing output is
bit-identical to 1.3.1. Two checks prove it:

- **A new 796,704-byte golden dump is byte-identical.** It covers, at four configs:
  training, every forward mode at every window length, the backward, packing,
  serialization, two loads, and the bare attention core with its gradients. It was
  recorded from a 1.3.1 worktree before the first edit.
- **The demo's output is identical, timings aside,** after the akshara bump below.

### Added

- **Incremental decode** (`src/decode.cyr`, a new library module; the constraints are
  [`docs/architecture/002-incremental-decode.md`](docs/architecture/002-incremental-decode.md)).
  - **The calls.** `tx_dec_reset()` starts a sequence on the live packed model.
    `tx_dec_step(tok)` runs the next position against a cache of the earlier positions'
    keys and values, writes its logits to `ki_logits[0..V)`, and returns the position.
    `tx_dec_len()` gives the positions cached, and `tx_dec_alloc_bytes(T, C)` = 16·T·C
    the cache's size.
  - **Bit-identical** to row p of `tx_fwd_packed` over the same p + 1 tokens. Every stage
    but attention is per position. The step computes its attention row with the function
    `attn_fwd` now runs for every row (`_attn_row`, factored out of `attn_fwd`), so every
    sum runs in the same order. The suite checks it at every window length on five
    models: C = 20 / F = 40, a trained C = 4 model, C = 32 / F = 64, T = 1, and V = 256.
    It also checks a loaded model, and interleaves full forwards between steps to show
    the cache survives them. Fuzz property P11 checks it on every random model, before
    and after the bytes round trip.
  - **Speed.** Decoding a 32-token window at the bench config (V = 256, T = 32, C = 128,
    F = 512) costs 73.5 µs per token, against 1.14 ms with a full forward per token
    (15.5×). Stepping a prompt in costs what one forward over it does (2.35 vs 2.33 ms
    for 32 tokens), so there is no separate prefill call.
  - **The window.** Positions are absolute, so the cache holds at most T, and a step on a
    full cache fails loud. Past T a consumer serves full forwards over the last T, which
    is what a sliding window has always cost.
  - **Staleness.** `tx_rev` is a model revision, bumped by `tx_init`, `tx_param_init`,
    `tx_sgd`, `tx_pack_init`, `tx_pack` and every successful load. A step on a cache
    older than the model fails loud.
  - **Memory.** The first reset allocates the cache, and so does a reset for a model
    larger than any before. Steps allocate nothing: the suite runs 50 sequences and
    allocates 0 bytes.
- **A load over a loaded model of the same dims reuses its buffers**
  ([ADR 0003](docs/adr/0003-a-load-reuses-only-a-loaded-models-buffers.md)).
  - **The rule.** `tx_pack_deserialize` allocates nothing when the live model is itself a
    loaded one and every init family was last sized for exactly the artifact's V, T, C,
    F. A server reloading a same-shaped model no longer keeps a model per reload: hoosh
    measured 802,376 bytes per load at T = 30, C = 16.
  - **Over a model built in the process,** a load allocates as before. That model's
    latent and gradient buffers hold training state a fresh load would have zeroed.
  - **The `max_bytes` check still uses the model's figure,** so whether a buffer loads
    does not depend on what was loaded before.
  - **Serving after a reuse load is bit-identical** to a fresh load. The suite runs 40
    alternating loads of two models and then a re-serialize; fuzz P12 loads a second
    model over the first and back, and checks that neither load allocates.
  - **Faster:** 4.22 ms against 6.73 ms for a fresh load at the bench config.
- **A vocabulary in the artifact.**
  - **The flag.** The format's flags word defines bit 0, the byte vocabulary: token id =
    byte value, V = 256. `tx_pack_set_vocab(kind)` declares it before
    `tx_pack_serialize`, and `tx_pack_vocab()` reports it. The kinds are
    `pk_vocab_none()` / `pk_vocab_bytes()`. `tx_pack_init` resets the declaration and
    `tx_pack` keeps it.
  - **The format stays version 1.** A model with no vocabulary declared writes the bytes
    1.2.0 – 1.3.1 wrote. Setting the flag changes only the flags word and the checksum.
  - **The decoder refuses any other bit, and bit 0 at V ≠ 256,** with `pk_err_header`. A
    1.2.0 – 1.3.1 decoder accepts only flags 0, so it refuses a flagged artifact the same
    way, as the format reserved.
- **`tx_pack_peek(buf, n, info)`** runs the load's header checks without loading. On
  success it writes V, T, C, F and the vocabulary; otherwise it returns the status the
  load would return for that header and writes nothing. A server can now refuse a V or a
  vocabulary it cannot serve before the load replaces its model. hoosh did this by
  reading offset 24 by hand. Fuzz P10 checks on every mutated artifact that the peek
  agrees with the load.
- **`examples/serve.cyr`** now uses the new calls:
  - load mode peeks the header first;
  - both modes decode the tokens incrementally and require the full forward's digest
    (exit 4 if not);
  - both modes generate greedily from a one-token prompt.

  It builds from the bundles alone, as before, on x86_64 and aarch64.
- **Bench rows:** `deserialize same dims`, `decode, full fwd/token` and
  `decode, cached step`. The `deserialize` row still measures an allocating load.
  `scripts/bench-history.sh` parses `ns/tok`.
- **Tests:**
  - Suite 188 → 280 assertions, in three groups:
    - decode: bit identity, allocation, eight guard paths and two controls;
    - reload: allocation, alternating loads, the rule's edges, a refused load;
    - vocabulary and peek: the round trip, the refusals, the peek's verdicts, five guards.
  - Fuzz properties: P10 gains a byte-vocabulary base artifact and the peek check, and
    P11 (decode) and P12 (same-dims reload) are new.
  - Both pass on aarch64 under qemu.
- **Mutation run:** 22 mutants of the new code.
  - **The suite kills 20.** The two survivors are explained:
    - dropping the `tx_T0` comparison from the reuse rule is equivalent, since the block
      and scratch checks compare T too;
    - a softmax max that skips one score changes both sides of every identity, and
      softmax is shift-invariant, so only the golden dump sees it. It does.
  - **The fuzz harness kills all 10 it was run against.**

### Changed

- **akshara 1.2.0 → 1.3.1** (test scope). 1.3.0 sped up `tok_encode`, and 1.3.1 checks
  `stream_tok` ids against the vocabulary. tentib calls neither; it uses only `corpus_set`
  / `gd_ld`, whose outputs are unchanged. `cyrius.lock` was regenerated with akshara
  `523992e`; rosnet `af2bb1a` and tyche `cdb74e2` are unchanged. The toolchain (6.7.18),
  rosnet (1.4.1) and tyche (1.2.2) were already the latest releases.
- **`attn_fwd` runs each row through `_attn_row`** (internal, not frozen); no bit changed.
- **`tx_pack_deserialize`'s header checks are now `_pk_header`,** shared with the peek. It
  checks the dims before the flags word now. Both failures are `pk_err_header`, so no
  status changed.
- **`dist/tentib.cyr` includes `src/decode.cyr`** (`[lib].modules`), and CI's docs check
  covers it.
- **`tx_model_alloc_bytes` keeps its value.** What it means is now "what a load allocates
  unless it reuses a loaded model's buffers" (`api.md`, the format doc).

### Roadmap

- The 1.3.x consumer asks are shipped. What remains of them is the 1.4.x band: a token
  table in the artifact, for models trained on a tokenizer other than bytes. Adopting
  1.4.0 in hoosh is a hoosh minor, and murti still waits for its port.
- The later bands moved up one: serving from packed storage is 1.5.x, the Pi 5 is 1.6.x,
  and GPU kernels via mabda are 1.7.x.

## [1.3.1] — 2026-11-09

**The first consumer.** hoosh 2.11.0 serves tentib models in its own process: a
`type = "tentib"` provider loads a packed-model file through `dist/tentib.cyr` and answers
chat completions with `tx_fwd_packed`, buffered and streamed. That is the roadmap's 1.3.x
item. murti, the other planned consumer, is still a Rust tree, so its adoption waits for its
Cyrius port. **No code changed.** The library and every output are 1.3.0's, and the bundles
differ only in their version stamp. What the integration found is documentation.

### Integration

- hoosh 2.11.0 declares `[deps.tentib]` (tag 1.3.0; this release's code is the same) with
  `modules = ["dist/tentib.cyr"]`. `cyrius deps` vendored tentib, rosnet 1.4.1 and tyche
  1.2.2, and no akshara, as 1.3.0 promised. No function or global name collides with hoosh,
  its vendored bundles or its stdlib set: 181 names on tentib's side, 3,007 on hoosh's.
  The gateway builds with no new warning on x86_64, aarch64 and AGNOS, and grows by 1.9%
  under DCE.
- hoosh's suite trains models through tentib's API, writes them, and serves them back
  through its provider: a known-answer reply, refused files (missing, V = 8, a flipped byte
  that tentib reports as `pk_err_checksum`), a stream, reloads, and the memory each load
  keeps, which matches `tx_model_alloc_bytes` to within 4 KiB.
- No accessor was missing: `tx_pack_dims` (1.3.0) gave hoosh the dims of a model it did
  not size.

### Documented

- **Serving from a long-running process**, a new §4 in
  [`docs/guides/consuming-tentib.md`](docs/guides/consuming-tentib.md), with hoosh as the
  worked example. Each point follows from the existing contract, and each was something
  hoosh had to handle:
  - one model per process, and no call is reentrant: serialize every call, hold the lock
    for one forward, and read `ki_logits` before releasing it;
  - `guard()` ends the process on a bad call, so a server makes bad calls impossible (ids in
    [0, V), a window of at most the model's T, nothing before a load);
  - format v1 has no tokenizer, so the consumer owns the vocabulary (hoosh: byte identity,
    V = 256, checked from the header before loading);
  - each new token is a full forward over the window (about 50 µs per token at C = 16 in
    hoosh);
  - each load keeps its `tx_model_alloc_bytes` for the life of the process, so reload only
    when the bytes change (hoosh compares SHA-256s) and reuse the read buffer;
  - a load replaces the model under its readers (hoosh's epoch check).
- The guide's dependency example pins `1.3.1`.

### Roadmap

- 1.3.x records hoosh's adoption and keeps murti's open until its port.
- Three things the integration asked for, proposed as additive 1.x work: **incremental
  decode** (a KV-cached step, bit-identical to the last row of a full forward), **a load
  that reuses the live model's buffers** when the dims match (no reload leak, no signature
  change), and **a vocabulary in the artifact** (through the reserved flags word).
- The 2.0.0 model-handles item now cites the measured cost of the global state: a server
  keeps every model it has loaded (802,376 bytes each at T = 30, C = 16).

## [1.3.0] — 2026-11-09

**tentib as a library.** Consumers (hoosh / murti, or any program) depend on tentib
through a distlib bundle and ship models as files. The release is additive: no frozen
signature changed, and on valid input every existing output is bit-identical to 1.2.0. The
333,281-value golden dump is byte-identical, including the akshara path after the dep
realign below.

### Added

- **`dist/tentib.cyr`**: the library bundle, built by `cyrius distlib` from a new `[lib]`
  module list (`src/ternary.cyr` through `src/serial.cyr`, in include order).
  `dist/tentib-gpu.cyr` holds the AGNOS `#82` tile path as a separate profile, as rosnet
  does for its GPU backend. A consumer declares only `[deps.tentib]` with
  `modules = ["dist/tentib.cyr"]`. `cyrius deps` then vendors rosnet and tyche with it, and
  not akshara (see below), so a consumer defines no akshara hooks and no printer.

  The bundle's sidecar and requires block carry the one stdlib leaf it needs, `math`. That
  leaf used to come in silently through akshara's sidecar; once akshara left the bundle's
  dependency set, distlib's aarch64 check showed that `f64_exp` / `f64_ln` need
  `lib/math.cyr`'s polyfills there. `src/layers.cyr` and `src/model.cyr` now include it
  themselves.

  This was checked with a throwaway consumer project that declared only `[deps.tentib]`. It
  got exactly tentib, rosnet and tyche, built without warnings, and served the same logits
  digest as the in-repo example.
- **`examples/serve.cyr`**, the consumer shape. It includes only the three dependency
  bundles and the stdlib modules it calls, and builds with `--no-deps`, so it proves the
  bundle compiles on its own.
  - `serve save <path>` trains, packs and writes a model file. `serve load <path>` reads it
    with a bounded read (64 MiB), loads it under a memory budget (1 GiB), and serves it.
  - Both modes print the argmax row and a digest of the logits, and the digests must match:
    the model crosses a process boundary bit for bit.
  - Exit codes: 1 for usage (an empty path or one past 4095 bytes is refused), 2 for an I/O
    error, 3 for a file `tx_pack_deserialize` refuses (its status is printed).
- **`tx_pack_dims(dims)`** reports the live packed model's V, T, C, F, which are the
  arguments `tx_fwd_packed` needs. Before it, a consumer that loaded a model it did not
  size itself had no API to learn them. Writing the example found the gap.
- **Portability checked under qemu:**
  - a model file written on x86_64 loads and serves on aarch64;
  - aarch64's own save → load round trip is bit-identical.

  Serving is promised bit-identical **within** a target. Across targets the integer kernels
  agree exactly, but `f64_exp` (attention, GELU) is only promised to within 1 ulp per
  target. On this test model the two targets happened to give the same digest. The format
  doc says so.
- **CI:**
  - a bundle drift check (`cyrius distlib --all` + `git diff --exit-code dist/`);
  - the consumer example: build with `--no-deps`, save in one process, load in a second,
    compare the digests, and require exit 3 on a corrupted file;
  - on aarch64: the example's round trip, and an x86_64-written file loaded under qemu;
  - the docs job requires both bundles and checks their version stamps.

  The **release** regenerates the bundles and ships them as `tentib-<tag>.cyr` and
  `tentib-gpu-<tag>.cyr` in `SHA256SUMS`.
- [`docs/guides/consuming-tentib.md`](docs/guides/consuming-tentib.md): declare the dep,
  ship a model, serve it, and the rules (untrusted files, the two bounds, token ids,
  packed-only, portability). It is linked from `api.md` (a new 1.3.0 section) and the
  getting-started guide.
- Suite 186 → 188 assertions (`tx_pack_dims` after a load, and its guard).

### Changed

- **akshara 1.1.2 → 1.2.0, as a test-scope dependency.** akshara 1.2.0 pins rosnet
  1.4.1, as tentib does, so the akshara → rosnet version mismatch the roadmap tracked is
  closed: the resolver's "rosnet 1.2.2 (wanted by akshara) not used" note is gone. 1.2.0
  speeds up `bpe_learn`; its outputs and signatures are unchanged, and tentib calls only
  `corpus_set` / `gd_ld`. The vendored `lib/akshara.cyr` byte-matches the tag, and
  `cyrius.lock` was regenerated: akshara `02a3d4e`, rosnet `af2bb1a`, tyche `cdb74e2`.

  `scope = "test"` because the library never tokenizes; only the demo, the examples and
  the harnesses do. The dep is still vendored and pinned by every resolve and prepended
  to test, bench and fuzz builds, while the demo and examples include it explicitly. A
  consumer never inherits it.
- `programs/gpumm.cyr` no longer includes akshara, which it never used. That removes four
  "undefined function" warnings its build always printed, and the AGNOS binary drops from
  317,296 to 300,312 bytes.
- `CLAUDE.md` names the bundle in the project type. `CONTRIBUTING.md` and the
  getting-started guide add the `cyrius distlib --all` step. `SECURITY.md` covers the
  bundle, the test-scope dep, the release assets and the example's bounded I/O. The
  roadmap's 1.3.x packaging item and the akshara maintenance item are shipped.

## [1.2.0] — 2026-11-09

**Packed-model serialization.** A trained, packed model can now be written to bytes and
loaded back, in another process, to serve bit-identically, with no latent weights and no
retraining. The release is additive: no frozen signature changed. On valid input every
existing output is bit-identical to 1.1.2; the 333,281-value golden dump is byte-identical.
Toolchain and deps are unchanged (cyrius 6.7.18; tyche 1.2.2, rosnet 1.4.1, akshara 1.1.2).

### Added

- **`src/serial.cyr`** adds the following functions:
  - `tx_pack_serial_bytes()` and `tx_pack_serialize(buf, cap)` write the live packed model;
  - `tx_pack_deserialize(buf, n, max_bytes)` loads one;
  - `tx_model_alloc_bytes(V, T, C, F)` gives the memory a model of those dims allocates;
  - `pk_err_*()` return the nine status codes.

  tentib still does no file I/O; the consumer owns the bytes. The format is specified in
  [`docs/architecture/001-packed-model-format.md`](docs/architecture/001-packed-model-format.md).
  - **What it holds:** the 7 BitLinears' γ and their ternary weights at **2 bits each**, in
    the serving layout, plus the f64 token / position embeddings, the three RMSNorm gains and
    the head bias. The SIMD kernel's `wsum` offsets are recomputed at load, not stored. At
    the bench config the artifact is 357,504 bytes: 57,344 of weights and 300,032 of f64
    parameters.
  - **Decoding is an untrusted-input surface.** Every field is checked before anything is
    allocated or changed:
    - the length, magic, version and flags;
    - each dim, in [1, 2²⁰];
    - the exact length the dims imply;
    - the memory the load needs, against the caller's `max_bytes`;
    - the FNV-1a checksum;
    - every γ (finite, non-negative), every weight code (no 3, zero padding), and every
      f64 parameter (finite).

    Bad data returns a status code (−1 … −9), never a `guard()` exit, and a refused load
    leaves the current model serving. The memory cap exists because rosnet's `t_alloc`
    exits the process when an allocation fails, so a header must not be able to ask for
    more than the consumer allowed. The checksum detects corruption; it does not
    authenticate.
  - **Canonical:** any accepted buffer re-serializes to the same bytes, and the encoder
    writes only buffers the decoder accepts. It refuses (−2) a model holding a non-finite
    parameter.
  - **A loaded model is packed-only.** It has no latent weights, so `tx_fwd`, `tx_fwd_q`,
    `tx_bwd`, `tx_sgd` and `tx_pack` fail loud on it rather than compute with zeros.
    `tx_fwd_packed` serves it at any T up to the stored T, and `tx_init` or
    `tx_param_init` makes it an ordinary model again.
- **Suite 139 → 186 assertions**:
  - the allocation formula equals `alloc_used()` at two configs;
  - round trips of a trained model and of a wide one (C = 20, F = 40, so every layer has a
    16-lane span and a tail), each served bit-identically after the in-memory model is
    overwritten, served at a prefix, and re-serialized to the same bytes;
  - **every status code** from a real mutation, with the checksum recomputed to reach the
    deeper checks, and C = 3 so padding bits exist;
  - 19 refused loads that leave the model serving;
  - the encoder's −1 and −2;
  - the five packed-only guards in forked children.
- **Fuzz P1–P8 → P1–P10:**
  - **P9** round-trips random small models through the bytes.
  - **P10** sends mutated artifacts to the decoder: bit flips, truncation and extension,
    header words, body bytes, γ and parameter values (NaN and Inf among them), and memory
    budgets below the need. The decoder must return a documented code, never exit, leave
    a refused load without effect, and re-serialize an accepted buffer canonically; an
    independent oracle checks the loaded values. All ten outcomes (0 and −1 … −9) occur in
    a run.
- **Mutation testing** of `src/serial.cyr`:

  | injected defect | caught by |
  |---|---|
  | a code of 3 accepted | suite and fuzz |
  | `wsum` recomputed over all K | suite and fuzz |
  | padding bits unchecked | suite and fuzz |
  | a negative γ accepted | suite and fuzz |
  | the memory budget off by 8 bytes | suite |
  | the encoder writing −1 as +1 | suite and fuzz |

- `examples/quickstart.cyr` step 5 serializes, reloads under a memory budget, and requires
  the same logits bit for bit. The demo prints the artifact size and the reload check.
  The bench gains `serialize` (3.6 ms) and `deserialize` (6.4 ms, which allocates
  12.2 MB) rows at the bench config, and `scripts/bench-history.sh` parses them.

### Fixed

- **The fuzz harness's random stream repeated.** `tx_param_init` reseeds tyche's global
  stream, and the harness drew its own random choices from that same stream. So each
  model init restarted the draws: with P9's constant seed the decoder mutations repeated
  every 16 rounds, and P8's model rounds cut the entropy to about 20 bits. Model inits in
  the harness now save and restore `_rng_state`.

### Changed

- `docs/api.md` adds the 1.2.0 section (symbols, status codes, the packed-only rule).
  `SECURITY.md` covers artifacts as untrusted input. CI checks `src/serial.cyr`'s docs and
  that the format doc exists. The roadmap's 1.2.0 item is shipped, and its 1.4.x item adds
  a serving-only allocation for loaded models, since today a load also allocates training
  buffers.

Measured on one binary per run, pinned to core 4, median of three runs. Alternating A/B
runs against 1.1.2's bench show no change in the forward, serving, packing or training
rows.

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
