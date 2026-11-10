# Security Policy

## Reporting

Report vulnerabilities to **cyriusmaccken@gmail.com**. Include reproduction
steps and the tentib version from `VERSION`. Expect an initial response within
one week. Coordinated disclosure is appreciated — do not open a public GitHub
issue with exploit details.

## Threat model

tentib ships two things: a single-process, CPU-only **demo binary** and the
**library** — the modules in `src/` (the frozen 1.x API in `docs/api.md`), shipped
since 1.3.0 as the distlib bundle `dist/tentib.cyr` — that a consumer includes: a
trainer, the matmul-free integer kernels, the pack-once serving path, and
packed-model serialization (hoosh / murti are the intended serving consumers).
Neither has networking, privileged operations or persistence, and neither reads or
writes files on its own. The consumer-shaped example `examples/serve.cyr` does file
I/O, as a consumer would: a bounded read (64 MiB) and a bounded load (1 GiB), with
the path taken from its command line (1 to 4095 bytes).

- **The binary's only input** is the embedded, compile-time corpus (a literal
  string byte-tokenized through akshara) plus the hyperparameters compiled into
  `src/main.cyr`. There is no checkpoint load, no `--corpus` / `--stdin` file
  surface and no deserialization path.
- **The library's input is whatever the consumer passes**: token and target ids,
  dimensions, weight / activation buffers, and (since 1.2.0) **the bytes of a
  packed model** handed to `tx_pack_deserialize`. Token ids are data — from a
  tokenized corpus, or on the serving path from a consumer's tokenizer — and a
  vocabulary mismatch is enough to produce one past the embedding table. A model
  artifact is a file the consumer read from somewhere; tentib treats its bytes as
  hostile.

A realistic attacker is therefore assumed able to:

- supply the build inputs (source / embedded corpus) — i.e. they already control
  what gets compiled, which is equivalent to code execution;
- choose the token ids a consumer feeds the serving path (e.g. through prompt
  text and a mismatched tokenizer), and the dimensions a consumer passes;
- supply or tamper with a packed-model artifact a consumer loads.

tentib *does not* defend against:

- an attacker with arbitrary code execution in the build or host process,
- remote / network attacks (tentib has no networking),
- side channels (timing, cache),
- a consumer passing buffers smaller than the documented sizing (no bounds
  checks on caller buffers — the rosnet substrate contract),
- adversarial *content* in the training corpus: tentib learns whatever the text
  contains — its *meaning* is untrusted by definition.

## Attack surfaces & mitigations

| Surface | Reality / mitigation |
|---|---|
| **Token / target ids** | Since 1.1.2 every whole-model forward (`tx_fwd`, `tx_fwd_q`, `tx_fwd_packed`; since 1.4.0 `tx_dec_step`), `tx_bwd` and `softmax_xent_fwd`/`_bwd` check each id against [0, V) once per call and **fail loud**. Before, an id ≥ V read past the embedding table, and in `tx_bwd` **wrote** C f64s past the gradient buffer (heap corruption). |
| **Dimensions and call order** | Since 1.1.2 each `*_init` records its dims, and every later call checks V, C, F (exact) and T (in [1, init T]) against them, and that the init ran. A shorter T is legal and serves the prefix's rows bit for bit. Before, a larger value read and wrote past init-owned buffers. |
| **Incremental decode (1.4.0)** | `tx_dec_step` writes one key and one value row per step into a cache sized for the live model's T·C, and **fails loud** before it would write row T (a full cache), on an id outside [0, V), on a cache older than the model (`tx_rev`, bumped by every model change and load), and before any packed model. A stale cache would mix two models' keys and values, which is wrong output but not out of bounds; the revision check makes it an exit instead. The cache grows only on a reset for a larger model. |
| **Numeric / memory layout** | Training and inference buffers are allocated by the `*_init` pattern at dimension-derived sizes; no allocation or unbounded growth inside the training or inference loops. Caller-passed buffers (`y`, scratch) are caller-sized, documented per symbol in `docs/api.md`. |
| **NaN / Inf** | Since 1.1.2 a NaN activation propagates to NaN outputs on the integer paths (it used to become a valid-looking code and a finite logit). |
| **Packed-ternary codec** | `tpack2` / `tunpack2` pack `{−1, 0, +1}` to 2 bits/weight over dimension-derived buffers; the round-trip is gated in the suite and the fuzz harness. A non-ternary weight **fails loud** at pack time (`tpack2`, `tsimd_pack_w`, since 0.8.0). |
| **Packed-model bytes (1.2.0)** | `tx_pack_deserialize(buf, n, max_bytes)` checks every field before it allocates or changes anything — length, magic, version, flags, each dim in [1, 2²⁰], the exact length the dims imply, the memory the load needs against the caller's `max_bytes`, the checksum, every γ (finite, non-negative), every weight code (no 3, zero padding), every f64 parameter (finite) — and returns a status code for bad data, never a `guard()` exit; a refused load leaves the current model serving. Derived data (the SIMD `wsum` offsets) is recomputed, not read. The **memory cap** matters because rosnet's `t_alloc` exits on a failed allocation. The checksum (FNV-1a 64) detects corruption only; it is not an authentication tag — tentib does not authenticate artifacts, so a consumer that needs provenance must check it before loading. Gated in the suite (every status code; refused loads keep the model) and by fuzz property P10 (mutated artifacts: documented codes only, never an exit, canonical when accepted). Format: [`docs/architecture/001-packed-model-format.md`](docs/architecture/001-packed-model-format.md). **1.4.0:** `tx_pack_peek(buf, n, info)` runs the same header checks without loading, and reads `buf[0..64)` only once `n ≥ 72` (P10 checks it agrees with the load on every mutated artifact). The flags word accepts only bit 0, the byte vocabulary, and only at V = 256. A load over a loaded model of the same dims overwrites its buffers, but only after every check has passed, so a refused buffer still changes nothing, and the cap still applies. |
| **Integer accumulate** | The scalar kernels accumulate in i64 (\|acc\| ≤ 127·K). The SIMD kernel's iv_dp8 sums in i32 (255·K < 2³¹, K < ~8.4M) and is **guarded** at K ≤ 2²². Its activation codes must lie in [−128, +127] — a documented caller contract (`act_quant_int` produces [−127, +127]); a check inside the kernel loop was measured at 4–10% and not adopted (ADR 0002). |
| **Reporting** | A failed guard writes `tentib guard: <reason>` to **stderr** and exits 1 (since 1.1.2; it used to print through the consumer's `puts` on stdout). |
| **PRNG** | Weight init goes through tyche (xorshift) via rosnet — for reproducibility only, **never** for any security purpose. |
| **Supply chain** | The cyrius toolchain is pinned in `cyrius.cyml` (single source of truth); tyche / rosnet / akshara are pinned by exact git tag and commit (`cyrius.lock`, verified in CI with `--verify` and `--locked`). akshara is a test-scope dependency (1.3.0), so a consumer of `dist/tentib.cyr` inherits only rosnet and tyche. The bundles are regenerated and diffed in CI (drift check), and ship as release assets listed in `SHA256SUMS`. CI installs the toolchain via the upstream `scripts/install.sh`. Workflow actions are pinned by commit SHA, the token is read-only except on the release job, and checkouts do not persist credentials. |

## Hardening posture (0.8.0 and 1.1.2 audits)

Records: [`docs/audit/2026-07-06-audit.md`](docs/audit/2026-07-06-audit.md)
(0.8.0: 6 fixed/guarded, 5 verified sound) and
[`docs/audit/2026-11-09-audit.md`](docs/audit/2026-11-09-audit.md) (1.1.2: 2
MEDIUM, 9 LOW, 6 documentation, all fixed). Posture
([ADR 0002](docs/adr/0002-hardening-preserves-bits.md)): preconditions whose
violation would mean silent heap corruption or silently-wrong numerics are
guarded **fail-loud** at call entry or in cold passes — never inside a kernel's
loops — and every optimization is bit-identical on valid input, proven by a
golden dump. Buffer *sizing* stays caller-guarantees. Untrusted *bytes* (a
packed-model artifact, 1.2.0) are validated in full and refused with a status
code. The correctness gates are the suite (`tests/tentib.tcyr`, 186 assertions,
including every guard's fail path in a forked child and every decoder status
code) and the property fuzzer (`tests/tentib.fcyr`, properties P1–P10), both run
on x86_64 and aarch64 in CI.
