# Security Policy

## Reporting

Report vulnerabilities to **cyriusmaccken@gmail.com**. Include reproduction
steps and the tentib version from `VERSION`. Expect an initial response within
one week. Coordinated disclosure is appreciated — do not open a public GitHub
issue with exploit details.

## Threat model

tentib ships two things: a single-process, CPU-only **demo binary** and the
**library modules** in `src/` (the frozen 1.x API in `docs/api.md`) that a
consumer includes — a trainer, the matmul-free integer kernels, and the
pack-once serving path (`examples/quickstart.cyr`; hoosh / murti are the
intended serving consumers). Neither has networking, privileged operations or
persistence, and neither reads or writes files on its own.

- **The binary's only input** is the embedded, compile-time corpus (a literal
  string byte-tokenized through akshara) plus the hyperparameters compiled into
  `src/main.cyr`. There is no checkpoint load, no `--corpus` / `--stdin` file
  surface and no deserialization path.
- **The library's input is whatever the consumer passes**: token and target ids,
  dimensions, and weight / activation buffers. Token ids in particular are data
  — from a tokenized corpus, or on the serving path from a consumer's tokenizer —
  and a vocabulary mismatch is enough to produce one past the embedding table.

A realistic attacker is therefore assumed able to:

- supply the build inputs (source / embedded corpus) — i.e. they already control
  what gets compiled, which is equivalent to code execution;
- choose the token ids a consumer feeds the serving path (e.g. through prompt
  text and a mismatched tokenizer), and the dimensions a consumer passes.

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
| **Token / target ids** | Since 1.1.2 every whole-model forward (`tx_fwd`, `tx_fwd_q`, `tx_fwd_packed`), `tx_bwd` and `softmax_xent_fwd`/`_bwd` check each id against [0, V) once per call and **fail loud**. Before, an id ≥ V read past the embedding table, and in `tx_bwd` **wrote** C f64s past the gradient buffer (heap corruption). |
| **Dimensions and call order** | Since 1.1.2 each `*_init` records its dims, and every later call checks V, C, F (exact) and T (in [1, init T]) against them, and that the init ran. A shorter T is legal and serves the prefix's rows bit for bit. Before, a larger value read and wrote past init-owned buffers. |
| **Numeric / memory layout** | Training and inference buffers are allocated by the `*_init` pattern at dimension-derived sizes; no allocation or unbounded growth inside the training or inference loops. Caller-passed buffers (`y`, scratch) are caller-sized, documented per symbol in `docs/api.md`. |
| **NaN / Inf** | Since 1.1.2 a NaN activation propagates to NaN outputs on the integer paths (it used to become a valid-looking code and a finite logit). |
| **Packed-ternary codec** | `tpack2` / `tunpack2` pack `{−1, 0, +1}` to 2 bits/weight over dimension-derived buffers; the round-trip is gated in the suite and the fuzz harness. A non-ternary weight **fails loud** at pack time (`tpack2`, `tsimd_pack_w`, since 0.8.0). The packed bytes are produced and consumed in-process — there is no external packed-weight file yet. |
| **Integer accumulate** | The scalar kernels accumulate in i64 (\|acc\| ≤ 127·K). The SIMD kernel's iv_dp8 sums in i32 (255·K < 2³¹, K < ~8.4M) and is **guarded** at K ≤ 2²². Its activation codes must lie in [−128, +127] — a documented caller contract (`act_quant_int` produces [−127, +127]); a check inside the kernel loop was measured at 4–10% and not adopted (ADR 0002). |
| **Reporting** | A failed guard writes `tentib guard: <reason>` to **stderr** and exits 1 (since 1.1.2; it used to print through the consumer's `puts` on stdout). |
| **PRNG** | Weight init goes through tyche (xorshift) via rosnet — for reproducibility only, **never** for any security purpose. |
| **Supply chain** | The cyrius toolchain is pinned in `cyrius.cyml` (single source of truth); tyche / rosnet / akshara are pinned by exact git tag and commit (`cyrius.lock`, verified in CI with `--verify` and `--locked`). CI installs the toolchain via the upstream `scripts/install.sh`. Workflow actions are pinned by commit SHA, the token is read-only except on the release job, and checkouts do not persist credentials. |

## Hardening posture (0.8.0 and 1.1.2 audits)

Records: [`docs/audit/2026-07-06-audit.md`](docs/audit/2026-07-06-audit.md)
(0.8.0: 6 fixed/guarded, 5 verified sound) and
[`docs/audit/2026-11-09-audit.md`](docs/audit/2026-11-09-audit.md) (1.1.2: 2
MEDIUM, 9 LOW, 6 documentation, all fixed). Posture
([ADR 0002](docs/adr/0002-hardening-preserves-bits.md)): preconditions whose
violation would mean silent heap corruption or silently-wrong numerics are
guarded **fail-loud** at call entry or in cold passes — never inside a kernel's
loops — and every optimization is bit-identical on valid input, proven by a
golden dump. Buffer *sizing* stays caller-guarantees. The correctness gates are
the suite (`tests/tentib.tcyr`, 139 assertions, including every guard's fail
path in a forked child) and the property fuzzer (`tests/tentib.fcyr`), both run
on x86_64 and aarch64 in CI.
