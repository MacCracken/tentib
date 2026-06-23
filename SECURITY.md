# Security Policy

## Reporting

Report vulnerabilities to **cyriusmaccken@gmail.com**. Include reproduction
steps and the tentib version from `VERSION` (currently **0.4.0**). Expect an
initial response within one week. Coordinated disclosure is appreciated — do not
open a public GitHub issue with exploit details.

## Threat model

tentib is a single-process, CPU-only, **0.x** reference binary: a ternary
(1.58-bit) ML trainer + matmul-free integer inference kernel. It has **no
networking**, no privileged operations, and no persistence — it neither reads
nor writes files on its own. Its only input is the embedded, compile-time corpus
(a literal string byte-tokenized through akshara) plus the hyperparameters
compiled into `src/main.cyr`. There is no checkpoint-load, no `--corpus` /
`--stdin` file surface, and no untrusted-file deserialization path.

A realistic attacker is therefore assumed able to:

- supply the build inputs (source / embedded corpus) — i.e. they already control
  what gets compiled, which is equivalent to code execution, and/or
- edit `src/main.cyr` to drive the model with attacker-chosen dimensions.

tentib *does not* defend against:

- an attacker with arbitrary code execution in the build or host process
  (trivially owns the whole address space),
- remote / network attacks (tentib has no networking),
- side channels (timing, cache),
- adversarial *content* in the training corpus: tentib learns whatever the
  embedded text contains — its *meaning* is untrusted by definition.

## Attack surfaces & mitigations

| Surface | Reality / mitigation |
|---|---|
| **Input data** | The corpus is an embedded compile-time string, byte-tokenized by akshara into a fixed-size vocabulary; there is no runtime file or stdin read. Anyone who can change the corpus already controls the build. |
| **Numeric / memory layout** | All training and inference buffers (latent f64 weights, optimizer state, activation caches, the integer kernel's scratch) are allocated once at fixed, dimension-derived sizes via the `*_init` pattern; no allocation or unbounded growth in the training or inference loop. Indices are computed from compile-time dims. |
| **Packed-ternary codec** | `tpack2` / `tunpack2` pack `{−1, 0, +1}` to 2 bits/weight over fixed, dimension-derived buffers; the round-trip is gated in `tests/tentib.tcyr`. The packed bytes are produced and consumed in-process — there is no external packed-weight file to validate yet. |
| **Integer accumulate** | The matmul-free kernel accumulates int8 activations over ternary weights into i64; at the model's small K the signed accumulate cannot overflow i64. Realistic-K overflow bounds are a tracked hardening item for **0.8.0**. |
| **PRNG** | rosnet's weight init / sampling go through tyche (xorshift/splitmix-class) — used for reproducibility only, **never** for any security purpose. |
| **Supply chain** | The cyrius toolchain version is pinned in `cyrius.cyml` (single source of truth). The sibling deps (rosnet / tyche / akshara) are pinned by exact git tag. CI installs the toolchain via the upstream `scripts/install.sh`. |

## Maturity note

tentib is **pre-1.0 (0.4.0)** and has **not** had a formal security/hardening
audit. That audit is an explicit roadmap item — **0.8.0** (bounds on the
packed-ternary codec, the int8-quant clamps, integer-overflow on the accumulate
at realistic K, alloc sizing) is a stated v1.0 criterion. Until then, treat the
surfaces above as described from the code, not as audited guarantees. The repo's
correctness gate is the finite-difference suite (`tests/tentib.tcyr`), not a
security review.
