# Security Policy

## Reporting

Report vulnerabilities to **cyriusmaccken@gmail.com**. Include reproduction
steps and the tentib version from `VERSION`. Expect an initial response within
one week. Coordinated disclosure is appreciated — do not open a public GitHub
issue with exploit details.

## Threat model

tentib is a single-process, CPU-only reference binary (**1.x stable**, API
frozen): a ternary (1.58-bit) ML trainer + matmul-free integer inference kernel. It has **no
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
| **Packed-ternary codec** | `tpack2` / `tunpack2` pack `{−1, 0, +1}` to 2 bits/weight over fixed, dimension-derived buffers; the round-trip is gated in `tests/tentib.tcyr`. Since 0.8.0, a non-ternary input weight **fails loud** at pack time (both `tpack2` and the SIMD `tsimd_pack_w`) instead of silently corrupting. The packed bytes are produced and consumed in-process — there is no external packed-weight file to validate yet. |
| **Integer accumulate** | The scalar kernels accumulate into i64 (\|acc\| ≤ 127·K — cannot overflow at any allocatable K). The SIMD kernel's i32 accumulate is exact for K ≲ 8.4M and **guarded** at K ≤ 2²² since 0.8.0 (once per call). |
| **PRNG** | rosnet's weight init / sampling go through tyche (xorshift/splitmix-class) — used for reproducibility only, **never** for any security purpose. |
| **Supply chain** | The cyrius toolchain version is pinned in `cyrius.cyml` (single source of truth). The sibling deps (rosnet / tyche / akshara) are pinned by exact git tag. CI installs the toolchain via the upstream `scripts/install.sh`. |

## Hardening posture (0.8.0 audit)

The **0.8.0 security/hardening audit is done** — record at
[`docs/audit/2026-07-06-audit.md`](docs/audit/2026-07-06-audit.md) (6 findings
fixed/guarded, 5 paths verified sound by re-deriving from source). Posture:
public-API preconditions whose violation would mean silent heap corruption or
silently-wrong numerics are guarded **fail-loud** (`guard()` — cold paths only;
hot loops carry zero checks; the packed serving throughput is unchanged). Buffer
*sizing* stays caller-guarantees per the rosnet substrate contract, documented
per symbol in [`docs/api.md`](docs/api.md). The repo's correctness gate remains
the finite-difference suite (`tests/tentib.tcyr`, 101/101).
