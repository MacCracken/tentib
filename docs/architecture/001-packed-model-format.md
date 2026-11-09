# 001 — The packed-model format (version 1)

What `tx_pack_serialize` writes and `tx_pack_deserialize` reads (`src/serial.cyr`,
since 1.2.0). A reader in another language can implement it from this page; the
decoder's checks below are part of the format, not an implementation detail.

## What it holds

Everything `tx_fwd_packed` reads, and nothing it does not: the seven BitLinears'
ternary weights and scales, plus the f64 parameters that stay full precision under
b1.58. **No latent weights**, so a loaded model can be served but not trained or
re-packed.

## Layout

All words are 8 bytes, little-endian (the native order of both cyrius targets).
Offsets are in bytes from the start of the artifact.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic: the ASCII bytes `TENTIBPK` |
| 8 | 8 | format version: `1` |
| 16 | 8 | total length of the artifact in bytes, checksum included |
| 24 | 8 | V, vocabulary size |
| 32 | 8 | T, the context length (rows of the position embedding) |
| 40 | 8 | C, model width |
| 48 | 8 | F, MLP width |
| 56 | 8 | flags: `0` (no flag is defined in version 1) |
| 64 | … | the seven layers, in order (below) |
| … | 8·(V·C + T·C + 3C + V) | the f64 parameters (below) |
| length − 8 | 8 | FNV-1a 64 checksum of bytes [0, length − 8) |

**Layers**, in the packed store's order: 0–3 = attention Q / K / V / O (K = C,
N = C), 4 = MLP up (K = C, N = F), 5 = MLP down (K = F, N = C), 6 = the head (K = C,
N = V). Each layer is:

- **γ**, its f64 scale (8 bytes);
- **K·N weight codes**, 2 bits each, 4 per byte, in the serving layout: code `j` is
  weight `(n, k)` with `j = n·K + k` (output-major, as `tsimd_pack_w` lays out
  `w8t`). Code `j` sits in byte `j / 4` at bit `2·(j mod 4)`: `0` = 0, `1` = +1,
  `2` = −1. Code `3` is invalid. The block is padded with zero bits to a whole byte
  and with zero bytes to a multiple of 8.

**f64 parameters**, as raw bit patterns: the token embedding (V·C, row-major by
token), the position embedding (T·C), the two RMSNorm gains of the block (2C, the
attention norm's then the MLP norm's), the final RMSNorm gain (C), and the head bias
(V).

The length is therefore
`64 + Σ_layers (8 + pad8(ceil(K·N / 4))) + 8·(V·C + T·C + 3C + V) + 8`.
At the bench config (V = 256, T = 32, C = 128, F = 512) that is 357,504 bytes, of
which the 229,376 ternary weights take 57,344 and the f64 parameters 300,032.

## What the decoder checks, in order

`tx_pack_deserialize(buf, n, max_bytes)` returns `0` or the first failure:

| Check | Status |
|---|---|
| `n ≥ 72` | `pk_err_length()` = −1 |
| the magic | `pk_err_magic()` = −2 |
| version `1` | `pk_err_version()` = −3 |
| flags `0`; each of V, T, C, F in [1, 2²⁰] | `pk_err_header()` = −4 |
| the length word and `n` both equal the length the dims imply | `pk_err_length()` = −1 |
| `tx_model_alloc_bytes(V, T, C, F) ≤ max_bytes` | `pk_err_too_big()` = −6 |
| the checksum | `pk_err_checksum()` = −5 |
| per layer: γ is finite and not negative (sign bit clear) | `pk_err_gamma()` = −8 |
| per layer: no code is 3; every padding bit and byte is 0 | `pk_err_weights()` = −7 |
| every f64 parameter is finite | `pk_err_param()` = −9 |

Every check runs before anything is allocated or changed, so a refused buffer leaves
the current model serving exactly as before. The SIMD kernel's `wsum` offsets are
derived data and are recomputed from the codes, never read from the artifact.

## Invariants

- **Canonical.** A buffer the decoder accepts re-serializes to the same bytes, and
  every artifact `tx_pack_serialize` writes, the decoder accepts. (The encoder
  refuses, with −2, a model holding a non-finite parameter or an invalid γ.)
- **Bit-identical serving.** A loaded model serves the same logits, bit for bit, as
  the live packed model it was written from, at every T up to the stored T.
- **Bounded memory.** A load allocates exactly `tx_model_alloc_bytes(V, T, C, F)`
  bytes, the four `*_init` calls' total, and refuses before allocating if that
  exceeds `max_bytes`. The cap matters because rosnet's `t_alloc` exits the process
  when an allocation fails. Today a load runs the full `tx_init`, latent and gradient
  buffers included: 12,192,728 bytes for the 357,504-byte bench artifact.
- **The checksum detects corruption, it does not authenticate.** Anyone can recompute
  it; memory safety comes from the field checks, not from the checksum.
- **A loaded model has no latent weights.** `tx_fwd`, `tx_fwd_q`, `tx_bwd`, `tx_sgd`
  and `tx_pack` refuse it (they would compute with the zeros `tx_init` allocated)
  until `tx_init` or `tx_param_init` makes an ordinary model again.

## Versioning

A change to the layout or to the meaning of a field is a new format version; this
decoder refuses any version but 1. The flags word is reserved for additive variants
(for example a future layout that also carries the latent weights), and version 1
requires it to be 0.
