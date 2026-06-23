# tentib — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.4.0** — M3 (the matmul-free integer inference kernel, **whole-model**), cut
2026-06-23 (user tags). The kernel runs through every BitLinear of the trained
transformer: int8 activations + ternary weights → integer signed-accumulate (no
multiply), **exact parity < 1e-9** vs the full-quant f64 forward, **argmax 10/10** vs
the trained model, 2-bit packed weights (32×), + a **branchless scalar kernel** (~25%
over branchy). The SIMD throughput lever is deferred to **0.4.1**, gated on cyrius
integer SIMD (proposal filed). **86/86.** Prior: **0.3.0** (M2 — ternary transformer
trains), **0.2.0** (M1 — BitLinear + STE), **0.1.0** (M0 — ternary quantizer).

## Toolchain

- **Cyrius pin**: `6.2.37` (in `cyrius.cyml [package].cyrius`)

## Source

- `src/ternary.cyr` — the M0 core (raw f64, no deps): `absmean` (γ), `ternary_one`
  / `ternary_quantize` (clip(round(w/γ),−1,+1) → {−1,0,+1}), `ternary_dot`
  (matmul-free signed-accumulate · γ), `ref_dot` (full-precision baseline).
- `src/bitlinear.cyr` — **M1 BitLinear** on rosnet: `bl_weight_effective` (ternary
  Weff=γ·Wq), `bl_act_quant` (per-token int8), `bl_forward_wonly`/`bl_forward_full`;
  STE backward `bl_backward_wonly` + `bl_ste_mask_w`/`bl_ste_mask_x` (dW no-γ-factor
  clip-mask, dx through Weff); FD-gate losses `bl_surr_lin_loss`/`bl_loss_of_x`/
  `bl_quant_loss_wonly`.
- `src/model.cyr` — **M2 ternary LM**: `softmax_xent_fwd`/`_bwd` (ported from
  attn11), `lm_init`/`lm_forward`/`lm_train`/`lm_eval_sweep`/`lm_correct`, and
  `bl_xent_loss` (the FD-gate's quantized-loss probe). Embedding (full-precision) →
  BitLinear head (ternary) → softmax-CE, SGD on the latent weight.
- `src/layers.cyr` — **M2 differentiable layers**: `rmsnorm_fwd`/`bwd` (pre-norm),
  `gelu_fwd`/`bwd` (+ a `_tanh` from `f64_exp`), `attn_fwd`/`bwd` (single-head causal
  scaled-dot-product attention), each with a `*_loss` FD probe.
- `src/block.cyr` — **M2 full transformer block + LM** (E1-E4): `attn_sublayer`
  (E1) + `mlp_sublayer` (E2) + `block_fwd`/`bwd` (E3, pre-norm) + `tx_*` — the
  multi-token LM (embed + pos → block → final RMSNorm → ternary head → softmax-CE)
  with `tx_train` (E4). Module-global scratch; each level FD-gated; the LM trains
  (CE 1.86 → 0.04 on a synthetic sequence).
- `src/kernel.cyr` — **M3 matmul-free kernel + whole-model integer inference**:
  `act_quant_int` (int8 codes) + `ternary_matmul_free` (i64 signed-accumulate, no
  multiply, dequant + bias) + **M3b** `ternary_matmul_free_bl` (branchless mask
  arithmetic, bit-identical) + `tpack2`/`tunpack2` (2-bit packed storage). **M3c**
  whole-model integer inference: `bl_forward_q` (mode 0 integer / mode 1 full-quant-f64
  reference) + `attn_sublayer_q`/`mlp_sublayer_q`/`block_q`/`tx_fwd_q` (the trained
  transformer's 7 BitLinears, integer) + `tx_int_init` + `logits_argmax`.
- `src/main.cyr` — demo driver (M0 quantizer · M1 BitLinear · M2 transformer-trains ·
  M3 matmul-free kernel · M3b branchless bench · M3c whole-model integer inference).

## Tests

- `tests/tentib.tcyr` — **86/86** green: rosnet smoke, M0 quantization, BitLinear
  forward, the **M1 STE FD-gate** (dW vs surrogate, dx vs Weff, γ-cancellation @ γ=3,
  falsifiers, descent), int8 activation quant; **M2** — softmax-CE FD gate, the
  **end-to-end gradient gate** (head dW vs surrogate w/ softmax-CE dy, embedding
  grad vs live loss, falsifier), the **ternary-LM descent** (final CE < initial,
  argmax ≥ 6/8), **RMSNorm + GELU + attention** fwd/bwd FD gates, the **E1/E2/E3
  end-to-end dx gates** (attention sublayer, MLP sublayer, full block), the **E4a LM
  embedding-gradient FD gate**, the **E4b descent** + **bite-F** (trains on akshara
  text), **M3** (matmul-free integer kernel: logit-parity < 1e-9, 2-bit pack
  round-trip), **M3b** (branchless kernel bit-identical to branchy), and **M3c**
  (whole-model integer inference: < 1e-9 relerr vs the full-quant f64 forward,
  finite activation-quant delta vs the trained model, argmax agreement at all T).
- `tests/tentib.bcyr` / `.fcyr` — benchmark / fuzz stubs (no-op).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math
- **rosnet** 0.2.0 (latent f64 weights + `linear_fwd`/`linear_bwd`) — M1, active
- **tyche** 0.1.1 (PRNG; rosnet's `t_randn` → `rng_normal`) — M1, active
- **akshara** 0.1.0 (tokenizer; `corpus_set`/`gd_ld` byte-vocab path) — M2, active

## Consumers

_None yet._ (Eventual: hoosh / murti serving the ternary model.)

## Next

**v0.4.0 cut (M3 complete).** Next is **v0.4.1 — the integer-SIMD ternary kernel**, the
b1.58 wall-clock throughput lever. It is **gated on the cyrius toolchain** gaining
integer SIMD (typed `iNxM` vectors + int8/16/32 ops); the proposal is filed at
[`proposals/2026-06-23-cyrius-integer-simd.md`](proposals/2026-06-23-cyrius-integer-simd.md)
(+ the arity-check issue at [`issues/2026-06-23-cyrius-call-arity-no-check.md`](issues/2026-06-23-cyrius-call-arity-no-check.md)).
Until then the scalar matmul-free kernel (multiply-free + branchless + 32× memory) is
the shipped surface. Then **M4** (allocation-clean + API freeze + benchmarks → v1.0).
See [`roadmap.md`](roadmap.md).
