# Consuming tentib as a library

How a program (hoosh, murti, or your own) depends on tentib, ships a trained ternary
model as a file, and serves it. Since 1.3.0. The worked, CI-gated example is
[`examples/serve.cyr`](../../examples/serve.cyr).

## 1. Declare the dependency

tentib ships a distlib bundle, `dist/tentib.cyr`. One entry in your `cyrius.cyml`
is enough:

```toml
[deps.tentib]
git = "https://github.com/MacCracken/tentib"
tag = "1.3.0"
modules = ["dist/tentib.cyr"]
```

`cyrius deps` resolves tentib's own dependencies with it: it vendors `lib/tentib.cyr`,
`lib/rosnet.cyr` and `lib/tyche.cyr`. akshara is tentib's test-scope dependency, so
you do not get it and do not define its hooks. The bundle's requires block includes
the one stdlib leaf it needs (`math`). tentib calls no printer you must define.

The AGNOS GPU tile path (`ternary_matmul_free_gpu`) is a separate bundle: add
`"dist/tentib-gpu.cyr"` to `modules` if you call it.

## 2. Ship a model: train, pack, serialize

```cyrius
tx_train(tokens, targets, V, T, C, F, steps, seed, stats);   # or your own training loop
tx_int_init(V, T, C, F);
tx_pack_init(V, T, C, F);
tx_pack(V, T, C, F);                                        # quantize + pack, once

var n = tx_pack_serial_bytes();
var bytes = alloc(n);
tx_pack_serialize(bytes, n);                                # n bytes: write them anywhere
```

tentib does no file I/O. Write the bytes with your own I/O (the example uses
`file_write_all`, which treats a short write as an error).

## 3. Serve a model: read, load, serve

```cyrius
var lenp[8];
var buf = file_read_whole_max(path, MAX_FILE, &lenp);   # bound the read
var rc = tx_pack_deserialize(buf, load64(&lenp), MAX_LOAD);   # bound the memory
if (rc != 0) { ... }                                    # a pk_err_* code; nothing changed

var dims[32];
tx_pack_dims(&dims);                                    # V, T, C, F of the loaded model
tx_fwd_packed(tokens, V, T_used, C, F);                 # T_used in [1, T]; logits in ki_logits
```

- **Treat the file as untrusted.** `tx_pack_deserialize` checks every field before it
  allocates or changes anything, returns a status code for bad data (it never exits),
  and keeps the current model serving when it refuses. The checksum detects
  corruption, not tampering: if you need provenance, verify it before loading.
- **Choose the two bounds.** `MAX_FILE` caps your read, and `MAX_LOAD` caps what a load
  may allocate (`tx_model_alloc_bytes(V, T, C, F)` is the exact figure for a model).
  The load bound matters: rosnet's allocator exits the process when an allocation fails.
- **Token ids must be in [0, V).** `tx_fwd_packed` fails loud on one that is not, so map
  your tokenizer's ids onto the model's vocabulary before you serve.
- **A loaded model only serves.** It has no latent weights, so `tx_fwd`, `tx_fwd_q`,
  `tx_bwd`, `tx_sgd` and `tx_pack` refuse it.
- **Portability.** A model file written on x86_64 loads on aarch64 and the reverse (the
  format is little-endian on both). Serving is bit-identical within a target. Across
  targets the f64 parts of the forward (`f64_exp` in attention and GELU) are promised to
  within 1 ulp, not bit for bit.

The format, every check and every status code are specified in
[`../architecture/001-packed-model-format.md`](../architecture/001-packed-model-format.md);
the full API is [`../api.md`](../api.md).
