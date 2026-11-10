# Consuming tentib as a library

How a program (hoosh, murti, or your own) depends on tentib, ships a trained ternary
model as a file, and serves it. Since 1.3.0. The worked, CI-gated example is
[`examples/serve.cyr`](../../examples/serve.cyr). A server that stays up, reloads and
answers many callers has more to get right: see [§4](#4-serve-from-a-long-running-process),
written from hoosh 2.11.0's adoption (1.3.1) and updated for what 1.4.0 added in answer
(incremental decode, buffer-reusing reloads, a vocabulary in the file).

## 1. Declare the dependency

tentib ships a distlib bundle, `dist/tentib.cyr`. One entry in your `cyrius.cyml`
is enough:

```toml
[deps.tentib]
git = "https://github.com/MacCracken/tentib"
tag = "1.4.0"
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

tx_pack_set_vocab(pk_vocab_bytes());                        # 1.4.0, optional: token id = byte value (V = 256)
var n = tx_pack_serial_bytes();
var bytes = alloc(n);
tx_pack_serialize(bytes, n);                                # n bytes: write them anywhere
```

Declare the byte vocabulary when the model was trained on bytes as token ids: a server
can then check it instead of assuming it. Leave it out for any other tokenizer (then the
file's ids mean what you document; a token table in the file is open roadmap work). A
declared file needs tentib 1.4.0 or later to load.

tentib does no file I/O. Write the bytes with your own I/O (the example uses
`file_write_all`, which treats a short write as an error).

## 3. Serve a model: read, load, serve

```cyrius
var lenp[8];
var buf = file_read_whole_max(path, MAX_FILE, &lenp);   # bound the read
var info[40];
var pr = tx_pack_peek(buf, load64(&lenp), &info);       # 1.4.0: V, T, C, F, vocabulary
if (pr != 0) { ... }                                    # the status the load would return
# refuse here a V or vocabulary you cannot serve: nothing has changed yet
var rc = tx_pack_deserialize(buf, load64(&lenp), MAX_LOAD);   # bound the memory
if (rc != 0) { ... }                                    # a pk_err_* code; nothing changed

var dims[32];
tx_pack_dims(&dims);                                    # V, T, C, F of the loaded model
tx_fwd_packed(tokens, V, T_used, C, F);                 # T_used in [1, T]; logits in ki_logits
```

To generate, decode one token at a time (1.4.0). Each step runs one position against
the cached keys and values of the earlier ones, and its logits are the same bits as
`tx_fwd_packed`'s row for that position:

```cyrius
tx_dec_reset();                                         # a new sequence
var i = 0;
while (i < n_prompt) { tx_dec_step(load64(prompt + i * 8)); i = i + 1; }
while ((tx_dec_len() < T) & (more)) {                   # the cache holds at most T
    var next = logits_argmax(ki_logits, V);             # the newest position's logits
    tx_dec_step(next);                                  # emit next, feed it back
}
```

- **Treat the file as untrusted.** `tx_pack_deserialize` checks every field before it
  allocates or changes anything, returns a status code for bad data (it never exits),
  and keeps the current model serving when it refuses. The checksum detects
  corruption, not tampering: if you need provenance, verify it before loading.
- **Choose the two bounds.** `MAX_FILE` caps your read, and `MAX_LOAD` caps what a load
  may allocate (`tx_model_alloc_bytes(V, T, C, F)` is the exact figure for a model).
  The load bound matters: rosnet's allocator exits the process when an allocation fails.
- **Token ids must be in [0, V).** `tx_fwd_packed` and `tx_dec_step` fail loud on one that
  is not, so map your tokenizer's ids onto the model's vocabulary before you serve.
- **A loaded model only serves.** It has no latent weights, so `tx_fwd`, `tx_fwd_q`,
  `tx_bwd`, `tx_sgd` and `tx_pack` refuse it.
- **Portability.** A model file written on x86_64 loads on aarch64 and the reverse (the
  format is little-endian on both). Serving is bit-identical within a target. Across
  targets the f64 parts of the forward (`f64_exp` in attention and GELU) are promised to
  within 1 ulp, not bit for bit.

The format, every check and every status code are specified in
[`../architecture/001-packed-model-format.md`](../architecture/001-packed-model-format.md);
the full API is [`../api.md`](../api.md).

## 4. Serve from a long-running process

What hoosh 2.11.0 had to handle to serve a tentib model in a gateway that stays up, answers
several callers at once and reloads its config (its `src/lib/tentib_local.cyr` and ADR 014
are the worked example). None of it is new behavior; all of it follows from the contract
above and [`../api.md`](../api.md).

- **One model per process, one caller at a time.** Every tentib family is module-global
  (`tx_*`, `blk_*`, `ki_*`, `pk_*`), so a process holds one model, and no call is
  reentrant. Serialize every call into tentib, loads included, under one lock. Hold it for
  one forward, not a whole reply, so callers interleave, and read what you need from
  `ki_logits` before you release it: the next forward overwrites it. Freeable model
  handles, and with them more than one model, are the 2.0.0 backlog's first item.
- **A bad argument ends the process.** Bad *file data* is a status code, but a bad *call*
  fails loud: `guard()` exits the process on a token id outside [0, V), a T outside
  [1, the model's T], or a forward before a load. A server cannot catch that, so it has to
  make those calls impossible: map every id into [0, V), keep its window at most the T
  that `tx_pack_dims` reports, and call nothing before a load succeeds.
- **Check the vocabulary before the load.** A file declares the byte vocabulary or
  nothing (1.4.0). `tx_pack_peek` reports it with V, T, C, F, and changes nothing, so a
  model you cannot serve never replaces the one serving. hoosh serves bytes as tokens
  (V = 256, token id = byte value). Through 1.3.1 it read the V header word (offset 24)
  by hand, because there was no peek. A file that declares nothing means whatever its
  producer says, and a model trained on another tokenizer has to ship that tokenizer
  beside the file.
- **Decode with the cache, up to T.** `tx_dec_step` makes a new token one position of
  work instead of a forward over the whole window: at the bench config 73.5 µs against
  1.14 ms per token over a 32-token window, 15.5× (`docs/benchmarks.md`). Feed the prompt
  with steps too. That costs the same as one full forward over it (2.35 ms for 32 steps
  against 2.33 ms for one forward). Positions are absolute,
  so the cache holds at most T positions. Past T, serve each token with `tx_fwd_packed`
  over the last T, as a sliding window always cost, or reset and re-feed a shorter tail
  (cheaper, but a different context). Before 1.4.0 every token was a full forward; hoosh
  measured about 50 µs per token that way at C = 16, T ≤ 25.
- **A same-shaped reload keeps no memory; a change of shape still does.** A load over a
  *loaded* model of the same V, T, C, F overwrites its buffers and allocates nothing
  (1.4.0, [ADR 0003](../adr/0003-a-load-reuses-only-a-loaded-models-buffers.md)). Any
  other load allocates `tx_model_alloc_bytes(V, T, C, F)`, and the model it replaces is not
  freed (rosnet's allocator never frees; see the api's *Allocation & lifetime*). So budget
  one model's memory per change of shape. Skipping a load of unchanged bytes (hoosh
  compares SHA-256 digests) still saves the validation and copy, a few milliseconds at
  the bench config. A refused load allocates nothing. Read the file into a buffer you
  reuse, or that buffer leaks per reload as well.
- **A load swaps the model under its readers.** It replaces the model in place, so a reply
  begun on the old model must not go on with the new one. hoosh keeps an epoch it bumps
  under the lock at each load, and a decode step that finds a different epoch ends its
  reply (a buffered one is retried, a stream ends with an error). tentib enforces the
  same for its cache: a `tx_dec_step` after a load, with no `tx_dec_reset` between, fails
  loud, so call `tx_dec_reset` at the start of every reply.
