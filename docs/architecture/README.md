# Architecture notes

Non-obvious constraints, quirks, and invariants that a reader cannot derive from the code alone. Numbered chronologically — never renumber.

Not decisions (those live in [`../adr/`](../adr/)) and not guides (those live in [`../guides/`](../guides/)). An item here describes *how the world is*, not *what we chose* or *how to do something*.

## Items

- [001 — The packed-model format (version 1)](001-packed-model-format.md): the bytes `tx_pack_serialize` writes and `tx_pack_deserialize` reads, the decoder's checks and status codes, and the invariants (canonical, bit-identical serving, bounded memory).
