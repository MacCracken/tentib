# Architecture Decision Records

Decisions about tentib — what we chose, the context, and the consequences we accept. Use these when a future reader would reasonably ask *"why did we do it this way?"*

## Conventions

- **Filename**: `NNNN-kebab-case-title.md`, zero-padded to four digits. Never renumber.
- **One decision per ADR.** If a decision supersedes a prior one, add a new ADR and set the old one's status to `Superseded by NNNN`.
- **Status lifecycle**: `Proposed` → `Accepted` → (optionally) `Superseded` or `Deprecated`.
- Use [`template.md`](template.md) as the starting point.

## ADR vs. architecture note vs. guide

| Kind | Lives in | Answers |
|---|---|---|
| ADR | `docs/adr/` | *Why did we choose X over Y?* |
| Architecture note | `docs/architecture/` | *What non-obvious constraint is true about the code?* |
| Guide | `docs/guides/` | *How do I do X?* |

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-tentib-scope.md) | tentib scope, the integer-native thesis, and the name | Accepted |
| [0002](0002-hardening-preserves-bits.md) | Hardening and optimization preserve bits; guards sit at call entry, never in kernel loops | Accepted |
| [0003](0003-a-load-reuses-only-a-loaded-models-buffers.md) | A load reuses only a loaded model's buffers | Accepted |
| [0004](0004-serving-only-load-is-a-new-entry-point.md) | A serving-only load is a new entry point, not a change to `tx_pack_deserialize` | Accepted |
