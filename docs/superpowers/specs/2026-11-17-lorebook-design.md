# Lorebooks — design (DRAFT, not yet approved)

Status: approved 2026-11-18 ("Go!"). Open questions resolved below.

## Why

Long-running worlds lose specifics. "Story so far" keeps the plot, but drops details like a minor
NPC's name or what a monster is like. A lorebook holds those details and sends only the relevant
ones with each reply. It is also a real RAG project (hybrid search, branch-aware retrieval).

## Goals / non-goals

- Goal: relevant world facts reach the narrator when needed, with a small, capped context cost.
- Goal: lore updates itself as the story goes, per chat.
- Goal: import lorebooks from SillyTavern / Character Card files.
- Non-goal (for now): the narrator looking up lore through a tool call, or searching past chat messages
  (RAG over messages). Both come later and reuse the step 2 pieces.
- Anti-goal: context bloat. Lore must never pile up in history or be sent whole.

## Decided

**Entry fields (4):** name, keys (comma-separated, like categories), text, enabled.
- No "always-on" field: always-present world basics belong in the scenario's **system prompt**
  (stable, cached cheaply).

**Where lore lives**
- **Scenario lorebook** = the starting template (hand-written or imported).
- On chat start, the scenario's entries are **copied into the chat's own storage** (same pattern
  as `Conversation.system_prompt`). Later scenario edits don't affect running chats.
- Chats without a scenario start with an empty lorebook and fill up through updates.

**Import:** SillyTavern World Info JSON, Character Card V2/V3 `character_book`, and our own export.
Only the 4 fields are kept. The import message and README say other settings are ignored for now
(secondary keys, insert position, probability, always-on → becomes a normal entry, etc.).
Not yet: PNG character cards, NovelAI `.lorebook`.

**Sending lore (keeps cache and context lean)**
- Matched entries go into the **newest user turn only**, wrapped in labeled tags
  ("world notes for the narrator, not written by the user"). They are never saved into history.
- Cache cost: only the previous user message is re-read uncached each turn (it lost its lore).
  System prompt + history stay cached. Lore in the system prompt would break the whole cache;
  lore saved with each message would bloat the context.
- Cap per reply in `config.yaml` (e.g. max 4 entries / ~800 tokens) plus a minimum score:
  nothing relevant → zero extra tokens.
- Each reply shows a small "Lore used: …" line (store the entry names on the message), so you can
  tune the cap and score and see what each reply cost.
- Check after shipping: the cached-token share per reply should stay about the same.

**Search: two steps**
- **Step 1, FTS5 / BM25** (built into SQLite, already checked: SQLite 3.53.1 has FTS5).
  - Weight the columns: `bm25(lore_fts, 10.0, 5.0, 1.0)` = name, keys, text.
  - Query = words from the last few messages + the new user message, each word quoted
    (FTS5 syntax safety), joined with OR (FTS5 defaults to AND).
  - `porter` tokenizer for stemming.
- **Step 2, add vectors**: fastembed (`BAAI/bge-small-en-v1.5`, 384 dims, CPU/ONNX, no PyTorch) +
  sqlite-vec. Both resolve for Python 3.14 / Windows (fastembed 0.9.1 ≈ 30 packages, sqlite-vec 0.1.9).
  - Embed an entry when it's saved.
  - Merge the BM25 and vector rankings with reciprocal rank fusion: score = Σ 1/(60 + rank).
  - A small before/after test: does adding vectors beat BM25 alone on a sample world?
- Alembic can't autogenerate virtual tables (FTS5 / vec0): write those migrations by hand and exclude
  the virtual tables from autogenerate. Load the sqlite-vec extension on every connection via
  a SQLAlchemy connect event.

**Self-updating**
- Every N replies on a branch (`lore_update: every_replies: N` in `config.yaml`), after a reply
  finishes, the chat's model gets the current entries plus the messages since the last update.
  It returns JSON: `add` or `update` entries. It never deletes; only the user deletes.
- Applied automatically. The chat shows a note: "Lorebook updated: + X, ~ Y", which you can
  expand, edit or **undo**. The chat shows a status line while the update runs.
- Storage: only changed entries are saved, each version tied to the message where it changed.
  No full copies per update.

**Also:** a "Draft from prompt" button on a scenario: the model pulls starter entries out of the
system prompt for the user to edit.

## Resolved

1. **Lore follows branches** (Claude's call; user said "Go!" without picking). Per-entry versions
   tied to messages, like "Story so far". A branch uses the newest version of each entry on its own
   path. Branches that split earlier don't see later changes, forks keep their lore, and undo =
   delete that version. Search filters to the branch's versions.
2. "Every X" = every X **replies** within a chat (user confirmed).

## Left to the builder

Exact table and column names, tag wording, default cap and score values, UI layout of the
lorebook section (scenario editor + a chat drawer like the system-prompt drawer).
