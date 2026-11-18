# Lorebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-chat, branch-aware, self-updating lorebooks whose relevant entries are found by hybrid
search (FTS5/BM25 + vectors) and sent with the newest user turn only.

**Architecture:** A scenario holds a template lorebook (`ScenarioLore`). Starting a chat copies it
into `LoreEntry` + `LoreVersion` rows; a version with `message_id = None` applies to every branch,
one with a `message_id` was made by a self-update after that reply and applies only below it. Before
each reply, `lore.find_for_reply` picks the branch's current versions, ranks them with FTS5 (step 1)
fused with sqlite-vec cosine distance (step 2), and `chat.build_messages` prepends the winners to the
newest user message. Every N replies the chat's model proposes add/update changes (step 3).

**Tech Stack:** FastHTML + htmx, SQLAlchemy 2 + Alembic on SQLite 3.53 (FTS5 built in), fastembed
(`BAAI/bge-small-en-v1.5`), sqlite-vec, pytest + respx.

**Spec:** `docs/superpowers/specs/2026-11-17-lorebook-design.md`

## Global Constraints

- Simple, junior-readable code: plain functions, explicit names, short "why" comments, no speculative abstractions.
- Entry fields are exactly: name, keys (comma-separated string), text, enabled. No always-on field.
- Lore is never saved into message history and never put in the system prompt; only the newest user turn carries it.
- Caps come from `config.yaml` → `lorebook:` (defaults: `scan_messages: 4`, `max_entries: 4`, `max_tokens: 800`, `vectors: true`, `min_similarity: 0.6`, `update_every_replies: 10`; `0` = never update).
- FTS ranking: `bm25(lore_fts, 10.0, 5.0, 1.0)` over columns (name, keys, text); query words quoted and joined with `OR`; tokenizer `porter unicode61 remove_diacritics 2`.
- Rank fusion: reciprocal rank fusion, score = Σ 1/(60 + rank), rank starting at 1.
- Self-updates add or update entries; they never delete. Only the user deletes.
- Import keeps only the 4 fields and always tells the user: "Only name, keys, text and on/off are used for now. Other lorebook settings (secondary keys, always-on, position, probability, …) were ignored."
- Token estimate is `len(text) // 4` (no tokenizer dependency).
- Tests never download the embedding model or call Z.ai: fake `embeddings.embed_passages` / `embeddings.embed_query` and `zai.complete`.
- Every Alembic migration that touches `lore_fts` is written by hand; autogenerate must ignore `lore_fts*` tables.

## Review Focus

1. Story text containing FTS5 syntax (`"`, `*`, `NOT`, `AND`, `-`, parentheses), emoji, accents or only short words: search returns results or nothing, never raises. → test in Task 2.
2. Switching branch, regenerating, editing a user message, or forking: a reply never sees a version made on another branch, and a fork keeps exactly its branch's versions. → tests in Tasks 2 and 10.
3. Deleting a chat or a scenario leaves no rows in `lore_fts`, `lore_versions` or `lore_entries` (cascade must fire the FTS triggers). → test in Task 1.
4. The model answers a lorebook update with prose, fenced JSON, or broken JSON; or the call fails: the reply still streams, nothing is half-applied, and the update is retried on a later reply. → tests in Task 10.
5. Importing a PNG character card, a NovelAI file, an empty book, or a 6 MB file: a clear error names the file, nothing is imported, other files in the same upload still import. → test in Task 4.

---

## Step 1: lorebook + BM25

### Task 1: Tables, FTS index and settings

**Files:**
- Modify: `zchat/models.py`, `zchat/config.py`, `config.yaml`, `migrations/env.py`
- Create: `migrations/versions/<rev>_add_lorebooks.py` (hand-written; `down_revision` = output of `uv run alembic heads`)
- Test: `tests/test_lore.py`

**Interfaces:**
- Produces (models.py):
  - `ScenarioLore` (`scenario_lore`): `id`, `scenario_id` (FK scenarios, `ondelete=CASCADE`, index), `name: String(200)`, `keys: String(500) = ""`, `text: Text = ""`, `enabled: Boolean = True`, `created_at`. `Scenario.lore` relationship, `cascade="all, delete-orphan"`, `order_by=ScenarioLore.id`.
  - `LoreEntry` (`lore_entries`): `id`, `conversation_id` (FK conversations, CASCADE, index); `versions` relationship (cascade all, delete-orphan, passive_deletes). `Conversation.lore_entries` relationship (cascade all, delete-orphan, passive_deletes).
  - `LoreVersion` (`lore_versions`): `id`, `entry_id` (FK lore_entries, CASCADE, index), `message_id: int | None` (FK messages, CASCADE, index; docstring: None = applies to every branch; set = made by the self-update after that reply), `name`, `keys`, `text`, `enabled` (same types as ScenarioLore), `created_at`. `LoreVersion.entry` relationship.
  - `Message.lore_used: Text = ""` (newline-separated entry names sent with this reply), `Message.lore_note: Text = ""` (set when a lorebook update ran right after this reply; see Task 10).
- Produces (config.py): `LORE_SCAN_MESSAGES`, `LORE_MAX_ENTRIES`, `LORE_MAX_TOKENS`, `LORE_VECTORS`, `LORE_MIN_SIMILARITY`, `LORE_UPDATE_EVERY`, read from `_settings.get("lorebook")` with the defaults in Global Constraints (same style as `_summary`).
- Produces (DB): virtual table `lore_fts(name, keys, text)`, `rowid` = `lore_versions.id`, kept in sync by triggers `lore_versions_ai` (after insert), `lore_versions_ad` (after delete), `lore_versions_au` (after update of name, keys, text: delete + insert).

- [ ] **Step 1: Write the failing tests** in `tests/test_lore.py`

```python
def fts_rowids(s):
    return [r[0] for r in s.execute(text("SELECT rowid FROM lore_fts ORDER BY rowid"))]

def test_fts_index_follows_lore_versions(s):
    conv = new_conv(s)                       # helper: Conversation + one exchange, like tests/test_chat.py
    entry = LoreEntry(conversation_id=conv.id); s.add(entry); s.flush()
    v = LoreVersion(entry_id=entry.id, name="Mire Hag", keys="hag, witch", text="Lures travelers with lantern light.")
    s.add(v); s.flush()
    assert fts_rowids(s) == [v.id]
    v.text = "Drowned in the marsh."; s.flush()
    assert s.execute(text("SELECT rowid FROM lore_fts WHERE lore_fts MATCH 'drowned'")).scalar() == v.id

def test_deleting_a_chat_or_scenario_leaves_no_lore_behind(s):
    # chat: entry + version, then s.delete(conv); flush
    assert fts_rowids(s) == [] and s.scalar(select(func.count(LoreVersion.id))) == 0
    assert s.scalar(select(func.count(LoreEntry.id))) == 0
    # scenario: Scenario with two ScenarioLore rows, s.delete(scenario)
    assert s.scalar(select(func.count(ScenarioLore.id))) == 0

def test_lorebook_settings_have_defaults():
    assert (config.LORE_SCAN_MESSAGES, config.LORE_MAX_ENTRIES, config.LORE_MAX_TOKENS) == (4, 4, 800)
    assert config.LORE_UPDATE_EVERY == 10
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_lore.py -v` — Expected: FAIL (ImportError on `LoreEntry`).

- [ ] **Step 3: Add the models, settings and `lorebook:` section in `config.yaml`** (with comments in the file's existing style).

- [ ] **Step 4: Write the migration by hand.** Create the three tables and two message columns with `op.create_table` / `batch_alter_table` (same column options as the models, `server_default=''` on the new message columns), then `op.execute` the FTS table and the three triggers:

```sql
CREATE VIRTUAL TABLE lore_fts USING fts5(name, keys, text, tokenize='porter unicode61 remove_diacritics 2');
CREATE TRIGGER lore_versions_ai AFTER INSERT ON lore_versions BEGIN
  INSERT INTO lore_fts(rowid, name, keys, text) VALUES (new.id, new.name, new.keys, new.text); END;
CREATE TRIGGER lore_versions_ad AFTER DELETE ON lore_versions BEGIN
  DELETE FROM lore_fts WHERE rowid = old.id; END;
CREATE TRIGGER lore_versions_au AFTER UPDATE OF name, keys, text ON lore_versions BEGIN
  DELETE FROM lore_fts WHERE rowid = old.id;
  INSERT INTO lore_fts(rowid, name, keys, text) VALUES (new.id, new.name, new.keys, new.text); END;
```

`downgrade()` drops triggers, `lore_fts`, the tables and the columns.

- [ ] **Step 5: Make autogenerate ignore the FTS tables.** In `migrations/env.py`, add `include_name(name, type_, parent_names)` returning `False` for table names starting with `lore_fts`, and pass `include_name=include_name` to both `context.configure` calls. Check: `uv run alembic check` → "No new upgrade operations detected."

- [ ] **Step 6: Run the tests** — `uv run pytest -q` → all pass (62 old + new).

- [ ] **Step 7: Commit** — `git commit -m "feat(lore): lorebook tables, FTS5 index and settings"`

### Task 2: Branch-aware lookup and BM25 search

**Files:**
- Create: `zchat/lore.py`
- Modify: `zchat/chat.py` (`fork`)
- Test: `tests/test_lore.py`

**Interfaces:**
- Consumes: models from Task 1, `config.LORE_*`.
- Produces (`zchat/lore.py`; must not import `zchat.chat`, because chat imports lore):
  - `split_keys(keys: str) -> list[str]` — `"hag, witch"` → `["hag", "witch"]`.
  - `seed_chat(s: Session, conv: Conversation, scenario: Scenario) -> None` — copies every `ScenarioLore` row into a new `LoreEntry` + `LoreVersion(message_id=None)`.
  - `current_versions(s: Session, conv_id: int, path: list[Message]) -> list[LoreVersion]` — the version in effect for each entry on this branch (disabled ones included), sorted by name, case-insensitively.
  - `query_words(text: str) -> list[str]`
  - `search_fts(s: Session, version_ids: list[int], words: list[str]) -> list[int]` — version ids, best first, at most 20.
  - `find_for_reply(s: Session, conv: Conversation, path: list[Message]) -> list[LoreVersion]` — what to send with the next reply.
  - `lore_block(versions: list[LoreVersion]) -> str` — `""` for an empty list.
  - `copy_to_fork(s: Session, old_conv_id: int, new_conv_id: int, message_ids: dict[int, int]) -> None` — `message_ids` maps old message id → new.

- [ ] **Step 1: Write the failing tests**

```python
def test_current_versions_follow_the_branch(s):
    conv, path = story_with_lore(s)   # helper: 3 exchanges; entry "Mire Hag" seeded (message_id None)
    # self-update after reply 2 (path[3]) changed her
    s.add(LoreVersion(entry_id=hag.id, message_id=path[3].id, name="Mire Hag", text="Dead.")); s.flush()
    assert [v.text for v in lore.current_versions(s, conv.id, chat.active_path(s, conv))] == ["Dead."]
    chat.edit_user_message(s, path[2], "user 2, take two")      # branch splits before the update
    assert [v.text for v in lore.current_versions(s, conv.id, chat.active_path(s, conv))] == ["Lures travelers."]

def test_fork_keeps_only_its_branch_versions(s):
    # versions on path[3] and on a sibling branch; fork from path[5]
    forked = chat.fork(s, conv, path[5])
    texts = [v.text for v in lore.current_versions(s, forked.id, chat.active_path(s, forked))]
    assert texts == ["Dead."]   # the sibling branch's version was not copied

@pytest.mark.parametrize("story", ['She says "NOT" (AND) -run* now', "🙂 é à", "a an of", ""])
def test_search_never_raises_on_story_text(s, story):
    ids = [v.id for v in lore.current_versions(s, conv.id, path)]
    lore.search_fts(s, ids, lore.query_words(story))   # no exception; [] is fine

def test_query_words():
    assert lore.query_words('The Mire-Hag\'s "lantern", the lantern!') == ["mire", "hag", "lantern"]

def test_name_and_keys_outrank_body_text(s):
    # entry A has "lantern" only in text; entry B has "lantern" in keys
    assert lore.search_fts(s, [a.id, b.id], ["lantern"]) == [b.id, a.id]

def test_find_for_reply_respects_caps_and_disabled(s, monkeypatch):
    monkeypatch.setattr(config, "LORE_MAX_ENTRIES", 2)
    # 3 matching entries, 1 matching but disabled, 1 matching with text longer than LORE_MAX_TOKENS*4
    names = [v.name for v in lore.find_for_reply(s, conv, path)]
    assert len(names) == 2 and "Disabled One" not in names and "Huge One" not in names

def test_find_for_reply_only_scans_recent_messages(s, monkeypatch):
    monkeypatch.setattr(config, "LORE_SCAN_MESSAGES", 2)
    # "Gloomfang" mentioned only in path[0] → not found; mentioned in path[-1] → found

def test_lore_block_wording():
    block = lore.lore_block([LoreVersion(name="Mire Hag", keys="", text="Swamp witch.")])
    assert block.startswith("<world_notes>\n") and "### Mire Hag\nSwamp witch." in block
    assert block.endswith("</world_notes>") and lore.lore_block([]) == ""
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_lore.py -v` → FAIL (no module `zchat.lore`).

- [ ] **Step 3: Implement `zchat/lore.py`.** Decisions the tests don't pin:
  - `current_versions`: `position = {m.id: i for i, m in enumerate(path)}`; load all versions of the chat's entries; a version counts if `message_id is None` (position −1) or `message_id in position`; per entry keep the one with the highest position, ties → highest `id`.
  - `query_words`: `re.findall(r"\w+", text.lower())`, drop words shorter than 3 characters and a small `STOPWORDS` set (~40 common English words: the, and, you, are, was, with, that, this, for, but, not, …), dedupe keeping order, cap at 64 words. Only `\w` characters survive, so FTS syntax can't get through.
  - `search_fts`: return `[]` if either list is empty. Otherwise query `SELECT rowid FROM lore_fts WHERE lore_fts MATCH :q AND rowid IN :ids ORDER BY bm25(lore_fts, 10.0, 5.0, 1.0) LIMIT 20`, with `:q` = `" OR ".join(f'"{w}"' for w in words)` and `ids` an expanding bindparam.
  - `find_for_reply`: enabled current versions → words from the `content` of the last `LORE_SCAN_MESSAGES` messages of `path` (the path ends with the new user message) → `search_fts` → walk in rank order, skipping entries whose `len(name + text) // 4` would push the total over `LORE_MAX_TOKENS`, stopping at `LORE_MAX_ENTRIES`.
  - `lore_block` copy:

```text
<world_notes>
Background from the lorebook, for the narrator. Not written by the user: use what fits the scene and never mention these notes.

### {name}
{text}

### {name}
{text}
</world_notes>
```

- [ ] **Step 4: In `chat.fork`, record `message_ids[m.id] = c.id` in the copy loop, copy `lore_used` and `lore_note` onto each copied message, and call `lore.copy_to_fork(s, conv.id, new.id, message_ids)` at the end.** `copy_to_fork` copies a version when its `message_id` is None or in the map, with the mapped id; it skips entries left with no versions.

- [ ] **Step 5: Run the tests** — `uv run pytest -q` → all pass.

- [ ] **Step 6: Commit** — `git commit -m "feat(lore): branch-aware lookup and BM25 search"`

### Task 3: Send lore with the newest user turn

**Files:**
- Modify: `zchat/chat.py` (`build_messages`), `zchat/generate.py` (`run`), `zchat/ui.py` (`message_view`), `static/app.css`
- Test: `tests/test_lore.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `lore.find_for_reply`, `lore.lore_block`.
- Produces: `build_messages(conv, path, model_id, lore: str = "") -> list[dict]` — when `lore` is set, the newest user message's text becomes `lore + "\n\n" + text` (for list content, the text part changes).

- [ ] **Step 1: Write the failing tests**

```python
def test_lore_goes_only_on_the_newest_user_turn(s):
    conv = make_story(s, 3)
    path = chat.active_path(s, conv)[:-1]          # drop the last reply: "user 3" is the newest turn
    msgs = chat.build_messages(conv, path, conv.model, lore="<world_notes>X</world_notes>")
    assert msgs[-1]["content"].startswith("<world_notes>X</world_notes>\n\nuser 3")
    assert all("world_notes" not in m["content"] for m in msgs[:-1])   # system + history untouched

def test_previous_turns_stay_identical_for_the_cache(s):
    # build turn N with lore, then turn N+1: every message before turn N's user message is identical
    assert turn_n[:-1] == turn_n_plus_1[:len(turn_n) - 1]

def test_reply_records_and_shows_lore_used(client):
    # scenario with entry "Mire Hag" (keys "hag"); start chat "I look for the hag."; stream it
    # the fake stream_chat captures messages: the last one contains "### Mire Hag"
    # the finished page shows "Lore: Mire Hag" on the reply
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_lore.py tests/test_app.py -v` → FAIL.

- [ ] **Step 3: Implement.**
  - In `generate.run`'s second session block: `found = lore.find_for_reply(s, conv, path)`, `m.lore_used = "\n".join(v.name for v in found)`, `msgs = build_messages(conv, path, model, lore=lore.lore_block(found))`.
  - In `message_view`, when `m.lore_used` is set, add `Span("Lore: " + ", ".join(names), cls="usage lore-used", title="Lorebook entries sent with this reply")` next to `usage_line(m)`.

- [ ] **Step 4: Run the tests** — `uv run pytest -q` → all pass.

- [ ] **Step 5: Commit** — `git commit -m "feat(lore): send matching lore with the newest user turn"`

### Task 4: Import and export

**Files:**
- Create: `zchat/lore_import.py`
- Modify: `zchat/scenario_json.py`
- Test: `tests/test_lore_import.py`, `tests/test_scenario_json.py`

**Interfaces:**
- Produces:
  - `scenario_json.load_json(raw: bytes) -> object` — the size check, decoding and `json.loads` moved out of `parse` (same errors as today).
  - `lore_import.IMPORT_NOTE: str` — the exact sentence in Global Constraints.
  - `lore_import.parse(raw: bytes) -> list[dict]` — each dict is `{"name": str, "keys": str, "text": str, "enabled": bool}`. Raises `ValueError` with a readable message.
  - `lore_import.entries_from(data: object) -> list[dict]` — same output, from already-parsed JSON (used by scenario import).
  - `lore_import.to_export(name: str, entries: list) -> dict` — `{"name": name, "entries": [{"name", "keys": [..], "content", "enabled"}]}` (character_book shape). Takes objects with `.name/.keys/.text/.enabled`.
  - `scenario_json.to_dict` adds `"lorebook": to_export(...)["entries"]` when the scenario has lore. `scenario_json.parse` returns each scenario with `"lorebook": list[dict]` (empty if none), read through `entries_from`.

- [ ] **Step 1: Write the failing tests**

```python
ST_WORLD = {"entries": {"0": {"uid": 0, "key": ["hag", "witch"], "keysecondary": ["swamp"], "comment": "Mire Hag",
                              "content": "Swamp witch.", "constant": True, "disable": False, "position": 0}}}
CARD_V2 = {"spec": "chara_card_v2", "data": {"name": "Ash", "character_book": {"entries": [
    {"keys": ["wyrm"], "content": "Ash Wyrm sleeps.", "enabled": False, "name": "Ash Wyrm", "insertion_order": 1}]}}}
CARD_V3 = {"spec": "chara_card_v3", "data": {"character_book": {"entries": [{"keys": "gate", "content": "Sealed."}]}}}

def test_sillytavern_world_info():
    assert lore_import.parse(json.dumps(ST_WORLD).encode()) == [
        {"name": "Mire Hag", "keys": "hag, witch", "text": "Swamp witch.", "enabled": True}]

def test_character_cards_v2_and_v3():
    assert lore_import.parse(json.dumps(CARD_V2).encode())[0] == {
        "name": "Ash Wyrm", "keys": "wyrm", "text": "Ash Wyrm sleeps.", "enabled": False}
    assert lore_import.parse(json.dumps(CARD_V3).encode())[0]["name"] == "gate"   # no name → first key

def test_our_export_round_trips():
    data = lore_import.to_export("World", [SimpleNamespace(name="A", keys="x, y", text="T", enabled=True)])
    assert lore_import.parse(json.dumps(data).encode()) == [{"name": "A", "keys": "x, y", "text": "T", "enabled": True}]

def test_entries_without_text_are_skipped_and_names_are_filled_in():
    # [{"content": ""}, {"content": "Only text"}] → one entry named "Entry 2"

@pytest.mark.parametrize("raw, message", [
    (b"\x89PNG\r\n\x1a\n...", "PNG character cards aren't supported yet"),
    (json.dumps({"lorebookVersion": 5, "entries": [{"text": "x"}]}).encode(), "NovelAI lorebooks aren't supported yet"),
    (json.dumps({"spec": "chara_card_v2", "data": {"name": "No book"}}).encode(), "has no lorebook"),
    (json.dumps({"entries": []}).encode(), "no entries"),
    (b"x" * 6_000_000, "too big"),
])
def test_clear_errors(raw, message):
    with pytest.raises(ValueError, match=message):
        lore_import.parse(raw)

def test_scenario_export_and_import_keep_the_lorebook():   # in tests/test_scenario_json.py
    [sc] = scenario_json.parse(json.dumps({"name": "W", "lorebook": [{"name": "A", "keys": ["x"], "content": "T"}]}).encode())
    assert sc["lorebook"] == [{"name": "A", "keys": "x", "text": "T", "enabled": True}]
```

Also update `test_parse_accepts_one_object_or_a_list` so its expected dict includes `"lorebook": []`.

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_lore_import.py tests/test_scenario_json.py -v` → FAIL.

- [ ] **Step 3: Implement.** Format detection in `entries_from`, in this order:
  - dict with `spec` starting `chara_card` → `data.character_book.entries`;
  - dict with `lorebookVersion` → the NovelAI error;
  - dict with `entries`: a dict means SillyTavern (use the values in key order), a list means character_book or ours;
  - a list → entries;
  - anything else → `ValueError("not a lorebook file")`.

  A raw PNG is detected in `parse` from the first 8 bytes, before decoding. Per entry:
  - name = `name` or `comment` or first key or `f"Entry {n}"`;
  - keys = `keys` or `key` (string or list; commas inside a single key become spaces);
  - text = `content` or `text`;
  - enabled = `enabled` is not False and `disable` is not True.

  Clean all fields with `scenario_json.clean` (name 200 chars, one line; keys 500, one line; text 20000).

- [ ] **Step 4: Run the tests** — `uv run pytest -q` → all pass.

- [ ] **Step 5: Commit** — `git commit -m "feat(lore): import SillyTavern / Character Card lorebooks, export ours"`

### Task 5: Scenario lorebook UI

**Files:**
- Create: `zchat/lore_ui.py` (lorebook components; imports `icon`, `markdown` from `zchat.ui`)
- Modify: `zchat/ui.py` (`scenario_editor` renders the section), `main.py` (routes; `/new` seeds; `/scenarios/clone` copies lore; `/scenarios/import` creates lore from `item.pop("lorebook")`), `static/app.css`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `lore.seed_chat`, `lore_import.parse / to_export / IMPORT_NOTE`.
- Produces (routes, all return the `#scenario-lore` list fragment unless noted):
  - `GET /lore/scenario/list?sid=`
  - `GET /lore/scenario/form?sid=&lid=` (lid optional; returns the form)
  - `POST /lore/scenario/save` (`sid`, `lid?`, `name`, `keys`, `text`, `enabled`)
  - `POST /lore/scenario/delete?lid=` (`hx_confirm` on the button)
  - `POST /lore/scenario/import?sid=` (multipart `files`; info toast with the count + `IMPORT_NOTE`, error toast per bad file)
  - `GET /lore/scenario/export?sid=` (JSON download, `<slug>-lorebook.json`)
- Produces (`lore_ui.py`):
  - `lore_list(entries, *, list_id, form_url, delete_url_prefix)` — one list component used for both scenario and chat lore.
  - `lore_form(*, post_url, cancel_url, target, name="", keys="", text="", enabled=True, hidden: dict)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_scenario_lorebook_add_edit_delete_import_export(client):
    sid = make_scenario(client, "Marsh")
    client.post("/lore/scenario/save", data={"sid": sid, "name": "Mire Hag", "keys": "hag", "text": "Swamp witch.", "enabled": "1"})
    page = client.get(f"/scenarios/edit?sid={sid}").text
    assert "Lorebook" in page and "Mire Hag" in page
    r = client.post(f"/lore/scenario/import?sid={sid}", files=[("files", ("w.json", json.dumps(ST_WORLD), "application/json")),
                                                             ("files", ("bad.json", "{", "application/json"))])
    assert "Imported 1 entry" in r.text and "Other lorebook settings" in r.text and "bad.json" in r.text
    exported = json.loads(client.get(f"/lore/scenario/export?sid={sid}").text)
    assert [e["name"] for e in exported["entries"]] == ["Mire Hag", "Mire Hag"]   # imports never overwrite
    # delete one → one left

def test_new_chat_and_clone_copy_the_scenario_lorebook(client):
    # chat started from the scenario has one LoreEntry per ScenarioLore, message_id None
    # clone has its own ScenarioLore copies; editing the scenario afterwards doesn't change the chat
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_app.py -v` → FAIL (404s).

- [ ] **Step 3: Implement the routes and components.** The section sits below the scenario form and only shows for a saved scenario (otherwise: "Save the scenario first to add a lorebook."). Its header holds Add entry, Import (file input that submits on change, like scenario import), and Export. Each list row shows: name, keys as tags, the first 160 characters of the text, a "Disabled" tag when off, and Edit / Delete buttons. An unchecked `enabled` checkbox isn't sent, so treat it as `False`.

- [ ] **Step 4: Run the tests** — `uv run pytest -q` → all pass. Then run the app (`uv run main.py`), add, import and export on a scenario, and check it in the browser in light and dark themes.

- [ ] **Step 5: Commit** — `git commit -m "feat(lore): scenario lorebook editor, import and export"`

### Task 6: Chat lorebook drawer and README

**Files:**
- Modify: `zchat/ui.py` (`chat_page` adds the drawer under `system_drawer`), `zchat/lore_ui.py`, `main.py`, `static/app.css`, `README.md`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `lore.current_versions`, `lore_ui.lore_list / lore_form`.
- Produces (routes, returning the `#chat-lore` list):
  - `GET /lore/chat/list?cid=`
  - `GET /lore/chat/form?cid=&vid=`
  - `POST /lore/chat/save` (`cid`, `vid?`, fields): with `vid`, edit that version in place, the same way "Story so far" edits work; without it, a new `LoreEntry` + version with `message_id=None`, which every branch sees.
  - `POST /lore/chat/delete?vid=`: deletes the version's whole `LoreEntry`, on every branch. The confirm text says so.
- `/switch` additionally returns the drawer list out-of-band, so it shows the new branch's lore.

- [ ] **Step 1: Write the failing tests**

```python
def test_chat_lorebook_drawer(client):
    # chat from a scenario with "Mire Hag": page shows the drawer "Lorebook · 1 entry" and the entry
    # edit text → current version text changed, still message_id None
    # add "Gloomfang" → 2 entries; delete "Mire Hag" → LoreEntry and its versions gone
    # /switch to another branch returns 'id="chat-lore"' with hx-swap-oob
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_app.py -v` → FAIL.

- [ ] **Step 3: Implement.** Drawer summary: `"Lorebook"` + muted `" · N entries · only matching entries are sent with each reply"`.

- [ ] **Step 4: README.** Add a **Lorebooks** bullet: scenario template copied into each chat; only matching entries (up to the caps) go with the newest message; the "Lore:" line; import formats plus the `IMPORT_NOTE` sentence; the `lorebook:` settings.

- [ ] **Step 5: Run the tests** — `uv run pytest -q` → all pass. Check the drawer in the browser.

- [ ] **Step 6: Commit** — `git commit -m "feat(lore): chat lorebook drawer, README"`

## Step 2: vectors

### Task 7: Embeddings and sqlite-vec

**Files:**
- Modify: `pyproject.toml` (`uv add fastembed sqlite-vec`), `zchat/db.py`, `zchat/models.py`, `main.py` (startup warm-up)
- Create: `zchat/embeddings.py`, `migrations/versions/<rev>_add_lore_embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Produces:
  - `LoreVersion.embedding: LargeBinary | None` (float32 bytes, 384 values).
  - `embeddings.MODEL_NAME = "BAAI/bge-small-en-v1.5"`, `embeddings.DIMENSIONS = 384`.
  - `embeddings.embed_passages(texts: list[str]) -> list[bytes]` — fastembed `TextEmbedding.passage_embed`.
  - `embeddings.embed_query(text: str) -> bytes` — `query_embed`, which adds bge's query instruction.
  - `embeddings.passage_text(name: str, keys: str, text: str) -> str` — `f"{name}. {keys}. {text}"`.
  - `embeddings.warm_up() -> None` — loads the model, downloading it once into `config.DATA_DIR / "models"` (fastembed `cache_dir`).
- The model loads lazily into a module global, so tests that fake the two `embed_*` functions never load it.
- `db.py`: the connect listener also runs `enable_load_extension(True)`, `sqlite_vec.load(conn)`, `enable_load_extension(False)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sqlite_vec_is_loaded(s):
    a = struct.pack("3f", 1, 0, 0); b = struct.pack("3f", 0, 1, 0)
    assert s.execute(text("SELECT vec_distance_cosine(:a, :b)"), {"a": a, "b": b}).scalar() == pytest.approx(1.0)

def test_passage_text():
    assert embeddings.passage_text("Mire Hag", "hag", "Swamp witch.") == "Mire Hag. hag. Swamp witch."
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_embeddings.py -v` → FAIL.

- [ ] **Step 3: Add the dependencies, the column, the migration (autogenerate is fine for this column), the extension loading, and `embeddings.py`.** In `main.py`'s startup, when `config.LORE_VECTORS` is on, run `warm_up` in a background thread (`asyncio.to_thread`) so the first reply doesn't wait on the download. Failures are only logged.

- [ ] **Step 4: Run the tests** — `uv run pytest -q` → all pass. Then check by hand that the real model loads and embeds: `uv run python -c "from zchat import embeddings; print(len(embeddings.embed_query('swamp witch')))"` → `1536` (384 × 4 bytes).

- [ ] **Step 5: Commit** — `git commit -m "feat(lore): local embeddings (fastembed) and sqlite-vec"`

### Task 8: Hybrid search

**Files:**
- Modify: `zchat/lore.py`, `main.py` (`/lore/chat/save` clears `embedding` when name/keys/text change)
- Test: `tests/test_lore.py`

**Interfaces:**
- Produces:
  - `lore.ensure_embeddings(s: Session, versions: list[LoreVersion]) -> None` — embeds the ones with `embedding is None`, in one call.
  - `lore.search_vectors(s: Session, version_ids: list[int], query_text: str) -> list[int]` — best first, at most 20, only those with cosine similarity ≥ `LORE_MIN_SIMILARITY` (that is, `vec_distance_cosine` ≤ 1 − min_similarity).
  - `lore.fuse(rankings: list[list[int]]) -> list[int]` — reciprocal rank fusion, k = 60. Ties keep first-seen order.
- `find_for_reply` ranks by `fuse([search_fts(...), search_vectors(...)])` when `config.LORE_VECTORS` is on. If embedding fails for any reason, it logs and uses FTS only.

- [ ] **Step 1: Write the failing tests** (an autouse fixture fakes `embeddings.embed_passages` / `embed_query` with a deterministic bag-of-words vector: hash each word to one of 384 slots, normalize)

```python
def test_fuse():
    assert lore.fuse([[1, 2, 3], [3, 1]]) == [1, 3, 2]

def test_vectors_find_an_entry_with_no_shared_words(s, fake_embeddings):
    # fake maps "bog" and "swamp" to the same slot; entry "Mire Hag": "swamp witch"; story: "Something rises from the bog."
    assert [v.name for v in lore.find_for_reply(s, conv, path)] == ["Mire Hag"]

def test_falls_back_to_fts_when_embedding_fails(s, monkeypatch):
    monkeypatch.setattr(embeddings, "embed_query", boom)          # raises RuntimeError
    assert [v.name for v in lore.find_for_reply(s, conv, path)] == ["Mire Hag"]   # found by its name

def test_editing_an_entry_re_embeds_it(client, fake_embeddings):
    # save via /lore/chat/save with new text → embedding is None until the next find_for_reply fills it
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_lore.py -v` → FAIL.

- [ ] **Step 3: Implement.** `search_vectors`: `SELECT id FROM lore_versions WHERE id IN :ids AND embedding IS NOT NULL AND vec_distance_cosine(embedding, :q) <= :max_d ORDER BY vec_distance_cosine(embedding, :q) LIMIT 20`. The query text is the same recent-message text `find_for_reply` builds for the FTS words, cut to its last 2000 characters.

- [ ] **Step 4: Run the tests** — `uv run pytest -q` → all pass.

- [ ] **Step 5: Commit** — `git commit -m "feat(lore): hybrid BM25 + vector search with rank fusion"`

### Task 9: Before/after evaluation

**Files:**
- Create: `scripts/lore_eval.py`, `scripts/lore_eval_world.json`

**Interfaces:**
- Consumes: `lore.query_words`, `lore.fuse`, `embeddings` (the real model), and an in-memory SQLite DB with the same FTS table. It doesn't touch `data/zchat.db`.

- [ ] **Step 1: Write the sample world:** ~15 entries (monsters, places, NPCs, items) and ~20 scenes, each `{"scene": "...", "expected": "Entry name"}`. At least 8 of the scenes must not use the entry's name or keys (e.g. "something rises from the bog" → Mire Hag).

- [ ] **Step 2: Write the script.** For each scene, rank with BM25 only, with vectors only, and with both fused. Print hit@1 and hit@4 per method, plus the scenes each method missed. Run: `uv run python scripts/lore_eval.py`.

- [ ] **Step 3: Run it and record the numbers** in a short "Results" section at the top of the script's docstring. If hybrid isn't better than BM25 at hit@4, tune `min_similarity` (try 0.5, 0.6, 0.7), write down what you tried, and update the `config.yaml` default.

- [ ] **Step 4: Commit** — `git commit -m "test(lore): before/after retrieval evaluation"`

## Step 3: self-updating

### Task 10: Lorebook updates every N replies

**Files:**
- Create: `zchat/lore_update.py`
- Modify: `zchat/generate.py` (`run`), `main.py` (`/stream` handles the `lore` event; `render_messages` shows notes; `POST /lore/undo`), `zchat/lore_ui.py` (`lore_note_view`), `static/app.js`, `static/app.css`
- Test: `tests/test_lore_update.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `lore.current_versions`, `story.transcript`, `zai.complete`, `scenario_json.clean`.
- Produces:
  - `lore_update.update_due(path: list[Message]) -> bool` — `LORE_UPDATE_EVERY > 0`, and the count of `done` assistant replies after the last message on the path with a non-empty `lore_note` (or since the start) is ≥ `LORE_UPDATE_EVERY`.
  - `lore_update.parse_changes(answer: str) -> list[dict]` — `[{"name", "keys": str, "text"}]`, at most 8, `[]` for anything unreadable.
  - `async lore_update.update_lore(leaf_id: int) -> int | None` — runs the update for the branch ending at `leaf_id` and returns the id of the reply the note was saved on (the newest `done` assistant message). Returns `None` when there's nothing to do; raises on model or network errors.
  - `lore_ui.lore_note_view(m: Message, versions: list[LoreVersion])` — `Details` with id `lore-note-{m.id}`, showing "Lorebook updated: + A, ~ B" (or "Lorebook checked: no changes"), the changed entries, and an Undo button.
  - `POST /lore/undo?mid=` — deletes the versions with `message_id=mid`, deletes entries left with no versions, sets `lore_note = "Undone."` (so the reply counter doesn't fire again right away), and returns `#messages`.
  - SSE event `lore`: `{"mid": int, "html": str}`. `app.js` inserts it after `#msg-{mid}`, the same way it handles `summary`.

- [ ] **Step 1: Write the failing tests**

```python
def test_update_due_counts_replies_since_the_last_update(s, monkeypatch):
    monkeypatch.setattr(config, "LORE_UPDATE_EVERY", 2)
    # 1 reply → False; 2 replies → True; lore_note on reply 2, then 1 more reply → False

@pytest.mark.parametrize("answer, names", [
    ('{"changes": [{"name": "Gloomfang", "keys": ["wolf"], "text": "Wounded."}]}', ["Gloomfang"]),
    ('```json\n{"changes": [{"name": "A", "text": "x"}]}\n```', ["A"]),
    ("Sure! Here are the changes: none really.", []),
    ('{"changes": [{"name": "", "text": "x"}, {"name": "B"}]}', []),          # missing name or text
])
def test_parse_changes(answer, names):
    assert [c["name"] for c in lore_update.parse_changes(answer)] == names

async def test_update_adds_and_updates_on_this_branch_only(monkeypatch):
    # entry "Gloomfang" exists; fake complete → update Gloomfang, add "Lantern"
    # new versions have message_id = newest reply; note == "Added: Lantern\nUpdated: Gloomfang"
    # a branch split before that reply still sees the old Gloomfang and no Lantern

async def test_failed_or_unreadable_update_changes_nothing(monkeypatch):
    # complete raises → update_lore raises, no versions, no lore_note (so it retries next reply)
    # complete returns prose → no versions, lore_note == "No changes."

def test_stream_runs_due_update_before_the_reply_and_undo_reverts_it(client, monkeypatch):
    # LORE_UPDATE_EVERY=1; second reply's stream contains "event: lore" and "Updating the lorebook"
    # page shows id="lore-note-<first reply>"; POST /lore/undo → versions gone, note shows "Undone."
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_lore_update.py -v` → FAIL.

- [ ] **Step 3: Implement `lore_update.py`.**
  - Update prompt: system copy stating the job (keep the narrator's lorebook current; add entries for new named characters, places, items or monsters that matter; update entries whose facts changed; never delete; keep names exact; each text under 120 words; reply with JSON only: `{"changes": [{"name", "keys": [...], "text"}]}`, `{"changes": []}` when nothing changed). The user turn contains `<lorebook>` (current enabled entries: name, keys, text) and `<new_messages>` (`story.transcript` of the messages after the last note).
  - Call `zai.complete(..., model=config.usable_model(conv.model), max_tokens=3000, timeout=300)`.
  - Apply the changes: a change whose name matches a current version (case-insensitive) becomes a new version of that entry, keeping its `enabled`; it's skipped if keys and text are unchanged. Any other name becomes a new entry. All new versions get `message_id` = the note's reply id. Note text: `"Added: A, B\nUpdated: C"` (omit an empty line), or `"No changes."`.
  - `parse_changes`: strip code fences, take the text from the first `{` to the last `}`, `json.loads`, catch every error → `[]`.

- [ ] **Step 4: Wire it into `generate.run`**, next to `summarize_first` and in the same style. If `kind != "image"` and `lore_update.update_due(path)`: yield `status` "Updating the lorebook…", then `await lore_update.update_lore(parent_id)`. On an exception, yield `status` "Couldn't update the lorebook (…). Carrying on." and continue. On success, yield `"lore", {"mid": ...}`. It runs after any summary and before the search, so the new lore is used right away. `/stream` renders `lore_note_view` for the `lore` event. `render_messages` appends the note view after each message with a `lore_note`. The undo route returns `render_messages`.

- [ ] **Step 5: Run the tests** — `uv run pytest -q` → all pass. Check in the browser with `update_every_replies: 1`.

- [ ] **Step 6: Commit** — `git commit -m "feat(lore): self-updating lorebook every N replies, with undo"`

### Task 11: Draft a starter lorebook from the system prompt

**Files:**
- Modify: `zchat/lore_update.py`, `main.py`, `zchat/lore_ui.py`, `README.md`
- Test: `tests/test_lore_update.py`, `tests/test_app.py`

**Interfaces:**
- Produces:
  - `async lore_update.draft_from_prompt(system_prompt: str, model: str) -> list[dict]` — same output as `parse_changes`, at most 12 entries.
  - `POST /lore/scenario/draft?sid=` — adds the drafted entries to the scenario's lore, skipping names already present (case-insensitive). Returns the list plus an info toast "Drafted N entries from the system prompt. Edit or delete any you don't want." Shows an error toast when the prompt is empty or the call fails.

- [ ] **Step 1: Write the failing tests**

```python
async def test_draft_from_prompt(monkeypatch):
    # fake complete returns 2 changes → 2 dicts

def test_draft_button_adds_entries_and_skips_existing_names(client, monkeypatch):
    # scenario with "Mire Hag"; draft returns "Mire Hag" + "Gloomfang" → only Gloomfang added; toast says 1
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_lore_update.py tests/test_app.py -v` → FAIL.

- [ ] **Step 3: Implement.** Draft prompt: pull the named characters, places, factions, monsters and items out of the setting, into lorebook entries with the same JSON shape as updates. It uses `config.DEFAULT_MODEL`. The button sits in the lorebook section header, with `hx_disabled_elt` for the wait.

- [ ] **Step 4: README** — add self-updating (`update_every_replies`, the note, undo) and Draft from prompt to the Lorebooks bullet.

- [ ] **Step 5: Run the tests** — `uv run pytest -q` → all pass.

- [ ] **Step 6: Commit** — `git commit -m "feat(lore): draft a starter lorebook from the system prompt"`

## Finish

- [ ] Full suite: `uv run pytest -q` → all pass, 0 failures.
- [ ] `uv run alembic check` → no new operations.
- [ ] Try it in the app: a scenario with a few entries; a long chat with `update_every_replies: 2`; a branch switch; a fork; undo; import a real SillyTavern file.
- [ ] Cache check: on a chat with lore, the cached-token share per reply stays roughly the same as without lore.
