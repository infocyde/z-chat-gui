"""The lorebook keeping itself up to date, and drafting a starter lorebook.

Every `update_every_replies` replies (config.yaml), before the next reply is written, the chat's
model reads the current lorebook and the story since the last update, and answers with changes:
new entries for things that became important, new text for entries whose facts changed. It never
deletes; only the user does.

Changes are saved as new versions tied to the newest reply (see lore.py), so they follow the branch
they happened on, and Undo just deletes those versions. That reply's `lore_note` records what
changed ("Added: X\nUpdated: Y", "No changes." or "Undone."), which is also how the next update
knows where to start counting.
"""

from __future__ import annotations

import json

from . import config, lore, zai
from .chat import path_to
from .db import session_scope
from .models import LoreEntry, LoreVersion, Message
from .scenario_json import clean
from .story import transcript

MAX_CHANGES = 8
MAX_DRAFT_ENTRIES = 12
MAX_ENTRY_CHARS = 2000

NO_CHANGES = "No changes."
UNDONE = "Undone."

UPDATE_SYSTEM = (
    "You keep the lorebook for an interactive story: short notes the narrator is shown when the story "
    "touches them. Read the current lorebook and the new part of the story. Add an entry for each new "
    "named character, place, item, creature or group that will matter later. Update an entry when the "
    "story changed its facts (someone was hurt, moved, changed sides, died, was revealed). Don't repeat "
    "entries that are still correct, and never delete anything. Keep names exactly as written; to update "
    "an entry, use its exact name. Each text is a few plain sentences, under 120 words, stating facts "
    "rather than retelling scenes. Keys are the words that bring the entry up in the story. "
    'Reply with JSON only: {"changes": [{"name": "...", "keys": ["..."], "text": "..."}]}, '
    'or {"changes": []} when nothing needs changing.'
)

UPDATE_REQUEST = """<setting>
{setting}
</setting>

<lorebook>
{entries}
</lorebook>

<new_messages>
{transcript}
</new_messages>

Reply with the JSON changes."""

DRAFT_SYSTEM = (
    "You start the lorebook for an interactive story: short notes the narrator is shown when the story "
    "touches them. From the setting below, make one entry for each named character, place, faction, "
    "creature and item. Each text is a few plain sentences, under 120 words, with only what the setting "
    "says. Keys are the words that bring the entry up in the story. "
    'Reply with JSON only: {"changes": [{"name": "...", "keys": ["..."], "text": "..."}]}'
)


# ---------------------------------------------------------------- when


def last_note_index(path: list[Message]) -> int:
    """Where the last lorebook update ran on this branch, or -1 if it never has."""
    for i in range(len(path) - 1, -1, -1):
        if path[i].lore_note:
            return i
    return -1


def update_due(path: list[Message]) -> bool:
    """True when `update_every_replies` replies have been written since the last update on this branch."""
    if config.LORE_UPDATE_EVERY == 0:
        return False
    since = path[last_note_index(path) + 1:]
    replies = [m for m in since if m.role == "assistant" and m.status == "done"]
    return len(replies) >= config.LORE_UPDATE_EVERY


# ---------------------------------------------------------------- reading the model's answer


def parse_changes(answer: str, limit: int = MAX_CHANGES) -> list[dict]:
    """The model's JSON answer -> [{name, keys, text}]. Anything unreadable counts as no changes."""
    start, end = answer.find("{"), answer.rfind("}")  # skips ```json fences and chatter around the JSON
    if start == -1 or end < start:
        return []
    try:
        data = json.loads(answer[start:end + 1])
    except (json.JSONDecodeError, RecursionError):
        return []
    items = data.get("changes") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    changes = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        name = clean(item.get("name"), 200, one_line=True)
        text = clean(item["text"], MAX_ENTRY_CHARS)
        if not name or not text:
            continue
        raw_keys = item.get("keys") or []
        if isinstance(raw_keys, str):
            raw_keys = raw_keys.split(",")
        keys = [clean(k, one_line=True).replace(",", " ") for k in raw_keys if isinstance(k, str)]
        keys = [" ".join(k.split()) for k in keys if k.strip()]
        changes.append({"name": name, "keys": clean(", ".join(keys), 500, one_line=True), "text": text})
        if len(changes) == limit:
            break
    return changes


# ---------------------------------------------------------------- applying an update


def format_entries(versions: list[LoreVersion]) -> str:
    if not versions:
        return "(empty)"
    return "\n\n".join(f"### {v.name}\nkeys: {v.keys or '(none)'}\n{v.text}" for v in versions)


async def update_lore(leaf_id: int | None) -> int | None:
    """Update the lorebook of the branch ending at `leaf_id` from the story since the last update.

    Returns the id of the reply the changes were saved on (the newest finished reply), or None if
    there's no reply yet. Raises (zai.ZaiError, network errors) if the model call fails; then nothing
    is saved, so the update is tried again after the next reply.
    """
    with session_scope() as s:
        path = [m for m in path_to(s, leaf_id) if m.status not in ("pending", "streaming")]
        replies = [i for i, m in enumerate(path) if m.role == "assistant" and m.status == "done"]
        if not replies:
            return None
        newest = replies[-1]
        conv = path[0].conversation
        current = lore.current_versions(s, conv.id, path)
        request = UPDATE_REQUEST.format(
            setting=conv.system_prompt.strip()[:4000] or "(none)",
            entries=format_entries(current),  # disabled ones too: still true, and it avoids duplicates
            transcript=transcript(path[last_note_index(path) + 1:newest + 1]))
        model = config.usable_model(conv.model)
        note_on, conv_id = path[newest].id, conv.id

    answer = await zai.complete([{"role": "system", "content": UPDATE_SYSTEM},
                                 {"role": "user", "content": request}],
                                model=model, max_tokens=3000, timeout=300)
    changes = parse_changes(answer)

    with session_scope() as s:
        note_message = s.get(Message, note_on)
        path = path_to(s, note_on)
        by_name = {v.name.lower(): v for v in lore.current_versions(s, conv_id, path)}
        added, updated = [], []
        for change in changes:
            old = by_name.get(change["name"].lower())
            if old is None:
                version = LoreVersion(message_id=note_on, name=change["name"], keys=change["keys"],
                                      text=change["text"], enabled=True)
                s.add(LoreEntry(conversation_id=conv_id, versions=[version]))
                by_name[change["name"].lower()] = version
                added.append(change["name"])
            elif (old.keys, old.text) != (change["keys"], change["text"]):
                # A new version of the same entry; it keeps the entry's name and on/off.
                s.add(LoreVersion(entry_id=old.entry_id, message_id=note_on, name=old.name, keys=change["keys"],
                                  text=change["text"], enabled=old.enabled))
                updated.append(old.name)
        lines = []
        if added:
            lines.append("Added: " + ", ".join(added))
        if updated:
            lines.append("Updated: " + ", ".join(updated))
        note_message.lore_note = "\n".join(lines) or NO_CHANGES
    return note_on


def undo_update(s, m: Message) -> None:
    """Remove what the update after `m` changed. Entries it added (with no other versions) go too."""
    entry_ids = {v.entry_id for v in s.query(LoreVersion).filter(LoreVersion.message_id == m.id)}
    s.query(LoreVersion).filter(LoreVersion.message_id == m.id).delete()
    for entry_id in entry_ids:
        if s.query(LoreVersion).filter(LoreVersion.entry_id == entry_id).count() == 0:
            s.delete(s.get(LoreEntry, entry_id))
    m.lore_note = UNDONE  # not "": that would make the next reply run the update again right away
    s.flush()
