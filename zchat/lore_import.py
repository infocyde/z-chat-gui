"""Reading lorebook files from other apps, and writing our own.

Understood formats:
  - SillyTavern World Info: {"entries": {"0": {"key": [...], "comment": "...", "content": "...", ...}}}
  - Character Card V2/V3: {"spec": "chara_card_v2", "data": {"character_book": {"entries": [...]}}}
  - A bare character_book, or a list of its entries: {"entries": [{"keys": [...], "content": "..."}]}
    Our own export (to_export) is this shape too.

Only name, keys, text and on/off are kept (see IMPORT_NOTE). Not supported yet: PNG character cards
(the JSON is hidden inside the image) and NovelAI .lorebook files.
"""

from __future__ import annotations

from .scenario_json import clean, load_json

IMPORT_NOTE = ("Only name, keys, text and on/off are used for now. Other lorebook settings "
               "(secondary keys, always-on, position, probability, …) were ignored.")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse(raw: bytes) -> list[dict]:
    """Read an uploaded lorebook file into [{name, keys, text, enabled}]. Raises ValueError if it's wrong."""
    if raw.startswith(PNG_SIGNATURE):
        raise ValueError("PNG character cards aren't supported yet. Export the card as JSON and import that.")
    entries = entries_from(load_json(raw))
    if not entries:
        raise ValueError("the lorebook has no entries with text")
    return entries


def entries_from(data) -> list[dict]:
    """Already-parsed JSON in any of the understood formats -> [{name, keys, text, enabled}]."""
    if isinstance(data, dict) and str(data.get("spec", "")).startswith("chara_card"):
        book = (data.get("data") or {}).get("character_book")
        if not isinstance(book, dict):
            raise ValueError("this character card has no lorebook")
        items = book.get("entries")
    elif isinstance(data, dict) and "lorebookVersion" in data:
        raise ValueError("NovelAI lorebooks aren't supported yet")
    elif isinstance(data, dict) and "entries" in data:
        items = data["entries"]
    else:
        items = data

    if isinstance(items, dict):
        items = list(items.values())  # SillyTavern keys its entries by number
    if not isinstance(items, list):
        raise ValueError("not a lorebook file")

    entries = []
    for number, item in enumerate(items, start=1):
        if isinstance(item, dict):
            entry = entry_from(item, number)
            if entry:
                entries.append(entry)
    return entries


def entry_from(item: dict, number: int) -> dict | None:
    """One entry in any of the formats, or None if it has no text."""
    text = clean(item.get("content") or item.get("text"), 20_000)
    if not text:
        return None

    raw_keys = item.get("keys") or item.get("key") or []
    if isinstance(raw_keys, str):
        raw_keys = [raw_keys]
    # Our keys are comma-separated, so a comma inside one key becomes a space.
    keys = [clean(k, one_line=True).replace(",", " ") for k in raw_keys]
    keys = [" ".join(k.split()) for k in keys if k.strip()]

    name = clean(item.get("name") or item.get("comment"), 200, one_line=True)
    if not name:
        name = keys[0][:200] if keys else f"Entry {number}"

    enabled = item.get("enabled") is not False and item.get("disable") is not True
    return {"name": name, "keys": clean(", ".join(keys), 500, one_line=True), "text": text, "enabled": enabled}


def to_export(name: str, entries: list) -> dict:
    """Our lorebook file: character_book-shaped, so other apps can read it too."""
    return {"name": name,
            "entries": [{"name": e.name, "keys": [k.strip() for k in e.keys.split(",") if k.strip()],
                         "content": e.text, "enabled": e.enabled} for e in entries]}
