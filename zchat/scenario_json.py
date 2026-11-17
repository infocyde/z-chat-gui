"""Scenario import/export as JSON.

The format is as plain as it gets. One scenario is an object:

    {"name": "The Gilded Lantern", "description": "A harbor tavern", "system_prompt": "You narrate..."}

"Export all" is a list of those objects. Import accepts either.

Files come from anywhere (other editors, other people), so import is forgiving about encodings and
strips characters that could break the database or the page. Export writes plain ASCII JSON
(non-English text becomes \\u escapes), which every tool can read back.
"""

from __future__ import annotations

import json
import re

from .models import Scenario

MAX_FILE_BYTES = 5_000_000

# Invisible control characters (keeps tab, newline and carriage return)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def to_dict(s: Scenario) -> dict:
    return {"name": s.name, "description": s.description, "system_prompt": s.system_prompt}


def dumps(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def filename_for(name: str) -> str:
    """'The Gilded Lantern' -> 'the-gilded-lantern.json' (ASCII only, safe in any download header)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]
    return f"{slug or 'scenario'}.json"


def _decode(raw: bytes) -> str:
    """Bytes -> text, whatever editor saved the file."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")  # Notepad's "Unicode"
    try:
        return raw.decode("utf-8-sig")  # UTF-8, with or without a byte-order mark
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")  # old Windows files


def clean(value, max_len: int | None = None, one_line: bool = False) -> str:
    """Any JSON value -> safe text for the database and the page."""
    text = value if isinstance(value, str) else ("" if value is None else str(value))
    text = text.encode("utf-8", errors="replace").decode("utf-8")  # broken "\ud800"-style escapes -> "?"
    text = _CONTROL_CHARS.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    if one_line:
        text = " ".join(text.split())
    text = text.strip()
    return text[:max_len] if max_len else text


def parse(raw: bytes) -> list[dict]:
    """Read an uploaded file into a list of {name, description, system_prompt}. Raises ValueError if it's wrong."""
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"file is too big (limit {MAX_FILE_BYTES // 1_000_000} MB)")
    try:
        # strict=False: allow raw tabs/newlines inside strings (common in hand-edited files); clean() tidies them
        data = json.loads(_decode(raw), strict=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON ({e.msg}, line {e.lineno})")
    except RecursionError:
        raise ValueError("not a scenario file (nested too deeply)")

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("expected a scenario object or a list of them")

    scenarios = []
    for number, item in enumerate(data, start=1):
        name = clean(item.get("name"), 200, one_line=True) if isinstance(item, dict) else ""
        if not name:
            raise ValueError(f"scenario #{number} needs a \"name\"")
        scenarios.append({
            "name": name,
            "description": clean(item.get("description"), 500, one_line=True),
            "system_prompt": clean(item.get("system_prompt")),
        })
    return scenarios


def unique_name(name: str, taken: set[str]) -> str:
    """Imports never overwrite: a clash becomes 'Name (2)', 'Name (3)', ..."""
    candidate, n = name, 2
    while candidate in taken:
        candidate = f"{name} ({n})"
        n += 1
    return candidate
