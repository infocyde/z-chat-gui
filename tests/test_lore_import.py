"""Reading lorebooks made by other apps (SillyTavern, Character Cards) and our own exports."""

import json
from types import SimpleNamespace

import pytest

from zchat import lore_import, scenario_json

ST_WORLD = {"entries": {
    "0": {"uid": 0, "key": ["hag", "witch"], "keysecondary": ["swamp"], "comment": "Mire Hag",
          "content": "Swamp witch.", "constant": True, "disable": False, "position": 0},
    "1": {"uid": 1, "key": ["wolf"], "comment": "Gloomfang", "content": "Cave wolf.", "disable": True},
}}
CARD_V2 = {"spec": "chara_card_v2", "data": {"name": "Ash", "character_book": {"entries": [
    {"keys": ["wyrm"], "content": "Ash Wyrm sleeps.", "enabled": False, "name": "Ash Wyrm", "insertion_order": 1}]}}}
CARD_V3 = {"spec": "chara_card_v3", "data": {"character_book": {"entries": [{"keys": "gate", "content": "Sealed."}]}}}


def parse(data) -> list[dict]:
    return lore_import.parse(json.dumps(data).encode())


def test_sillytavern_world_info():
    assert parse(ST_WORLD) == [
        {"name": "Mire Hag", "keys": "hag, witch", "text": "Swamp witch.", "enabled": True},
        {"name": "Gloomfang", "keys": "wolf", "text": "Cave wolf.", "enabled": False},
    ]


def test_character_cards_v2_and_v3():
    assert parse(CARD_V2) == [{"name": "Ash Wyrm", "keys": "wyrm", "text": "Ash Wyrm sleeps.", "enabled": False}]
    assert parse(CARD_V3) == [{"name": "gate", "keys": "gate", "text": "Sealed.", "enabled": True}]  # no name: first key


def test_a_bare_character_book_or_list_of_entries():
    book = {"name": "Book", "entries": [{"keys": ["a"], "content": "A."}]}
    assert parse(book) == parse(book["entries"]) == [{"name": "a", "keys": "a", "text": "A.", "enabled": True}]


def test_our_export_round_trips():
    data = lore_import.to_export("World", [SimpleNamespace(name="A", keys="x, y", text="T", enabled=True)])
    assert data == {"name": "World", "entries": [{"name": "A", "keys": ["x", "y"], "content": "T", "enabled": True}]}
    assert parse(data) == [{"name": "A", "keys": "x, y", "text": "T", "enabled": True}]


def test_entries_without_text_are_skipped_and_names_are_filled_in():
    assert parse([{"content": ""}, {"content": "Only text"}]) == [
        {"name": "Entry 2", "keys": "", "text": "Only text", "enabled": True}]


def test_fields_are_cleaned():
    [e] = parse([{"name": "Bad\x00\n name", "keys": ["one, two", "three"], "content": "line1\r\nline2"}])
    assert e == {"name": "Bad name", "keys": "one two, three", "text": "line1\nline2", "enabled": True}


@pytest.mark.parametrize("raw, message", [
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "PNG character cards aren't supported yet"),
    (json.dumps({"lorebookVersion": 5, "entries": [{"text": "x"}]}).encode(), "NovelAI lorebooks aren't supported yet"),
    (json.dumps({"spec": "chara_card_v2", "data": {"name": "No book"}}).encode(), "has no lorebook"),
    (json.dumps({"entries": []}).encode(), "no entries"),
    (json.dumps([{"content": ""}]).encode(), "no entries"),
    (json.dumps("just text").encode(), "not a lorebook file"),
    (b"{", "not valid JSON"),
    (b" " * (scenario_json.MAX_FILE_BYTES + 1), "too big"),
])
def test_clear_errors(raw, message):
    with pytest.raises(ValueError, match=message):
        lore_import.parse(raw)


def test_the_import_note_says_what_is_ignored():
    assert lore_import.IMPORT_NOTE == ("Only name, keys, text and on/off are used for now. Other lorebook settings "
                                       "(secondary keys, always-on, position, probability, …) were ignored.")
