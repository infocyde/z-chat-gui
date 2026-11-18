import json

import pytest

from zchat import scenario_json


def test_parse_accepts_one_object_or_a_list():
    one = scenario_json.parse(b'{"name": "Tavern", "system_prompt": "You narrate."}')
    assert one == [{"name": "Tavern", "description": "", "categories": "", "system_prompt": "You narrate.",
                    "lorebook": []}]
    many = scenario_json.parse(b'[{"name": "A"}, {"name": "B", "description": "two"}]')
    assert [s["name"] for s in many] == ["A", "B"]


@pytest.mark.parametrize("raw", [
    "﻿{\"name\": \"Café ☕\"}".encode("utf-8"),        # UTF-8 with a BOM
    "{\"name\": \"Café ☕\"}".encode("utf-16"),              # Notepad "Unicode"
])
def test_parse_handles_editor_encodings(raw):
    assert scenario_json.parse(raw)[0]["name"] == "Café ☕"


def test_parse_handles_old_windows_files():
    assert scenario_json.parse('{"name": "Café"}'.encode("cp1252"))[0]["name"] == "Café"


def test_parse_cleans_bad_characters():
    raw = json.dumps({"name": "Bad\x00 \n name \ud800", "description": None,
                      "system_prompt": "line1\r\nline2\x07\tok"}).encode()
    [s] = scenario_json.parse(raw)
    assert s["name"] == "Bad name ?"                     # control chars gone, one line, broken char -> ?
    assert s["description"] == ""
    assert s["system_prompt"] == "line1\nline2\tok"
    s["name"].encode("utf-8")  # safe to store


@pytest.mark.parametrize("raw, message", [
    (b"not json", "not valid JSON"),
    (b'"just a string"', "expected a scenario"),
    (b'[{"name": "ok"}, {"system_prompt": "no name"}]', "scenario #2 needs"),
    (b'[{"name": "   "}]', "scenario #1 needs"),
    (b"[" * 100_000, "nested too deeply"),
    (b" " * (scenario_json.MAX_FILE_BYTES + 1), "too big"),
])
def test_parse_rejects_bad_files_with_a_message(raw, message):
    with pytest.raises(ValueError, match=message):
        scenario_json.parse(raw)


def test_export_is_plain_ascii_and_round_trips():
    data = {"name": "Café ☕ \"quoted\"", "description": "", "categories": ["Sci-Fi", "Café"],
            "system_prompt": "Ünïcode\nlines"}
    text = scenario_json.dumps(data)
    assert text.isascii()
    assert scenario_json.parse(text.encode()) == [{**data, "categories": "Sci-Fi, Café", "lorebook": []}]


@pytest.mark.parametrize("value, expected", [
    (["fantasy", " Horror ", "fantasy"], "Fantasy, Horror"),  # tidied, repeats dropped, known spelling
    ("sci-fi,  Space   Opera ,", "Sci-Fi, Space Opera"),     # a comma-separated string works too
    (["Has, comma"], "Has comma"),                            # a comma inside a list item isn't a split
    (None, ""),
    (42, "42"),
])
def test_parse_reads_categories_as_a_list_or_a_string(value, expected):
    raw = json.dumps({"name": "X", "categories": value}).encode()
    assert scenario_json.parse(raw)[0]["categories"] == expected


def test_filename_and_unique_name():
    assert scenario_json.filename_for("The Gilded Lantern!") == "the-gilded-lantern.json"
    assert scenario_json.filename_for("☕☕") == "scenario.json"
    assert scenario_json.unique_name("Tavern", {"Tavern", "Tavern (2)"}) == "Tavern (3)"


def test_parse_accepts_raw_line_breaks_inside_strings():
    raw = b'{"name": "Hand\tmade", "system_prompt": "line one\nline two\x07"}'
    [s] = scenario_json.parse(raw)
    assert s["name"] == "Hand made" and s["system_prompt"] == "line one\nline two"


def test_scenario_export_and_import_keep_the_lorebook():
    from types import SimpleNamespace
    sc = SimpleNamespace(name="W", description="", categories="", system_prompt="P",
                         lore=[SimpleNamespace(name="A", keys="x, y", text="T", enabled=False)])
    data = scenario_json.to_dict(sc)
    assert data["lorebook"] == [{"name": "A", "keys": ["x", "y"], "content": "T", "enabled": False}]
    [back] = scenario_json.parse(scenario_json.dumps(data).encode())
    assert back["lorebook"] == [{"name": "A", "keys": "x, y", "text": "T", "enabled": False}]


def test_scenario_without_lore_exports_no_lorebook_key():
    from types import SimpleNamespace
    sc = SimpleNamespace(name="W", description="", categories="", system_prompt="P", lore=[])
    assert "lorebook" not in scenario_json.to_dict(sc)


def test_a_bad_lorebook_names_the_scenario():
    with pytest.raises(ValueError, match="scenario #1"):
        scenario_json.parse(b'{"name": "W", "lorebook": "not a list"}')
