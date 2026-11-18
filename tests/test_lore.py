"""Lorebooks: storage, branch-aware lookup, search, and what gets sent with a reply."""

import pytest
from sqlalchemy import func, select, text

from zchat import chat, config, lore
from zchat.models import Conversation, LoreEntry, LoreVersion, Scenario, ScenarioLore


def new_conv(s) -> Conversation:
    """A chat with one finished exchange."""
    conv = Conversation(model="glm-4.7-flash", system_prompt="You narrate a marsh.")
    s.add(conv)
    s.flush()
    user, reply = chat.send_user_message(s, conv, "I wade in.")
    reply.content, reply.status = "The reeds part.", "done"
    return conv


def fts_rowids(s) -> list[int]:
    return [row[0] for row in s.execute(text("SELECT rowid FROM lore_fts ORDER BY rowid"))]


# ---------------------------------------------------------------- storage


def test_fts_index_follows_lore_versions(s):
    conv = new_conv(s)
    entry = LoreEntry(conversation_id=conv.id)
    s.add(entry)
    s.flush()
    v = LoreVersion(entry_id=entry.id, name="Mire Hag", keys="hag, witch", text="Lures travelers with lantern light.")
    s.add(v)
    s.flush()
    assert v.id in fts_rowids(s)

    v.text = "Drowned in the marsh."
    s.flush()
    found = s.execute(text("SELECT rowid FROM lore_fts WHERE lore_fts MATCH 'drowned'")).scalars().all()
    assert found == [v.id]
    assert s.execute(text("SELECT rowid FROM lore_fts WHERE lore_fts MATCH 'lantern'")).scalars().all() == []


def test_deleting_a_chat_leaves_no_lore_behind(s):
    conv = new_conv(s)
    entry = LoreEntry(conversation_id=conv.id)
    s.add(entry)
    s.flush()
    v = LoreVersion(entry_id=entry.id, message_id=conv.active_leaf_id, name="Gloomfang", text="Cave wolf.")
    s.add(v)
    s.flush()
    version_id = v.id

    s.delete(conv)
    s.flush()
    s.expunge_all()

    assert version_id not in fts_rowids(s)
    assert s.get(LoreVersion, version_id) is None
    assert s.scalar(select(func.count(LoreEntry.id)).where(LoreEntry.id == entry.id)) == 0


def test_deleting_a_scenario_deletes_its_lorebook(s):
    sc = Scenario(name="Marsh", system_prompt="A marsh.")
    sc.lore = [ScenarioLore(name="Mire Hag", text="Swamp witch."), ScenarioLore(name="Ash Wyrm", text="Sleeps.")]
    s.add(sc)
    s.flush()
    ids = [x.id for x in sc.lore]

    s.delete(sc)
    s.flush()

    assert s.scalar(select(func.count(ScenarioLore.id)).where(ScenarioLore.id.in_(ids))) == 0


def test_lorebook_settings_have_defaults():
    assert (config.LORE_SCAN_MESSAGES, config.LORE_MAX_ENTRIES, config.LORE_MAX_TOKENS) == (4, 4, 800)
    assert config.LORE_VECTORS is True and config.LORE_MIN_SIMILARITY == 0.6
    assert config.LORE_UPDATE_EVERY == 10


# ---------------------------------------------------------------- branch-aware lookup


def story_with_lore(s, exchanges: int = 3):
    """A chat with `exchanges` finished exchanges and a lorebook seeded from a scenario."""
    sc = Scenario(name="Marsh", system_prompt="A marsh.")
    sc.lore = [ScenarioLore(name="Mire Hag", keys="hag, witch", text="Lures travelers.")]
    s.add(sc)
    s.flush()
    conv = Conversation(model="glm-4.7-flash", system_prompt=sc.system_prompt, scenario_id=sc.id)
    s.add(conv)
    s.flush()
    lore.seed_chat(s, conv, sc)
    for i in range(1, exchanges + 1):
        user, reply = chat.send_user_message(s, conv, f"user {i}")
        reply.content, reply.status = f"narrator {i}", "done"
    s.flush()
    return conv, chat.active_path(s, conv)


def add_entry(s, conv, name, text, keys="", enabled=True, message_id=None) -> LoreVersion:
    entry = LoreEntry(conversation_id=conv.id)
    s.add(entry)
    s.flush()
    v = LoreVersion(entry_id=entry.id, message_id=message_id, name=name, keys=keys, text=text, enabled=enabled)
    s.add(v)
    s.flush()
    return v


def texts(s, conv) -> list[str]:
    return [v.text for v in lore.current_versions(s, conv.id, chat.active_path(s, conv))]


def test_seed_chat_copies_the_scenario_lorebook(s):
    conv, path = story_with_lore(s)
    [v] = lore.current_versions(s, conv.id, path)
    assert (v.name, v.keys, v.text, v.enabled, v.message_id) == ("Mire Hag", "hag, witch", "Lures travelers.", True, None)


def test_current_versions_follow_the_branch(s):
    conv, path = story_with_lore(s)
    hag = lore.current_versions(s, conv.id, path)[0].entry
    # The lorebook update after reply 2 (path[3]) changed her.
    s.add(LoreVersion(entry_id=hag.id, message_id=path[3].id, name="Mire Hag", text="Dead."))
    s.flush()
    assert texts(s, conv) == ["Dead."]

    # A branch that splits off before that reply still sees the old version.
    chat.edit_user_message(s, path[2], "user 2, take two")
    assert texts(s, conv) == ["Lures travelers."]

    # Switching back brings the change back.
    chat.switch_to(s, path[5])
    assert texts(s, conv) == ["Dead."]


def test_fork_keeps_only_its_branch_versions(s):
    conv, path = story_with_lore(s)
    hag = lore.current_versions(s, conv.id, path)[0].entry
    s.add(LoreVersion(entry_id=hag.id, message_id=path[3].id, name="Mire Hag", text="Dead."))
    # A sibling branch of reply 3 made an entry of its own.
    other = chat.regenerate(s, path[5])
    other.status = "done"
    add_entry(s, conv, "Other Branch Thing", "Only over there.", message_id=other.id)
    chat.switch_to(s, path[5])

    forked = chat.fork(s, conv, path[5])

    assert texts(s, forked) == ["Dead."]
    forked_path = chat.active_path(s, forked)
    [v] = lore.current_versions(s, forked.id, forked_path)
    assert v.message_id == forked_path[3].id  # points at the fork's own copy of reply 2


def test_split_keys():
    assert lore.split_keys(" hag,  witch ,,") == ["hag", "witch"]


def test_query_words():
    assert lore.query_words('The Mire-Hag\'s "lantern", the lantern!') == ["mire", "hag", "lantern"]


@pytest.mark.parametrize("story", ['She says "NOT" (AND) -run* now: col:umn ^x', "🙂 é à", "a an of", ""])
def test_search_never_raises_on_story_text(s, story):
    conv, path = story_with_lore(s)
    ids = [v.id for v in lore.current_versions(s, conv.id, path)]
    assert isinstance(lore.search_fts(s, ids, lore.query_words(story)), list)


def test_search_matches_word_forms_and_accents(s):
    conv = new_conv(s)
    v = add_entry(s, conv, "Café Noir", "Lanterns hang over the door.")
    assert lore.search_fts(s, [v.id], lore.query_words("I look for a lantern near the cafe")) == [v.id]


def test_name_and_keys_outrank_body_text(s):
    conv = new_conv(s)
    a = add_entry(s, conv, "Old Road", "A lantern hangs on a post along the old road out of town.")
    b = add_entry(s, conv, "Keeper", "Tends the light.", keys="lantern")
    assert lore.search_fts(s, [a.id, b.id], ["lantern"]) == [b.id, a.id]


def test_search_only_looks_at_the_given_versions(s):
    conv = new_conv(s)
    a = add_entry(s, conv, "Lantern", "One.")
    add_entry(s, conv, "Lantern Two", "Two.")
    assert lore.search_fts(s, [a.id], ["lantern"]) == [a.id]


def test_find_for_reply_respects_caps_and_disabled(s, monkeypatch):
    monkeypatch.setattr(config, "LORE_MAX_ENTRIES", 2)
    conv = new_conv(s)
    add_entry(s, conv, "Disabled Lantern", "Off.", enabled=False)
    add_entry(s, conv, "Huge Lantern", "word " * (config.LORE_MAX_TOKENS * 4))
    add_entry(s, conv, "Lantern One", "One.")
    add_entry(s, conv, "Lantern Two", "Two.")
    add_entry(s, conv, "Lantern Three", "Three.")
    chat.send_user_message(s, conv, "I raise my lantern.")
    path = chat.active_path(s, conv)[:-1]  # the pending reply isn't part of what's searched

    names = [v.name for v in lore.find_for_reply(s, conv, path)]

    assert len(names) == 2
    assert "Disabled Lantern" not in names and "Huge Lantern" not in names


def test_find_for_reply_only_scans_recent_messages(s, monkeypatch):
    monkeypatch.setattr(config, "LORE_SCAN_MESSAGES", 2)
    conv, _ = story_with_lore(s, exchanges=0)
    add_entry(s, conv, "Gloomfang", "Cave wolf.")
    for text in ["Gloomfang howls.", "Quiet.", "Still quiet."]:
        user, reply = chat.send_user_message(s, conv, text)
        reply.content, reply.status = "Nothing happens.", "done"
    path = chat.active_path(s, conv)
    assert lore.find_for_reply(s, conv, path) == []

    chat.send_user_message(s, conv, "Is that Gloomfang?")
    path = chat.active_path(s, conv)[:-1]
    assert [v.name for v in lore.find_for_reply(s, conv, path)] == ["Gloomfang"]


def test_lore_block_wording():
    block = lore.lore_block([LoreVersion(name="Mire Hag", keys="", text="Swamp witch."),
                             LoreVersion(name="Gloomfang", keys="", text="Cave wolf.")])
    assert block.startswith("<world_notes>\nBackground from the lorebook, for the narrator.")
    assert "### Mire Hag\nSwamp witch.\n\n### Gloomfang\nCave wolf." in block
    assert block.endswith("</world_notes>")
    assert lore.lore_block([]) == ""
