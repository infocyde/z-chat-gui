"""Lorebooks: storage, branch-aware lookup, search, and what gets sent with a reply."""

from sqlalchemy import func, select, text

from zchat import chat, config
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
