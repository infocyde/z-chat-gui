"""The lorebook updating itself every N replies, and drafting a starter lorebook."""

import json

import pytest

from zchat import chat, config, lore, lore_update, zai
from zchat.db import session_scope
from zchat.models import Conversation, LoreEntry, LoreVersion, Message


def make_story(s, exchanges: int) -> Conversation:
    conv = Conversation(model="glm-4.7-flash", system_prompt="You narrate a marsh.")
    s.add(conv)
    s.flush()
    for i in range(1, exchanges + 1):
        user, reply = chat.send_user_message(s, conv, f"user {i}")
        reply.content, reply.status = f"narrator {i}", "done"
    s.flush()
    return conv


def add_entry(s, conv, name, text, keys="", enabled=True) -> LoreVersion:
    v = LoreVersion(name=name, keys=keys, text=text, enabled=enabled)
    s.add(LoreEntry(conversation_id=conv.id, versions=[v]))
    s.flush()
    return v


class FakeComplete:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    async def __call__(self, messages, model=None, max_tokens=0, timeout=0):
        self.requests.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


CHANGES = json.dumps({"changes": [
    {"name": "gloomfang", "keys": ["wolf", "beast"], "text": "Wounded by Sela's spear; hunting the party."},
    {"name": "Hag's Lantern", "keys": ["lantern"], "text": "The Mire Hag's lantern, stolen by the party."},
]})


# ---------------------------------------------------------------- when


def test_update_due_counts_replies_since_the_last_update(s, monkeypatch):
    monkeypatch.setattr(config, "LORE_UPDATE_EVERY", 2)
    conv = make_story(s, 1)
    assert not lore_update.update_due(chat.active_path(s, conv))

    conv = make_story(s, 2)
    path = chat.active_path(s, conv)
    assert lore_update.update_due(path)

    path[3].lore_note = "No changes."  # the update ran after reply 2
    user, reply = chat.send_user_message(s, conv, "user 3")
    reply.content, reply.status = "narrator 3", "done"
    assert not lore_update.update_due(chat.active_path(s, conv))


def test_update_every_zero_means_never(s, monkeypatch):
    monkeypatch.setattr(config, "LORE_UPDATE_EVERY", 0)
    assert not lore_update.update_due(chat.active_path(s, make_story(s, 5)))


# ---------------------------------------------------------------- reading the model's answer


@pytest.mark.parametrize("answer, names", [
    ('{"changes": [{"name": "Gloomfang", "keys": ["wolf"], "text": "Wounded."}]}', ["Gloomfang"]),
    ('```json\n{"changes": [{"name": "A", "text": "x"}]}\n```', ["A"]),
    ('Here you go: {"changes": [{"name": "A", "keys": "a, b", "text": "x"}]} Hope that helps!', ["A"]),
    ("Sure! Here are the changes: none really.", []),
    ('{"changes": [{"name": "", "text": "x"}, {"name": "B"}, "junk", {"name": "C", "text": 5}]}', []),
    ('{"changes": "nope"}', []),
    ('{"changes": [', []),
    ("[]", []),
])
def test_parse_changes(answer, names):
    assert [c["name"] for c in lore_update.parse_changes(answer)] == names


def test_parse_changes_cleans_and_caps():
    many = {"changes": [{"name": f"N{i}\n", "keys": ["a,b", "c"], "text": "t\x00ext"} for i in range(20)]}
    changes = lore_update.parse_changes(json.dumps(many))
    assert len(changes) == lore_update.MAX_CHANGES == 8
    assert changes[0] == {"name": "N0", "keys": "a b, c", "text": "text"}


# ---------------------------------------------------------------- applying an update


async def test_update_adds_and_updates_on_this_branch_only(monkeypatch):
    fake = FakeComplete(CHANGES)
    monkeypatch.setattr(zai, "complete", fake)
    with session_scope() as s:
        conv = make_story(s, 2)
        add_entry(s, conv, "Gloomfang", "A cave wolf.", keys="wolf", enabled=False)
        add_entry(s, conv, "Mire Hag", "Swamp witch.")
        cid, leaf = conv.id, conv.active_leaf_id

    note_on = await lore_update.update_lore(leaf)

    assert note_on == leaf  # the newest reply
    with session_scope() as s:
        conv = s.get(Conversation, cid)
        path = chat.active_path(s, conv)
        assert path[3].lore_note == "Added: Hag's Lantern\nUpdated: Gloomfang"
        current = {v.name: v for v in lore.current_versions(s, cid, path)}
        assert current["Gloomfang"].text == "Wounded by Sela's spear; hunting the party."
        assert current["Gloomfang"].keys == "wolf, beast"
        assert current["Gloomfang"].enabled is False  # kept from the old version
        assert current["Gloomfang"].message_id == note_on
        assert current["Hag's Lantern"].message_id == note_on
        assert current["Mire Hag"].text == "Swamp witch."

        # A branch that splits off before that reply still sees the old lore.
        chat.edit_user_message(s, path[2], "user 2, take two")
        branch = {v.name: v.text for v in lore.current_versions(s, cid, chat.active_path(s, conv))}
        assert branch == {"Gloomfang": "A cave wolf.", "Mire Hag": "Swamp witch."}

    # The model saw the current lore and the story since the last update.
    request = fake.requests[0][-1]["content"]
    assert "Gloomfang" in request and "Mire Hag" in request and "narrator 2" in request


async def test_unchanged_entries_get_no_new_version(monkeypatch):
    monkeypatch.setattr(zai, "complete", FakeComplete(json.dumps(
        {"changes": [{"name": "Mire Hag", "keys": ["hag"], "text": "Swamp witch."}]})))
    with session_scope() as s:
        conv = make_story(s, 1)
        add_entry(s, conv, "Mire Hag", "Swamp witch.", keys="hag")
        leaf = conv.active_leaf_id

    note_on = await lore_update.update_lore(leaf)

    with session_scope() as s:
        assert s.get(Message, note_on).lore_note == "No changes."
        assert s.query(LoreVersion).filter(LoreVersion.message_id == note_on).count() == 0


async def test_only_messages_since_the_last_update_are_sent(monkeypatch):
    fake = FakeComplete('{"changes": []}')
    monkeypatch.setattr(zai, "complete", fake)
    with session_scope() as s:
        conv = make_story(s, 3)
        chat.active_path(s, conv)[1].lore_note = "No changes."  # an update ran after reply 1
        leaf = conv.active_leaf_id

    await lore_update.update_lore(leaf)

    request = fake.requests[0][-1]["content"]
    assert "narrator 1" not in request and "narrator 2" in request and "narrator 3" in request


async def test_failed_update_changes_nothing(monkeypatch):
    monkeypatch.setattr(zai, "complete", FakeComplete(zai.ZaiError("rate limited")))
    with session_scope() as s:
        conv = make_story(s, 1)
        leaf = conv.active_leaf_id

    with pytest.raises(zai.ZaiError):
        await lore_update.update_lore(leaf)

    with session_scope() as s:
        assert s.get(Message, leaf).lore_note == ""  # so it's tried again after the next reply


async def test_unreadable_answer_counts_as_no_changes(monkeypatch):
    monkeypatch.setattr(zai, "complete", FakeComplete("I think the story is going great!"))
    with session_scope() as s:
        conv = make_story(s, 1)
        leaf = conv.active_leaf_id

    await lore_update.update_lore(leaf)

    with session_scope() as s:
        assert s.get(Message, leaf).lore_note == "No changes."


async def test_draft_from_prompt(monkeypatch):
    fake = FakeComplete(json.dumps({"changes": [{"name": f"E{i}", "keys": [], "text": "t"} for i in range(15)]}))
    monkeypatch.setattr(zai, "complete", fake)

    drafted = await lore_update.draft_from_prompt("A marsh with a witch called the Mire Hag.", "glm-4.7-flash")

    assert len(drafted) == lore_update.MAX_DRAFT_ENTRIES == 12
    assert drafted[0] == {"name": "E0", "keys": "", "text": "t"}
    assert "Mire Hag" in fake.requests[0][-1]["content"]


async def test_nothing_to_update_without_a_reply():
    with session_scope() as s:
        conv = Conversation(model="glm-4.7-flash", system_prompt="")
        s.add(conv)
        s.flush()
        user, reply = chat.send_user_message(s, conv, "hi")  # the reply is still pending
        leaf = user.id

    assert await lore_update.update_lore(leaf) is None
