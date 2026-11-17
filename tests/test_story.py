"""Story so far: summarizing older messages so long stories keep fitting in the context window."""

from zchat import chat, config, generate, story, zai
from zchat.db import session_scope
from zchat.models import Conversation, Message


def make_story(s, exchanges: int, system_prompt: str = "You narrate a heist.") -> Conversation:
    conv = Conversation(model="glm-4.7-flash", system_prompt=system_prompt)
    s.add(conv)
    s.flush()
    for i in range(1, exchanges + 1):
        user, reply = chat.send_user_message(s, conv, f"user {i}")
        reply.content, reply.status = f"narrator {i}", "done"
    return conv


class FakeComplete:
    """Stands in for zai.complete and remembers what it was asked."""

    def __init__(self, answer="SUMMARY"):
        self.answer = answer
        self.requests = []

    async def __call__(self, messages, model=None, max_tokens=0, timeout=0):
        self.requests.append(messages[-1]["content"])
        return self.answer


def test_build_messages_sends_summary_instead_of_older_messages(s):
    conv = make_story(s, 3)
    path = chat.active_path(s, conv)
    path[3].summary = "They met at the docks."  # covers user 1 .. narrator 2

    msgs = chat.build_messages(conv, chat.active_path(s, conv), conv.model)

    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("You narrate a heist.\n\n## Story so far")
    assert msgs[0]["content"].endswith("They met at the docks.")
    assert [m["content"] for m in msgs[1:]] == ["user 3", "narrator 3"]


def test_summary_follows_the_branches_it_belongs_to(s):
    conv = make_story(s, 3)
    path = chat.active_path(s, conv)
    path[3].summary = "S"

    # A branch made after the summarized message inherits the summary...
    chat.edit_user_message(s, path[4], "user 3, take two")
    assert "S" in chat.build_messages(conv, chat.active_path(s, conv), conv.model)[0]["content"]

    # ...one that splits off before it doesn't.
    chat.edit_user_message(s, path[2], "user 2, take two")
    assert "Story so far" not in chat.build_messages(conv, chat.active_path(s, conv), conv.model)[0]["content"]

    # A fork copies it.
    chat.switch_to(s, path[5])
    forked = chat.fork(s, conv, path[5])
    assert chat.latest_summary(chat.active_path(s, forked)).summary == "S"


def test_pick_cutoff_keeps_recent_messages_and_ends_on_a_reply(s, monkeypatch):
    monkeypatch.setattr(config, "SUMMARY_KEEP_RECENT", 4)
    conv = make_story(s, 5)  # 10 messages
    path = chat.active_path(s, conv)
    assert story.pick_cutoff(path) == 5  # narrator 3; the last 4 messages stay

    # Ending on a user turn (a reply about to be written): still cuts on a narrator reply.
    assert story.pick_cutoff(path[:9]) == 3

    # Nothing new since the last summary -> nothing to do. Too short -> nothing to do.
    path[5].summary = "S"
    assert story.pick_cutoff(path) is None
    assert story.pick_cutoff(path[:4]) is None


def test_summary_due_uses_the_smaller_of_the_setting_and_80_percent(s, monkeypatch):
    conv = make_story(s, 1)
    reply = chat.active_path(s, conv)[-1]
    monkeypatch.setattr(config, "SUMMARY_AT_TOKENS", 100_000)

    reply.prompt_tokens = 99_000
    assert not story.summary_due([reply], "glm-5.3")       # 1M window: the 100k setting decides
    reply.prompt_tokens = 100_000
    assert story.summary_due([reply], "glm-5.3")

    monkeypatch.setattr(config, "SUMMARY_AT_TOKENS", 500_000)
    reply.prompt_tokens = 103_000
    assert story.summary_due([reply], "glm-4.5")           # 128k window: 80% is ~102k

    monkeypatch.setattr(config, "SUMMARY_AT_TOKENS", 0)    # 0 = off
    assert not story.summary_due([reply], "glm-4.5")


async def test_summarize_branch_rolls_the_previous_summary_forward(monkeypatch):
    monkeypatch.setattr(config, "SUMMARY_KEEP_RECENT", 2)
    with session_scope() as s:
        conv = make_story(s, 4)
        conv_id, leaf_id = conv.id, conv.active_leaf_id
        ids = [m.id for m in chat.active_path(s, conv)]

    fake = FakeComplete("First summary.")
    monkeypatch.setattr(zai, "complete", fake)
    assert await story.summarize_branch(leaf_id) == ids[5]  # narrator 3
    assert "narrator 1" in fake.requests[0] and "narrator 3" in fake.requests[0]
    assert "narrator 4" not in fake.requests[0] and "You narrate a heist." in fake.requests[0]

    with session_scope() as s:  # two more exchanges, then summarize again
        conv = s.get(Conversation, conv_id)
        for i in (5, 6):
            _, reply = chat.send_user_message(s, conv, f"user {i}")
            reply.content, reply.status = f"narrator {i}", "done"
        leaf_id = conv.active_leaf_id

    fake.answer = "Second summary."
    summary_on = await story.summarize_branch(leaf_id)
    second = fake.requests[1]
    assert "First summary." in second                                  # previous summary is the starting point
    assert "narrator 4" in second and "narrator 5" in second
    assert "narrator 3" not in second and "narrator 6" not in second  # covered already / kept word for word
    with session_scope() as s:
        assert s.get(Message, summary_on).summary == "Second summary."


def nearly_full_story_with_pending_reply(monkeypatch) -> int:
    monkeypatch.setattr(config, "SUMMARY_KEEP_RECENT", 2)
    monkeypatch.setattr(config, "SUMMARY_AT_TOKENS", 50_000)
    with session_scope() as s:
        conv = make_story(s, 3)
        chat.active_path(s, conv)[-1].prompt_tokens = 60_000  # the last reply nearly filled the window
        _, pending = chat.send_user_message(s, conv, "user 4")
        return pending.id


async def test_reply_summarizes_first_when_the_context_is_nearly_full(monkeypatch):
    pending_id = nearly_full_story_with_pending_reply(monkeypatch)
    sent = []

    async def fake_stream(messages, model, thinking=False, temperature=None, effort="low"):
        sent.extend(messages)
        yield zai.StreamEvent("content", "narrator 4")

    monkeypatch.setattr(zai, "complete", FakeComplete("They cracked the vault."))
    monkeypatch.setattr(zai, "stream_chat", fake_stream)

    events = [e async for e in generate.run(pending_id)]

    names = [name for name, _ in events]
    assert names.index("summary") < names.index("delta")
    assert "They cracked the vault." in sent[0]["content"]
    assert [m["content"] for m in sent[1:]] == ["user 3", "narrator 3", "user 4"]


async def test_a_failed_summary_still_sends_the_reply(monkeypatch):
    pending_id = nearly_full_story_with_pending_reply(monkeypatch)

    async def broken(*args, **kwargs):
        raise zai.ZaiError("rate limited")

    async def fake_stream(messages, model, thinking=False, temperature=None, effort="low"):
        yield zai.StreamEvent("content", "narrator 4")

    monkeypatch.setattr(zai, "complete", broken)
    monkeypatch.setattr(zai, "stream_chat", fake_stream)

    events = [e async for e in generate.run(pending_id)]

    assert ("status", {"t": "Couldn't summarize (rate limited). Sending the full story."}) in events
    assert events[-1][0] == "done"
    with session_scope() as s:
        assert s.get(Message, pending_id).content == "narrator 4"
