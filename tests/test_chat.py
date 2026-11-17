from zchat import chat, config
from zchat.documents import Processed
from zchat.models import Conversation, Message


def new_conv(s, **kw) -> Conversation:
    conv = Conversation(model=kw.pop("model", "glm-4.7-flash"), system_prompt=kw.pop("system_prompt", "Be brief."), **kw)
    s.add(conv)
    s.flush()
    return conv


def reply(s, conv, text_in, text_out):
    """Add a finished user/assistant exchange to the active branch."""
    user, ans = chat.send_user_message(s, conv, text_in)
    ans.content, ans.status = text_out, "done"
    return user, ans


def test_send_appends_to_active_branch(s):
    conv = new_conv(s)
    u1, a1 = reply(s, conv, "hi", "hello")
    u2, a2 = reply(s, conv, "how are you", "fine")
    assert u2.parent_id == a1.id
    assert [m.id for m in chat.active_path(s, conv)] == [u1.id, a1.id, u2.id, a2.id]
    assert a2.status == "done" and conv.active_leaf_id == a2.id


def test_edit_user_message_creates_branch_and_keeps_original(s):
    conv = new_conv(s)
    u1, a1 = reply(s, conv, "hi", "hello")
    u2, a2 = reply(s, conv, "tell me a joke", "a joke")

    new_user, new_reply = chat.edit_user_message(s, u2, "tell me a poem")

    assert new_user.parent_id == a1.id
    assert new_reply.status == "pending"
    assert [m.id for m in chat.siblings(s, new_user)] == [u2.id, new_user.id]
    assert chat.active_path(s, conv)[-2].content == "tell me a poem"

    # Switching back to the original branch restores its reply
    chat.switch_to(s, u2)
    assert chat.active_path(s, conv)[-1].id == a2.id


def test_regenerate_makes_sibling_reply(s):
    conv = new_conv(s)
    u1, a1 = reply(s, conv, "hi", "hello")
    a1b = chat.regenerate(s, a1)
    assert a1b.parent_id == u1.id and a1b.status == "pending"
    assert len(chat.siblings(s, a1)) == 2


def test_fork_copies_branch_up_to_message(s):
    conv = new_conv(s, title="Campaign")
    u1, a1 = reply(s, conv, "start", "the tavern")
    reply(s, conv, "go on", "a dragon")

    new = chat.fork(s, conv, a1)

    path = chat.active_path(s, new)
    assert [m.content for m in path] == ["start", "the tavern"]
    assert new.system_prompt == conv.system_prompt
    assert new.title == "Campaign (fork)"
    assert all(m.conversation_id == new.id for m in path)


def test_build_messages_includes_system_docs_and_skips_errors(s):
    conv = new_conv(s, system_prompt="You are a narrator.")
    u1, a1 = reply(s, conv, "summarize", "summary")
    chat.add_attachment(s, u1, Processed("document", "notes.txt", "uploads/x.txt", "text/plain", 5, text="DOC TEXT"))
    u2, a2 = chat.send_user_message(s, conv, "again")
    a2.status, a2.error = "error", "boom"
    u3, a3 = chat.send_user_message(s, conv, "third")
    s.flush()
    s.expire_all()

    msgs = chat.build_messages(conv, chat.path_to(s, a3.parent_id), "glm-4.7-flash")

    assert msgs[0] == {"role": "system", "content": "You are a narrator."}
    assert '<document name="notes.txt">' in msgs[1]["content"] and "DOC TEXT" in msgs[1]["content"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "user"]  # error reply skipped


def test_images_only_go_to_vision_models(s, tmp_path):
    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG fake")
    conv = new_conv(s)
    u, a = chat.send_user_message(s, conv, "what is this?")
    chat.add_attachment(s, u, Processed("image", "cat.png", str(img), "image/png", 9))
    s.flush()
    s.expire_all()
    path = chat.path_to(s, a.parent_id)

    assert chat.effective_model(conv, path) == config.FALLBACK_VISION_MODEL
    vision_msgs = chat.build_messages(conv, path, "glm-4.6v-flash")
    assert vision_msgs[-1]["content"][0]["type"] == "image_url"
    text_msgs = chat.build_messages(conv, path, "glm-4.7-flash")
    assert "not visible to this model" in text_msgs[-1]["content"]


def test_usage_and_cost(s):
    conv = new_conv(s, model="glm-4.6")
    _, a = reply(s, conv, "q", "a")
    a.model, a.prompt_tokens, a.cached_tokens, a.completion_tokens = "glm-4.6", 1_000_000, 500_000, 100_000
    s.flush()
    u = chat.usage_for(s, conv.id)
    assert (u.prompt, u.cached, u.completion) == (1_000_000, 500_000, 100_000)
    # 500k uncached * 0.6 + 500k cached * 0.11 + 100k out * 2.2  (per 1M)
    assert round(u.cost, 4) == round(0.3 + 0.055 + 0.22, 4)
    assert chat.context_used(chat.active_path(s, conv)) == (1_100_000, "glm-4.6")
