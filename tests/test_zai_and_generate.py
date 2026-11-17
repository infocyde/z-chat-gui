import json

import httpx
import pytest
import respx

from zchat import chat, config, generate, zai
from zchat.db import session_scope
from zchat.models import Conversation, Message


def sse_body(*chunks) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


@respx.mock
async def test_stream_chat_parses_reasoning_content_and_usage():
    body = sse_body(
        {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}}},
    )
    route = respx.post(f"{config.ZAI_BASE_URL}/chat/completions").mock(return_value=httpx.Response(200, text=body))

    events = [e async for e in zai.stream_chat([{"role": "user", "content": "hi"}], "glm-4.7-flash")]

    assert [(e.type, e.text) for e in events if e.type in ("reasoning", "content")] == [
        ("reasoning", "hmm"), ("content", "Hel"), ("content", "lo")]
    assert next(e for e in events if e.type == "usage").data["prompt_tokens"] == 10
    sent = json.loads(route.calls[0].request.content)
    assert sent["thinking"] == {"type": "disabled"} and sent["stream"] is True


@respx.mock
async def test_stream_chat_raises_friendly_balance_error():
    respx.post(f"{config.ZAI_BASE_URL}/chat/completions").mock(return_value=httpx.Response(
        429, json={"error": {"code": "1113", "message": "Insufficient balance"}}))
    with pytest.raises(zai.ZaiError) as err:
        [e async for e in zai.stream_chat([], "glm-4.6")]
    assert "no balance" in str(err.value) and err.value.code == "1113"


def test_always_thinking_models_get_reasoning_effort():
    assert zai.thinking_params("glm-5.3", False, "max") == {"reasoning_effort": "max"}
    assert zai.thinking_params("glm-5.3-flash", True, "bogus") == {"reasoning_effort": "low"}
    assert zai.thinking_params("glm-4.7-flash", True, "max") == {"thinking": {"type": "enabled"}}


def make_pending(text="hello?") -> int:
    with session_scope() as s:
        conv = Conversation(model="glm-4.7-flash", system_prompt="")
        s.add(conv)
        s.flush()
        _, pending = chat.send_user_message(s, conv, text)
        return pending.id


async def test_generate_run_saves_reply_usage_and_title(monkeypatch):
    async def fake_stream(messages, model, thinking=False, temperature=None, effort="low"):
        yield zai.StreamEvent("content", "Hi ")
        yield zai.StreamEvent("content", "there")
        yield zai.StreamEvent("usage", data={"prompt_tokens": 7, "completion_tokens": 2,
                                             "prompt_tokens_details": {"cached_tokens": 3}})

    async def fake_complete(messages, model=None, max_tokens=0):
        return "Friendly Greeting"

    monkeypatch.setattr(zai, "stream_chat", fake_stream)
    monkeypatch.setattr(zai, "complete", fake_complete)
    mid = make_pending()

    events = [name async for name, _ in generate.run(mid)]

    assert events == ["delta", "delta", "title", "done"]
    with session_scope() as s:
        m = s.get(Message, mid)
        assert (m.content, m.status, m.prompt_tokens, m.cached_tokens) == ("Hi there", "done", 7, 3)
        assert m.conversation.title == "Friendly Greeting"


async def test_generate_run_reports_api_errors(monkeypatch):
    async def failing_stream(*a, **kw):
        raise zai.ZaiError("Insufficient balance", 429, "1113")
        yield  # makes this an async generator

    monkeypatch.setattr(zai, "stream_chat", failing_stream)
    mid = make_pending()

    events = [(name, data) async for name, data in generate.run(mid)]

    assert events[-1][0] == "error" and "no balance" in events[-1][1]["msg"]
    with session_scope() as s:
        assert s.get(Message, mid).status == "error"


async def test_image_generation_downloads_to_image_output(monkeypatch):
    async def fake_generate(prompt, size="1280x1280", quality="hd"):
        return "https://cdn.example.com/abc.png?sig=1"

    async def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"png")
        return 3

    monkeypatch.setattr(zai, "generate_image", fake_generate)
    monkeypatch.setattr(zai, "download", fake_download)
    with session_scope() as s:
        conv = Conversation(model="glm-4.7-flash")
        s.add(conv)
        s.flush()
        conv.title = "Has title"
        _, pending = chat.send_user_message(s, conv, "a red lantern", image_mode=True)
        mid = pending.id

    events = [name async for name, _ in generate.run(mid)]

    assert events[-1] == "done"
    with session_scope() as s:
        m = s.get(Message, mid)
        [att] = m.attachments
        assert att.kind == "generated" and att.path.endswith(".png")
        assert (config.IMAGE_DIR / att.filename).exists()
        assert m.content == "a red lantern"
        assert chat.usage_for(s, m.conversation_id).images == 1


@respx.mock
async def test_plan_usage_reads_5_hour_and_weekly_limits():
    respx.get(f"{config.ZAI_USAGE_URL}/quota/limit").mock(return_value=httpx.Response(200, json={"data": {
        "level": "lite",
        "limits": [
            {"type": "CREDIT_LIMIT", "unit": 3, "number": 5, "usage": 2000, "currentValue": 115,
             "percentage": 5, "nextResetTime": 1794864556675},
            {"type": "CREDIT_LIMIT", "unit": 6, "number": 1, "usage": 10000, "currentValue": 444,
             "percentage": 4, "nextResetTime": 1795419217976},
        ]}}))
    respx.get(f"{config.ZAI_USAGE_URL}/model-usage").mock(return_value=httpx.Response(200, json={"data": {
        "modelSummaryList": [{"modelName": "GLM-5.3", "totalTokens": 400}, {"modelName": "GLM-5.3-Flash", "totalTokens": 600}]}}))

    plan = await zai.plan_usage()

    assert plan.level == "lite"
    assert [(x.label, x.used, x.total, x.percent) for x in plan.limits] == [
        ("5-hour", 115, 2000, 5), ("Weekly", 444, 10000, 4)]
    assert plan.week_tokens == {"GLM-5.3": 400, "GLM-5.3-Flash": 600}
