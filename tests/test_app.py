"""End-to-end through the web routes, with the Z.ai calls faked."""

import re

import pytest
from starlette.testclient import TestClient

from zchat import zai


@pytest.fixture
def client(monkeypatch):
    async def fake_stream(messages, model, thinking=False, temperature=None, effort="low"):
        yield zai.StreamEvent("content", "**Welcome** to the tavern.")
        yield zai.StreamEvent("usage", data={"prompt_tokens": 50, "completion_tokens": 8})

    async def fake_complete(messages, model=None, max_tokens=0):
        return "Tavern Start"

    monkeypatch.setattr(zai, "stream_chat", fake_stream)
    monkeypatch.setattr(zai, "complete", fake_complete)
    import main

    return TestClient(main.app)


def test_full_flow_scenario_chat_stream_edit_fork(client):
    # Create a scenario and see it on the scenarios tab and the new-chat page
    r = client.post("/scenarios/save", data={"name": "Tavern", "system_prompt": "You narrate a fantasy tavern."})
    assert r.status_code in (200, 303)
    assert "Tavern" in client.get("/scenarios").text
    home = client.get("/").text
    assert "What are we doing today?" in home and "Tavern" in home
    sid = re.search(r'<option value="(\d+)">Tavern</option>', home).group(1)
    assert "fantasy tavern" in client.get(f"/scenario_prompt?scenario_id={sid}").text

    # Start a chat from the first page
    r = client.post("/new", data={"content": "I walk in.", "model": "glm-4.7-flash", "scenario_id": sid,
                                  "system_prompt": "You narrate a fantasy tavern."},
                    headers={"HX-Request": "true"})
    cid = re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1)
    page = client.get(f"/c/{cid}").text
    mid = re.search(r'data-stream-url="/stream\?mid=(\d+)"', page).group(1)

    # Stream the reply
    events = client.get(f"/stream?mid={mid}").text
    assert "event: delta" in events and "event: finished" in events and "event: sidebar" in events
    page = client.get(f"/c/{cid}").text
    assert "<strong>Welcome</strong>" in page and "Tavern Start" in page and "in 50" in page

    # Edit the user message -> new branch with a pending reply and a 2/2 navigator
    user_mid = re.search(r'id="msg-(\d+)" class="msg msg-user', page).group(1)
    r = client.post(f"/edit?mid={user_mid}", data={"content": "I kick the door open."})
    assert "I kick the door open." in r.text and "2/2" in r.text and "data-stream-url" in r.text

    # Fork from the first reply
    r = client.post(f"/fork?mid={mid}", headers={"HX-Request": "true"})
    fork_page = client.get(r.headers["HX-Redirect"]).text
    assert "(fork)" in fork_page and "I walk in." in fork_page


def test_effort_control_follows_the_model(client):
    # New-chat page swaps the control when the model changes
    assert 'name="reasoning_effort"' in client.get("/reasoning_control?model=glm-5.3").text
    assert 'name="thinking"' in client.get("/reasoning_control?model=glm-4.7-flash").text
    # ...and the price label next to it
    assert 'id="model-price"' in client.get("/reasoning_control?model=glm-5.3").text
    assert "$1.40" in client.get("/reasoning_control?model=glm-5.3").text
    assert ">Free" in client.get("/reasoning_control?model=glm-4.7-flash").text

    r = client.post("/new", data={"content": "hi", "model": "glm-5.3", "reasoning_effort": "max"},
                    headers={"HX-Request": "true"})
    cid = re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1)
    assert '<option value="max" selected>' in client.get(f"/c/{cid}").text

    # Switching to a thinking-switch model returns the switch control (out-of-band)
    r = client.post(f"/settings?cid={cid}", data={"model": "glm-4.7-flash", "reasoning_effort": "max"})
    assert 'name="thinking"' in r.text and "reasoning-control" in r.text


def test_clone_and_delete_scenario(client):
    from sqlalchemy import select

    from zchat.db import session_scope
    from zchat.models import Scenario

    client.post("/scenarios/save", data={"name": "Heist", "system_prompt": "You run a heist."})
    with session_scope() as s:
        sid = s.scalar(select(Scenario.id).where(Scenario.name == "Heist"))

    r = client.post(f"/scenarios/clone?sid={sid}", headers={"HX-Request": "true"})
    editor = client.get(r.headers["HX-Redirect"]).text
    assert 'value="Heist (copy)"' in editor and "You run a heist." in editor

    client.post(f"/scenarios/delete?sid={sid}", headers={"HX-Request": "true"})
    page = client.get("/scenarios").text
    assert "Heist (copy)" in page and ">Heist</h2>" not in page


def test_empty_send_shows_toast(client):
    r = client.post("/new", data={"content": "  "}, headers={"HX-Request": "true"})
    assert "toast" in r.text


def test_unknown_media_is_404(client):
    assert client.get("/media/99999").status_code == 404


def fake_plan():
    from datetime import datetime, timedelta

    soon = datetime.now() + timedelta(hours=2)
    return zai.PlanUsage("lite", [zai.PlanLimit("5-hour", 1700, 2000, 85, soon),
                                  zai.PlanLimit("Weekly", 444, 10000, 4, soon + timedelta(days=5))],
                         {"GLM-5.3": 1_200_000})


@pytest.fixture
def coding_plan(monkeypatch):
    """Pretend a Coding Plan key is configured; count how often Z.ai's quota API gets called."""
    import main
    from zchat import config

    calls = []

    async def fake_plan_usage():
        calls.append(1)
        return fake_plan()

    monkeypatch.setattr(zai, "plan_usage", fake_plan_usage)
    monkeypatch.setattr(config, "USING_CODING_PLAN", True)
    monkeypatch.setattr(main, "plan_cache", {"plan": None, "fetched_at": None, "replies": 0})
    return calls


def add_finished_replies(n: int):
    from zchat.db import session_scope
    from zchat.models import Conversation, Message

    with session_scope() as s:
        conv = Conversation(model="glm-5.3-flash")
        s.add(conv)
        s.flush()
        for _ in range(n):
            s.add(Message(conversation_id=conv.id, role="assistant", content="ok", status="done"))


def test_plan_usage_meters(client, coding_plan, monkeypatch):
    from zchat import config

    assert 'hx-get="/plan_usage?fresh=1"' in client.get("/").text
    html = client.get("/plan_usage").text
    assert "5-hour" in html and "Weekly" in html and "1,700 / 2,000" in html
    assert "meter-fill high" in html and "1.2M tokens" in html

    # Standard (pay-as-you-go) key: no plan, no meters
    monkeypatch.setattr(config, "USING_CODING_PLAN", False)
    assert 'id="plan-usage"' not in client.get("/").text
    assert client.get("/plan_usage").text == ""


def test_plan_usage_refreshes_every_n_replies(client, coding_plan, monkeypatch):
    from zchat import config

    monkeypatch.setattr(config, "PLAN_REFRESH_BY", "replies")
    monkeypatch.setattr(config, "PLAN_EVERY_REPLIES", 5)

    assert 'hx-get="/plan_usage?fresh=1"' in client.get("/").text  # page load / reload
    client.get("/plan_usage?fresh=1")
    client.get("/plan_usage")  # the check after a reply: reuses the saved numbers
    assert len(coding_plan) == 1
    client.get("/plan_usage?fresh=1")  # reloading the page always asks Z.ai
    assert len(coding_plan) == 2

    add_finished_replies(4)
    client.get("/plan_usage")
    assert len(coding_plan) == 2

    add_finished_replies(1)  # 5th reply since the last call
    html = client.get("/plan_usage").text
    assert len(coding_plan) == 3
    assert "every" not in html  # no timer in replies mode


def test_plan_usage_refreshes_every_n_seconds(client, coding_plan, monkeypatch):
    from datetime import datetime, timedelta

    import main
    from zchat import config

    monkeypatch.setattr(config, "PLAN_REFRESH_BY", "seconds")
    monkeypatch.setattr(config, "PLAN_EVERY_SECONDS", 60)

    html = client.get("/plan_usage").text
    assert "every 60s" in html
    client.get("/plan_usage")
    assert len(coding_plan) == 1

    main.plan_cache["fetched_at"] = datetime.now() - timedelta(seconds=61)
    client.get("/plan_usage")
    assert len(coding_plan) == 2


def test_header_controls_follow_the_model_actually_used(client, monkeypatch):
    """An old chat on a model this key can't use shows (and prices) the default model it falls back to."""
    from zchat import config
    from zchat.db import session_scope
    from zchat.models import Conversation

    monkeypatch.setattr(config, "AVAILABLE_MODELS", [config.MODEL_MAP["glm-5.3-flash"]])
    monkeypatch.setattr(config, "DEFAULT_MODEL", "glm-5.3-flash")
    with session_scope() as s:
        conv = Conversation(model="glm-4.5-flash")
        s.add(conv)
        s.flush()
        cid = conv.id

    page = client.get(f"/c/{cid}").text
    assert 'name="reasoning_effort"' in page and "$0.15" in page and ">Free<" not in page


def test_export_and_import_scenarios(client):
    import json

    from sqlalchemy import select

    from zchat.db import session_scope
    from zchat.models import Scenario

    client.post("/scenarios/save", data={"name": "Export Me ☕", "system_prompt": "Prompt one."})
    client.post("/scenarios/save", data={"name": "Export Me Too", "system_prompt": "Prompt two."})
    with session_scope() as s:
        ids = {sc.name: sc.id for sc in s.scalars(select(Scenario).where(Scenario.name.like("Export Me%")))}

    # One -> an object, named after the scenario
    r = client.get(f"/scenarios/export?sid={ids['Export Me ☕']}")
    assert 'filename="export-me.json"' in r.headers["content-disposition"]
    assert json.loads(r.text) == {"name": "Export Me ☕", "description": "", "system_prompt": "Prompt one."}

    # Several -> a list; none -> all
    two = json.loads(client.get(f"/scenarios/export?sid={ids['Export Me ☕']}&sid={ids['Export Me Too']}").text)
    assert sorted(x["name"] for x in two) == ["Export Me Too", "Export Me ☕"]
    everything = json.loads(client.get("/scenarios/export").text)
    assert len(everything) >= 2

    # Import several files at once: a list file, a single-object file, and a broken one
    files = [("files", ("two.json", json.dumps(two).encode(), "application/json")),
             ("files", ("one.json", b'{"name": "Fresh"}', "application/json")),
             ("files", ("broken.json", b"{oops", "application/json"))]
    r = client.post("/scenarios/import", files=files, headers={"HX-Request": "true"})
    assert "Imported 3 scenarios" in r.text and "broken.json: not valid JSON" in r.text
    assert "Export Me ☕ (2)" in r.text and "Fresh" in r.text  # clashing names don't overwrite
