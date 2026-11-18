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

    client.post("/scenarios/save", data={"name": "Export Me ☕", "system_prompt": "Prompt one.",
                                         "categories": "Drama"})
    client.post("/scenarios/save", data={"name": "Export Me Too", "system_prompt": "Prompt two."})
    with session_scope() as s:
        ids = {sc.name: sc.id for sc in s.scalars(select(Scenario).where(Scenario.name.like("Export Me%")))}

    # One -> an object, named after the scenario
    r = client.get(f"/scenarios/export?sid={ids['Export Me ☕']}")
    assert 'filename="export-me.json"' in r.headers["content-disposition"]
    assert json.loads(r.text) == {"name": "Export Me ☕", "description": "", "categories": ["Drama"],
                                  "system_prompt": "Prompt one."}

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


def scenario_named(name):
    from sqlalchemy import select

    from zchat.db import session_scope
    from zchat.models import Scenario

    with session_scope() as s:
        return s.scalar(select(Scenario).where(Scenario.name == name))


def test_scenario_categories_typed_and_suggested(client):
    # Typed categories are tidied: trimmed, repeats dropped, spelled like the common ones.
    client.post("/scenarios/save", data={"name": "Cat Typed", "system_prompt": "x",
                                         "categories": " sci-fi, Space  Opera ,SCI-FI,"})
    assert scenario_named("Cat Typed").categories == "Sci-Fi, Space Opera"

    # The editor offers common categories and ones already used, with the scenario's own lit up.
    page = client.get(f"/scenarios/edit?sid={scenario_named('Cat Typed').id}").text
    assert 'value="Sci-Fi, Space Opera"' in page
    assert re.search(r'class="tag-btn on"[^>]*>Space Opera<', page)
    assert re.search(r'class="tag-btn"[^>]*>Horror<', page)

    # The scenarios page shows them as tags that search for that category.
    assert 'href="/scenarios?q=Space%20Opera"' in client.get("/scenarios").text


def test_empty_categories_are_picked_by_the_ai(client, monkeypatch):
    asked = []

    async def fake_complete(messages, model=None, max_tokens=0):
        asked.append(messages[0]["content"])
        return "horror, Mystery."

    monkeypatch.setattr(zai, "complete", fake_complete)
    client.post("/scenarios/save", data={"name": "Cat Auto", "system_prompt": "A haunted lighthouse."})
    assert scenario_named("Cat Auto").categories == "Horror, Mystery"
    assert "haunted lighthouse" in asked[0]

    # A chatty reply instead of a list is ignored rather than saved.
    async def chatty(messages, model=None, max_tokens=0):
        return "Sure! I think this scenario would best fit into the category of gothic horror fiction overall"

    monkeypatch.setattr(zai, "complete", chatty)
    client.post("/scenarios/save", data={"name": "Cat Chatty", "system_prompt": "Fog."})
    assert scenario_named("Cat Chatty").categories == ""

    # Nothing to go on (no prompt, no description): no AI call at all.
    asked.clear()
    monkeypatch.setattr(zai, "complete", fake_complete)
    client.post("/scenarios/save", data={"name": "Cat Blank"})
    assert asked == [] and scenario_named("Cat Blank").categories == ""

    # Its card still gets a categories line: a link to add some.
    blank_id = scenario_named("Cat Blank").id
    assert f'href="/scenarios/edit?sid={blank_id}" class="tag tag-empty"' in client.get("/scenarios").text


def test_search_scenarios(client):
    client.post("/scenarios/save", data={"name": "Search Lighthouse", "categories": "Horror",
                                         "system_prompt": "A keeper hears knocking at 100% volume."})
    client.post("/scenarios/save", data={"name": "Search Starship", "categories": "Sci-Fi",
                                         "description": "Bridge crew drama", "system_prompt": "Captain."})

    def found(q):
        page = client.get("/scenarios", params={"q": q}).text
        return [name for name in ("Search Lighthouse", "Search Starship") if name in page]

    assert found("lighthouse") == ["Search Lighthouse"]           # name, any case
    assert found("sci-fi") == ["Search Starship"]                 # category
    assert found("knocking") == ["Search Lighthouse"]             # prompt text
    assert found("bridge") == ["Search Starship"]                 # description
    assert found("search") == ["Search Lighthouse", "Search Starship"]
    assert found("search horror") == ["Search Lighthouse"]        # every word must match
    assert found("100%") == ["Search Lighthouse"]                 # % is a character, not a wildcard
    assert found("1_0") == []                                     # so is _
    page = client.get("/scenarios", params={"q": "zzz-nothing"}).text
    assert "No scenarios match" in page and 'value="zzz-nothing"' in page


def test_summarize_button_edit_and_remove(client, monkeypatch):
    from zchat import config
    from zchat.db import session_scope
    from zchat.models import Message

    monkeypatch.setattr(config, "SUMMARY_KEEP_RECENT", 2)
    r = client.post("/new", data={"content": "I walk in.", "model": "glm-4.7-flash"},
                    headers={"HX-Request": "true"})
    cid = re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1)
    first_reply = int(re.search(r'data-stream-url="/stream\?mid=(\d+)"', client.get(f"/c/{cid}").text).group(1))
    client.get(f"/stream?mid={first_reply}")

    # One exchange: nothing to summarize yet.
    r = client.post(f"/summarize?cid={cid}", headers={"HX-Request": "true"})
    assert "Nothing new to summarize" in r.text

    # A second exchange; now the first one can be summarized.
    r = client.post(f"/send?cid={cid}", data={"content": "I order a drink."}, headers={"HX-Request": "true"})
    second_reply = re.search(r'data-stream-url="/stream\?mid=(\d+)"', r.text).group(1)
    client.get(f"/stream?mid={second_reply}")

    async def fake_complete(messages, model=None, max_tokens=0, timeout=0):
        return "**Arrived** at the tavern."

    monkeypatch.setattr(zai, "complete", fake_complete)
    r = client.post(f"/summarize?cid={cid}", headers={"HX-Request": "true"})
    assert "Story so far" in r.text and "<strong>Arrived</strong>" in r.text
    assert f'id="summary-{first_reply}"' in r.text  # saved on the first reply
    assert f'id="summary-{first_reply}"' in client.get(f"/c/{cid}").text  # and shown on the page

    # Edit it
    assert "Arrived" in client.get(f"/summary_form?mid={first_reply}").text
    r = client.post(f"/summary?mid={first_reply}", data={"summary": "Edited summary."},
                    headers={"HX-Request": "true"})
    assert "Edited summary." in r.text
    with session_scope() as s:
        assert s.get(Message, first_reply).summary == "Edited summary."

    # Remove it
    r = client.post(f"/summary_remove?mid={first_reply}", headers={"HX-Request": "true"})
    assert 'id="summary-' not in r.text
    with session_scope() as s:
        assert s.get(Message, first_reply).summary == ""


def test_reply_sends_and_shows_the_lore_it_used(client, monkeypatch):
    from zchat.db import session_scope
    from zchat.models import Scenario, ScenarioLore

    sent = []

    async def fake_stream(messages, model, thinking=False, temperature=None, effort="low"):
        sent.append(messages)
        yield zai.StreamEvent("content", "A lantern glows in the reeds.")
        yield zai.StreamEvent("usage", data={"prompt_tokens": 50, "completion_tokens": 8})

    monkeypatch.setattr(zai, "stream_chat", fake_stream)
    with session_scope() as s:
        sc = Scenario(name="Lore Marsh", system_prompt="A marsh.")
        sc.lore = [ScenarioLore(name="Mire Hag", keys="hag", text="Swamp witch."),
                   ScenarioLore(name="Gloomfang", keys="wolf", text="Cave wolf.")]
        s.add(sc)
        s.flush()
        sid = sc.id

    r = client.post("/new", data={"content": "I look for the hag.", "model": "glm-4.7-flash", "scenario_id": sid,
                                  "system_prompt": "A marsh."}, headers={"HX-Request": "true"})
    cid = re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1)
    mid = re.search(r'data-stream-url="/stream\?mid=(\d+)"', client.get(f"/c/{cid}").text).group(1)
    client.get(f"/stream?mid={mid}")

    last = sent[-1][-1]["content"]
    assert "### Mire Hag\nSwamp witch." in last and "Gloomfang" not in last
    assert last.endswith("I look for the hag.")
    assert "Lore: Mire Hag" in client.get(f"/c/{cid}").text


# ---------------------------------------------------------------- lorebooks

ST_WORLD = {"entries": {"0": {"key": ["hag"], "comment": "Mire Hag", "content": "Swamp witch.", "constant": True}}}


def lore_names(sid):
    from sqlalchemy import select

    from zchat.db import session_scope
    from zchat.models import ScenarioLore

    with session_scope() as s:
        return list(s.scalars(select(ScenarioLore.name).where(ScenarioLore.scenario_id == sid)
                              .order_by(ScenarioLore.id)))


def test_scenario_lorebook_add_edit_delete(client):
    client.post("/scenarios/save", data={"name": "Lore Editor", "system_prompt": "A marsh.", "categories": "Fantasy"})
    sid = scenario_named("Lore Editor").id

    # A new scenario's editor explains that the lorebook comes after saving.
    assert "Save the scenario first to add a lorebook." in client.get("/scenarios/edit").text
    page = client.get(f"/scenarios/edit?sid={sid}").text
    assert "Lorebook" in page and f'hx-get="/lore/scenario/form?sid={sid}"' in page

    assert 'name="name"' in client.get(f"/lore/scenario/form?sid={sid}").text
    r = client.post("/lore/scenario/save", data={"sid": sid, "name": "Mire Hag", "keys": "hag, witch",
                                                 "text": "Swamp witch.", "enabled": "1"})
    assert "Mire Hag" in r.text and "witch" in r.text and 'id="scenario-lore"' in r.text
    assert "Mire Hag" in client.get(f"/scenarios/edit?sid={sid}").text

    from zchat.db import session_scope
    from zchat.models import Scenario
    with session_scope() as s:
        lid = s.get(Scenario, sid).lore[0].id

    # Edit: the form is filled in; an unchecked "enabled" box means off.
    assert "Swamp witch." in client.get(f"/lore/scenario/form?sid={sid}&lid={lid}").text
    r = client.post("/lore/scenario/save", data={"sid": sid, "lid": lid, "name": "Mire Hag", "keys": "hag",
                                                 "text": "Drowned."})
    assert "Drowned." in r.text and "Disabled" in r.text
    assert lore_names(sid) == ["Mire Hag"]

    r = client.post(f"/lore/scenario/delete?lid={lid}")
    assert "Mire Hag" not in r.text and lore_names(sid) == []


def test_scenario_lorebook_import_and_export(client):
    import json

    client.post("/scenarios/save", data={"name": "Lore Files", "system_prompt": "A marsh.", "categories": "Fantasy"})
    sid = scenario_named("Lore Files").id
    client.post("/lore/scenario/save", data={"sid": sid, "name": "Mire Hag", "keys": "", "text": "Hand-written.",
                                             "enabled": "1"})

    files = [("files", ("world.json", json.dumps(ST_WORLD).encode(), "application/json")),
             ("files", ("bad.json", b"{", "application/json"))]
    r = client.post(f"/lore/scenario/import?sid={sid}", files=files, headers={"HX-Request": "true"})
    assert "Imported 1 entry" in r.text and "Other lorebook settings" in r.text
    assert "bad.json: not valid JSON" in r.text
    assert lore_names(sid) == ["Mire Hag", "Mire Hag"]  # imports never overwrite

    r = client.get(f"/lore/scenario/export?sid={sid}")
    assert 'filename="lore-files-lorebook.json"' in r.headers["content-disposition"]
    assert [e["content"] for e in json.loads(r.text)["entries"]] == ["Hand-written.", "Swamp witch."]


def test_new_chat_and_clone_copy_the_scenario_lorebook(client):
    from zchat import chat, lore
    from zchat.db import session_scope
    from zchat.models import Conversation

    client.post("/scenarios/save", data={"name": "Lore Copies", "system_prompt": "A marsh.", "categories": "Fantasy"})
    sid = scenario_named("Lore Copies").id
    client.post("/lore/scenario/save", data={"sid": sid, "name": "Mire Hag", "keys": "hag", "text": "Swamp witch.",
                                             "enabled": "1"})

    r = client.post("/new", data={"content": "Hello.", "model": "glm-4.7-flash", "scenario_id": sid,
                                  "system_prompt": "A marsh."}, headers={"HX-Request": "true"})
    cid = int(re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1))

    client.post(f"/scenarios/clone?sid={sid}", headers={"HX-Request": "true"})
    clone = scenario_named("Lore Copies (copy)")
    assert lore_names(clone.id) == ["Mire Hag"]

    # Editing the scenario afterwards doesn't change the chat that already started.
    from zchat.models import Scenario
    with session_scope() as s:
        lid = s.get(Scenario, sid).lore[0].id
    client.post(f"/lore/scenario/delete?lid={lid}")
    with session_scope() as s:
        conv = s.get(Conversation, cid)
        assert [v.name for v in lore.current_versions(s, cid, chat.active_path(s, conv))] == ["Mire Hag"]


def chat_lore(cid):
    """(name, text, message_id) of each current version on the chat's active branch."""
    from zchat import chat, lore
    from zchat.db import session_scope
    from zchat.models import Conversation

    with session_scope() as s:
        conv = s.get(Conversation, cid)
        return [(v.name, v.text, v.message_id) for v in lore.current_versions(s, cid, chat.active_path(s, conv))]


def test_chat_lorebook_drawer(client):
    client.post("/scenarios/save", data={"name": "Lore Drawer", "system_prompt": "A marsh.", "categories": "Fantasy"})
    sid = scenario_named("Lore Drawer").id
    client.post("/lore/scenario/save", data={"sid": sid, "name": "Mire Hag", "keys": "hag", "text": "Swamp witch.",
                                             "enabled": "1"})
    r = client.post("/new", data={"content": "Hello.", "model": "glm-4.7-flash", "scenario_id": sid,
                                  "system_prompt": "A marsh."}, headers={"HX-Request": "true"})
    cid = int(re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1))
    page = client.get(f"/c/{cid}").text
    assert 'id="chat-lore"' in page and "Mire Hag" in page and "1 entry" in page

    from zchat.db import session_scope
    from zchat.models import LoreVersion
    from sqlalchemy import select
    with session_scope() as s:
        vid = s.scalar(select(LoreVersion.id).join(LoreVersion.entry).where(LoreVersion.name == "Mire Hag")
                       .order_by(LoreVersion.id.desc()))

    # Edit in place: same version, still for every branch.
    assert "Swamp witch." in client.get(f"/lore/chat/form?cid={cid}&vid={vid}").text
    r = client.post("/lore/chat/save", data={"cid": cid, "vid": vid, "name": "Mire Hag", "keys": "hag",
                                             "text": "Drowned.", "enabled": "1"})
    assert "Drowned." in r.text
    assert chat_lore(cid) == [("Mire Hag", "Drowned.", None)]

    # Add one by hand: every branch sees it. The count updates too.
    r = client.post("/lore/chat/save", data={"cid": cid, "name": "Gloomfang", "keys": "wolf", "text": "Cave wolf.",
                                             "enabled": "1"})
    assert "Gloomfang" in r.text and "2 entries" in r.text
    assert chat_lore(cid) == [("Gloomfang", "Cave wolf.", None), ("Mire Hag", "Drowned.", None)]

    # Delete removes the whole entry, on every branch, and the confirm says so.
    assert 'from the lorebook? It goes on every branch.' in client.get(f"/c/{cid}").text
    r = client.post(f"/lore/chat/delete?vid={vid}")
    assert "Mire Hag" not in r.text and "1 entry" in r.text
    assert chat_lore(cid) == [("Gloomfang", "Cave wolf.", None)]


def test_switching_branches_refreshes_the_lorebook_drawer(client):
    r = client.post("/new", data={"content": "Hello.", "model": "glm-4.7-flash"}, headers={"HX-Request": "true"})
    cid = int(re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1))
    mid = re.search(r'data-stream-url="/stream\?mid=(\d+)"', client.get(f"/c/{cid}").text).group(1)
    client.get(f"/stream?mid={mid}")
    assert "No entries yet" in client.get(f"/c/{cid}").text
    r = client.get(f"/switch?mid={mid}")
    assert 'id="chat-lore"' in r.text and 'hx-swap-oob="true"' in r.text


def test_editing_a_chat_entry_re_embeds_it(client):
    from zchat import lore
    from zchat.db import session_scope
    from zchat.models import LoreVersion

    r = client.post("/new", data={"content": "Hello.", "model": "glm-4.7-flash"}, headers={"HX-Request": "true"})
    cid = int(re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1))
    client.post("/lore/chat/save", data={"cid": cid, "name": "Mire Hag", "text": "Swamp witch.", "enabled": "1"})
    with session_scope() as s:
        v = s.scalar(select_version(cid))
        lore.ensure_embeddings(s, [v])
        vid = v.id
        assert v.embedding is not None

    client.post("/lore/chat/save", data={"cid": cid, "vid": vid, "name": "Mire Hag", "text": "Drowned.", "enabled": "1"})
    with session_scope() as s:
        assert s.get(LoreVersion, vid).embedding is None  # made again before the next search


def select_version(cid):
    from sqlalchemy import select

    from zchat.models import LoreEntry, LoreVersion
    return select(LoreVersion).join(LoreEntry).where(LoreEntry.conversation_id == cid)


def start_chat_with_one_reply(client) -> tuple[str, str]:
    """A new chat whose first reply has streamed. Returns (chat id, reply id)."""
    r = client.post("/new", data={"content": "I walk in.", "model": "glm-4.7-flash"}, headers={"HX-Request": "true"})
    cid = re.search(r"/c/(\d+)", r.headers["HX-Redirect"]).group(1)
    first = re.search(r'data-stream-url="/stream\?mid=(\d+)"', client.get(f"/c/{cid}").text).group(1)
    events = client.get(f"/stream?mid={first}").text
    assert "event: lore" not in events  # no replies before this one: nothing to update yet
    return cid, first


def test_stream_runs_a_due_update_before_the_reply_and_undo_reverts_it(client, monkeypatch):
    import json

    from zchat import config
    from zchat.db import session_scope
    from zchat.models import LoreVersion, Message

    monkeypatch.setattr(config, "LORE_UPDATE_EVERY", 1)
    cid, first = start_chat_with_one_reply(client)

    async def fake_complete(messages, model=None, max_tokens=0, timeout=0):
        return json.dumps({"changes": [{"name": "Barkeep", "keys": ["barkeep"], "text": "Grumpy."}]})

    monkeypatch.setattr(zai, "complete", fake_complete)
    r = client.post(f"/send?cid={cid}", data={"content": "I order a drink."}, headers={"HX-Request": "true"})
    second = re.search(r'data-stream-url="/stream\?mid=(\d+)"', r.text).group(1)
    events = client.get(f"/stream?mid={second}").text
    assert "Updating the lorebook" in events and "event: lore" in events
    assert events.index("event: lore") < events.index("event: delta")  # before the reply, so it's used

    page = client.get(f"/c/{cid}").text
    assert f'id="lore-note-{first}"' in page and "Barkeep" in page
    assert "including any edits you made to them since" in page  # Undo's confirm

    r = client.post(f"/lore/undo?mid={first}", headers={"HX-Request": "true"})
    assert "undone" in r.text and 'id="chat-lore"' in r.text
    with session_scope() as s:
        assert s.query(LoreVersion).filter(LoreVersion.message_id == int(first)).count() == 0
        assert s.get(Message, int(first)).lore_note == "Undone."


def test_failed_update_does_not_stop_the_reply(client, monkeypatch):
    from zchat import config

    monkeypatch.setattr(config, "LORE_UPDATE_EVERY", 1)
    cid, first = start_chat_with_one_reply(client)

    async def failing_complete(messages, model=None, max_tokens=0, timeout=0):
        raise zai.ZaiError("rate limited")

    monkeypatch.setattr(zai, "complete", failing_complete)
    r = client.post(f"/send?cid={cid}", data={"content": "Again."}, headers={"HX-Request": "true"})
    second = re.search(r'data-stream-url="/stream\?mid=(\d+)"', r.text).group(1)
    events = client.get(f"/stream?mid={second}").text
    assert "update the lorebook (rate limited)" in events and "event: finished" in events


def test_draft_button_adds_entries_and_skips_existing_names(client, monkeypatch):
    import json

    async def fake_complete(messages, model=None, max_tokens=0, timeout=0):
        return json.dumps({"changes": [{"name": "mire hag", "keys": ["hag"], "text": "Duplicate."},
                                       {"name": "Gloomfang", "keys": ["wolf"], "text": "Cave wolf."}]})

    monkeypatch.setattr(zai, "complete", fake_complete)
    client.post("/scenarios/save", data={"name": "Lore Draft", "system_prompt": "A marsh.", "categories": "Fantasy"})
    sid = scenario_named("Lore Draft").id
    client.post("/lore/scenario/save", data={"sid": sid, "name": "Mire Hag", "text": "Swamp witch.", "enabled": "1"})
    assert f'hx-post="/lore/scenario/draft?sid={sid}"' in client.get(f"/scenarios/edit?sid={sid}").text

    r = client.post(f"/lore/scenario/draft?sid={sid}", headers={"HX-Request": "true"})

    assert "Drafted 1 entry from the system prompt" in r.text and "Gloomfang" in r.text
    assert lore_names(sid) == ["Mire Hag", "Gloomfang"]


def test_draft_needs_a_system_prompt_and_reports_failures(client, monkeypatch):
    client.post("/scenarios/save", data={"name": "Lore Draft Empty", "categories": "Fantasy"})
    sid = scenario_named("Lore Draft Empty").id
    r = client.post(f"/lore/scenario/draft?sid={sid}", headers={"HX-Request": "true"})
    assert "Write a system prompt first" in r.text

    async def failing_complete(messages, model=None, max_tokens=0, timeout=0):
        raise zai.ZaiError("rate limited")

    monkeypatch.setattr(zai, "complete", failing_complete)
    client.post("/scenarios/save", data={"sid": sid, "name": "Lore Draft Empty", "system_prompt": "A marsh.",
                                         "categories": "Fantasy"})
    r = client.post(f"/lore/scenario/draft?sid={sid}", headers={"HX-Request": "true"})
    assert "Couldn't draft a lorebook: rate limited" in r.text and lore_names(sid) == []


def test_edit_and_regenerate_refresh_the_lorebook_drawer(client):
    # Both move to a new branch, which can have different lore.
    cid, first = start_chat_with_one_reply(client)
    from zchat.db import session_scope
    from zchat.models import Message
    with session_scope() as s:
        user_id = s.get(Message, int(first)).parent_id

    r = client.post(f"/edit?mid={user_id}", data={"content": "I run in."}, headers={"HX-Request": "true"})
    assert 'id="chat-lore"' in r.text and 'hx-swap-oob="true"' in r.text

    r = client.post(f"/regen?mid={first}", headers={"HX-Request": "true"})
    assert 'id="chat-lore"' in r.text and 'hx-swap-oob="true"' in r.text
