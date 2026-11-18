"""zchat: a FastHTML chat client for Z.ai.

Run with:  uv run main.py   then open http://localhost:5001
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import httpx
from fasthtml.common import (Div, FileResponse, Link, Meta, P, Redirect, Response, Script, Span,
                             StreamingResponse, fast_app, serve, to_xml)
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from zchat import (category, chat, config, documents, embeddings, generate, lore, lore_import, lore_ui,
                   lore_update, pricing, scenario_json, story, ui, zai)
from zchat.db import session_scope, upgrade_db
from zchat.models import (Attachment, Conversation, LoreEntry, LoreVersion, Message, Scenario,
                          ScenarioLore, utcnow)

upgrade_db()  # apply any pending Alembic migrations (creates data/zchat.db on first run)

# Changes whenever app.css or app.js is edited, so browsers fetch the new file instead of a cached one.
ASSET_VERSION = max(int(f.stat().st_mtime) for f in (config.ROOT / "static").glob("app.*"))

# Light/dark: the choice saved by the sidebar toggle (see toggleTheme in app.js), otherwise the OS setting.
THEME_SCRIPT = """
let theme = null;
try { theme = localStorage.getItem("theme"); } catch (e) {}
if (theme !== "light" && theme !== "dark") {
  theme = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}
document.documentElement.dataset.theme = theme;
"""

background_tasks = []  # keep a reference so the task isn't garbage-collected


async def start_price_refresh():
    """Re-read Z.ai's pricing page in the background (at most once a day)."""
    background_tasks.append(asyncio.create_task(pricing.refresh_forever()))


async def warm_up_embeddings():
    """Load the lorebook's embedding model in the background (downloading it on first run)."""
    if not config.LORE_VECTORS:
        return

    async def load():
        try:
            await asyncio.to_thread(embeddings.warm_up)
        except Exception as e:  # no network on first run, etc.: the lorebook falls back to word search
            print(f"zchat: couldn't load the lorebook's embedding model ({e}); using word search only.")

    background_tasks.append(asyncio.create_task(load()))


app, rt = fast_app(
    pico=False,
    static_path="static",
    on_startup=[start_price_refresh, warm_up_embeddings],
    hdrs=(
        Meta(name="viewport", content="width=device-width, initial-scale=1"),
        Link(rel="icon", href="/favicon.svg", type="image/svg+xml"),
        Link(rel="preconnect", href="https://fonts.googleapis.com"),
        Link(rel="stylesheet",
             href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap"),
        Link(rel="stylesheet", href=f"/app.css?v={ASSET_VERSION}"),
        Script(THEME_SCRIPT),  # inline and not deferred, so the page never flashes the wrong theme
        Script(src=f"/app.js?v={ASSET_VERSION}", defer=True),
    ),
)


# ---------------------------------------------------------------- helpers


def recent_conversations(s: Session) -> list[Conversation]:
    return list(s.scalars(select(Conversation).order_by(Conversation.updated_at.desc()).limit(100)))


def all_scenarios(s: Session) -> list[Scenario]:
    return list(s.scalars(select(Scenario).order_by(Scenario.name)))


def search_scenarios(s: Session, q: str) -> list[Scenario]:
    """Scenarios where every word of `q` appears in the name, categories, description or prompt text."""
    query = select(Scenario).order_by(Scenario.name)
    for word in q.split():
        # Escape LIKE's wildcards so searching for "100%" or "_" finds those characters literally.
        word = word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{word}%"
        query = query.where(or_(Scenario.name.ilike(pattern, escape="\\"),
                                Scenario.categories.ilike(pattern, escape="\\"),
                                Scenario.description.ilike(pattern, escape="\\"),
                                Scenario.system_prompt.ilike(pattern, escape="\\")))
    return list(s.scalars(query))


def known_categories(s: Session) -> list[str]:
    """The common starter categories plus every category already used on a scenario."""
    names = list(category.COMMON)
    for text in s.scalars(select(Scenario.categories)):
        for name in category.split(text):
            if name.lower() not in [n.lower() for n in names]:
                names.append(name)
    return names


def sidebar(s: Session, active: str = "chats", conv_id: int | None = None):
    return ui.sidebar(recent_conversations(s), active, conv_id, chat.usage_for(s))


def branch_info(s: Session, m: Message) -> ui.Branch:
    sibs = chat.siblings(s, m)
    ids = [x.id for x in sibs]
    i = ids.index(m.id)
    return ui.Branch(index=i + 1, count=len(ids),
                     prev_id=ids[i - 1] if i > 0 else None,
                     next_id=ids[i + 1] if i + 1 < len(ids) else None)


def render_message(s: Session, m: Message):
    return ui.message_view(m, branch_info(s, m))


def render_lore_note(s: Session, m: Message):
    """The "Lorebook updated" note after a reply, with the versions that update saved."""
    versions = s.scalars(select(LoreVersion).where(LoreVersion.message_id == m.id).order_by(LoreVersion.id))
    return lore_ui.lore_note_view(m, list(versions))


def render_messages(s: Session, conv: Conversation) -> list:
    """The active branch, with the "Story so far" box right after the message it summarizes up to,
    and a note after each reply where the lorebook updated itself."""
    path = chat.active_path(s, conv)
    checkpoint = chat.latest_summary(path)
    out = []
    for m in path:
        out.append(render_message(s, m))
        if m.lore_note:
            out.append(render_lore_note(s, m))
        if m is checkpoint:
            out.append(ui.summary_view(m))
    return out


def render_chat_lore(s: Session, conv: Conversation, oob: bool = False):
    """The chat's lorebook list, as the active branch sees it."""
    return lore_ui.chat_lore_list(conv.id, lore.current_versions(s, conv.id, chat.active_path(s, conv)), oob=oob)


def render_stats(s: Session, conv: Conversation, oob: bool = False):
    used, model = chat.context_used(chat.active_path(s, conv))
    return ui.stats_view(conv, chat.usage_for(s, conv.id), used, model, oob=oob)


def get_conv(s: Session, cid: int) -> Conversation:
    conv = s.get(Conversation, cid)
    if conv is None:
        raise LookupError(f"Chat {cid} not found")
    return conv


async def read_uploads(req) -> tuple[list[documents.Processed], list[str]]:
    """Process every file in the multipart form. Returns (processed files, error messages)."""
    form = await req.form()
    processed, errors = [], []
    for f in form.getlist("files"):
        if not getattr(f, "filename", None):  # browsers send an empty part when no file is chosen
            continue
        try:
            processed.append(await documents.process_upload(f.filename, await f.read()))
        except documents.UnsupportedFile as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{f.filename}: could not read file ({e})")
    return processed, errors


def error_toast(errors: list[str]):
    """An error popup. It's an out-of-band swap, so it lands in #toasts whatever the request's target was."""
    if not errors:
        return None
    return Div(Div(*[P(e) for e in errors], cls="toast error-box", onclick="this.remove()"),
               hx_swap_oob="beforeend:#toasts")


def info_toast(messages: list[str]):
    """Like error_toast, but neutral, and it fades out by itself."""
    return Div(Div(*[P(m) for m in messages], cls="toast toast-info", onclick="this.remove()"),
               hx_swap_oob="beforeend:#toasts")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------- pages


@app.get("/")
def index(scenario: int | None = None):
    with session_scope() as s:
        chosen = s.get(Scenario, scenario) if scenario else None
        return ui.page("New chat", ui.new_chat_page(all_scenarios(s), chosen), sidebar=sidebar(s))


@app.get("/c/{cid}")
def conversation_page(cid: int):
    with session_scope() as s:
        conv = s.get(Conversation, cid)
        if conv is None:
            return Redirect("/")
        header = ui.chat_header(conv, render_stats(s, conv))
        drawer = lore_ui.chat_lore_drawer(cid, lore.current_versions(s, cid, chat.active_path(s, conv)))
        return ui.page(conv.title, ui.chat_page(conv, header, render_messages(s, conv), drawer),
                       sidebar=sidebar(s, conv_id=cid))


@app.get("/scenario_prompt")
def scenario_prompt(scenario_id: str = ""):
    """Fill the system-prompt box when a scenario is picked on the new-chat page."""
    with session_scope() as s:
        sc = s.get(Scenario, int(scenario_id)) if scenario_id.isdigit() else None
        return ui.system_prompt_area(sc.system_prompt if sc else "")


# ---------------------------------------------------------------- sending messages


@app.post("/new")
async def new_chat(req):
    form = await req.form()
    content = (form.get("content") or "").strip()
    image_mode = form.get("mode") == "image"
    files, errors = await read_uploads(req)
    if not content and not files:
        return error_toast(["Type a message or attach a file to start."])
    scenario_id = form.get("scenario_id") or ""
    model = form.get("model") or config.DEFAULT_MODEL
    with session_scope() as s:
        conv = Conversation(model=model if config.is_available(model) else config.DEFAULT_MODEL,
                            system_prompt=(form.get("system_prompt") or "").strip(),
                            scenario_id=int(scenario_id) if scenario_id.isdigit() else None,
                            thinking=bool(form.get("thinking")),
                            reasoning_effort=form.get("reasoning_effort") or config.DEFAULT_EFFORT)
        s.add(conv)
        s.flush()
        scenario = s.get(Scenario, conv.scenario_id) if conv.scenario_id else None
        if scenario:
            lore.seed_chat(s, conv, scenario)  # the chat gets its own copy of the scenario's lorebook
        user, _reply = chat.send_user_message(s, conv, content, image_mode=image_mode)
        for p in files:
            chat.add_attachment(s, user, p)
        cid = conv.id
    # Errors (e.g. an unsupported file) can't be shown after a redirect, so the chat just starts without it.
    return Redirect(f"/c/{cid}")


@app.post("/send")
async def send(req, cid: int):
    form = await req.form()
    content = (form.get("content") or "").strip()
    image_mode = form.get("mode") == "image"
    files, errors = await read_uploads(req)
    if not content and not files:
        return error_toast(errors or ["Nothing to send."])
    with session_scope() as s:
        conv = get_conv(s, cid)
        user, reply = chat.send_user_message(s, conv, content, image_mode=image_mode)
        for p in files:
            chat.add_attachment(s, user, p)
        s.flush()
        s.refresh(user)
        return render_message(s, user), render_message(s, reply), error_toast(errors)


@app.post("/illustrate")
def illustrate(cid: int):
    with session_scope() as s:
        conv = get_conv(s, cid)
        user, reply = chat.send_user_message(s, conv, "", kind="illustrate", image_mode=True)
        return render_message(s, user), render_message(s, reply)


@app.get("/stream")
async def stream(mid: int):
    """Server-sent events for one pending reply. app.js listens and updates the bubble."""

    async def events():
        async for name, data in generate.run(mid):
            if name in ("delta", "reasoning", "status"):
                yield sse(name, data)
            elif name == "summary":
                with session_scope() as s:
                    html = to_xml(ui.summary_view(s.get(Message, data["mid"])))
                yield sse("summary", {"mid": data["mid"], "html": html})
            elif name == "lore":
                with session_scope() as s:
                    noted = s.get(Message, data["mid"])
                    yield sse("lore", {"mid": data["mid"], "html": to_xml(render_lore_note(s, noted)),
                                       "drawer": to_xml(render_chat_lore(s, noted.conversation))})
            elif name == "title":
                with session_scope() as s:
                    conv = s.get(Message, mid).conversation
                    yield sse("sidebar", {"html": to_xml(ui.conv_list(recent_conversations(s), conv.id)),
                                          "title": conv.title})
            else:  # "done" or "error": send the final rendered bubble and updated token stats
                with session_scope() as s:
                    m = s.get(Message, mid)
                    yield sse("finished", {"html": to_xml(render_message(s, m)),
                                           "stats": to_xml(render_stats(s, m.conversation)),
                                           "total": to_xml(ui.total_usage(chat.usage_for(s)))})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/stop")
def stop(mid: int):
    generate.STOP.add(mid)
    return Response(status_code=204)


# ---------------------------------------------------------------- editing, branching, forking


@app.get("/msg")
def message_fragment(mid: int):
    with session_scope() as s:
        return render_message(s, s.get(Message, mid))


@app.get("/edit_form")
def edit_form(mid: int):
    with session_scope() as s:
        return ui.edit_form(s.get(Message, mid))


@app.post("/edit")
def edit(mid: int, content: str = ""):
    with session_scope() as s:
        m = s.get(Message, mid)
        if m.role == "assistant":
            m.content = content  # in-place: lets you steer a story by rewriting the narrator
            return render_message(s, m)
        chat.edit_user_message(s, m, content.strip())
        return render_messages(s, m.conversation)


@app.post("/regen")
def regen(mid: int):
    with session_scope() as s:
        m = s.get(Message, mid)
        chat.regenerate(s, m)
        return render_messages(s, m.conversation)


@app.get("/switch")
def switch(mid: int):
    with session_scope() as s:
        m = s.get(Message, mid)
        chat.switch_to(s, m)
        return (render_messages(s, m.conversation), render_stats(s, m.conversation, oob=True),
                render_chat_lore(s, m.conversation, oob=True))  # the other branch may know different things


@app.post("/fork")
def fork(mid: int):
    with session_scope() as s:
        m = s.get(Message, mid)
        new = chat.fork(s, m.conversation, m)
        new_id = new.id
    return Redirect(f"/c/{new_id}")


# ---------------------------------------------------------------- story so far (see zchat/story.py)


@app.post("/summarize")
async def summarize_now(cid: int):
    """The "Summarize" button: summarize older messages now instead of waiting for the context to fill."""
    with session_scope() as s:
        leaf_id = get_conv(s, cid).active_leaf_id
    try:
        summary_on = await story.summarize_branch(leaf_id)
    except Exception as e:
        return error_toast([f"Couldn't summarize: {e}"])
    if summary_on is None:
        return info_toast([f"Nothing new to summarize: the newest {config.SUMMARY_KEEP_RECENT} messages "
                           "are always sent word for word."])
    with session_scope() as s:
        conv = get_conv(s, cid)
        return render_messages(s, conv), info_toast(["Summarized. Older messages are now sent as the "
                                                     "Story so far."])


@app.get("/summary")
def summary_view(mid: int):
    with session_scope() as s:
        return ui.summary_view(s.get(Message, mid))


@app.get("/summary_form")
def summary_form(mid: int):
    with session_scope() as s:
        return ui.summary_form(s.get(Message, mid))


@app.post("/summary")
def summary_save(mid: int, summary: str = ""):
    with session_scope() as s:
        m = s.get(Message, mid)
        if summary.strip():  # an empty box is ignored; the Remove button is for removing
            m.summary = summary.strip()
        return ui.summary_view(m)


@app.post("/summary_remove")
def summary_remove(mid: int):
    """Drop a summary: the messages it covered are sent in full again (or an older summary takes over)."""
    with session_scope() as s:
        m = s.get(Message, mid)
        m.summary = ""
        return render_messages(s, m.conversation)


# ---------------------------------------------------------------- chat settings


@app.post("/settings")
def settings(cid: int, model: str = "", thinking: str = "", reasoning_effort: str = ""):
    with session_scope() as s:
        conv = get_conv(s, cid)
        # The form only contains the control that was on screen, which matches the *previous* model:
        # an effort dropdown for GLM-5.3-style models, a thinking switch for the rest.
        if config.uses_effort(config.usable_model(conv.model)):
            if reasoning_effort in config.EFFORT_LEVELS:
                conv.reasoning_effort = reasoning_effort
        else:
            conv.thinking = bool(thinking)  # an unchecked checkbox isn't sent at all
        if config.is_available(model):
            conv.model = model
        # Re-render the control (and price) in case the model changed.
        shown = config.usable_model(conv.model)
        return (render_stats(s, conv, oob=True),
                ui.reasoning_control(shown, conv.thinking, conv.reasoning_effort, oob=True),
                ui.model_price(shown, oob=True))


@app.get("/reasoning_control")
def reasoning_control(model: str = ""):
    """New-chat page: swap the thinking switch / effort dropdown (and the price) when the model changes."""
    return ui.reasoning_control(model, False, config.DEFAULT_EFFORT), ui.model_price(model, oob=True)


# The last Coding Plan numbers from Z.ai. Page loads always fetch new ones; after that, config.yaml
# decides when to ask again.
plan_cache = {"plan": None, "fetched_at": None, "replies": 0}


def finished_replies(s: Session) -> int:
    return s.scalar(select(func.count(Message.id)).where(Message.role == "assistant", Message.status == "done"))


def plan_refresh_due(replies_now: int) -> bool:
    if plan_cache["plan"] is None:
        return True
    if config.PLAN_REFRESH_BY == "seconds":
        return datetime.now() - plan_cache["fetched_at"] >= timedelta(seconds=config.PLAN_EVERY_SECONDS)
    return replies_now - plan_cache["replies"] >= config.PLAN_EVERY_REPLIES


@app.get("/plan_usage")
async def plan_usage(fresh: bool = False):
    """Sidebar meters for the Coding Plan's 5-hour and weekly limits.

    fresh=True (a page load or reload) always asks Z.ai; otherwise config.yaml decides.
    """
    if not config.USING_CODING_PLAN:
        return ""
    with session_scope() as s:
        replies_now = finished_replies(s)
    if fresh or plan_refresh_due(replies_now):
        try:
            plan_cache.update(plan=await zai.plan_usage(), fetched_at=datetime.now(), replies=replies_now)
        except (zai.ZaiError, httpx.HTTPError, KeyError, ValueError) as e:
            if plan_cache["plan"] is None:
                return ui.plan_usage(None, error=str(e))
            # otherwise keep showing the last good numbers
    return ui.plan_usage(plan_cache["plan"], updated=plan_cache["fetched_at"])


@app.post("/system")
def set_system(cid: int, system_prompt: str = ""):
    with session_scope() as s:
        get_conv(s, cid).system_prompt = system_prompt.strip()
    return Span("Saved. Applies to the next reply.")


@app.post("/save_scenario")
def save_scenario_from_chat(cid: int, system_prompt: str = "", name: str = ""):
    with session_scope() as s:
        conv = get_conv(s, cid)
        sc = Scenario(name=name.strip() or conv.title, system_prompt=system_prompt.strip())
        s.add(sc)
        s.flush()
        conv.scenario_id = sc.id
        conv.system_prompt = sc.system_prompt
        return Span(f'Saved as scenario "{sc.name}".')


@app.post("/rename")
def rename(cid: int, title: str = ""):
    with session_scope() as s:
        get_conv(s, cid).title = title.strip()[:200] or "Untitled"
        return ui.conv_list(recent_conversations(s), cid, oob=True)


@app.post("/delete_conv")
def delete_conv(req, cid: int):
    with session_scope() as s:
        conv = s.get(Conversation, cid)
        if conv:
            s.delete(conv)
    # If you deleted the chat you're looking at, go home; otherwise just remove it from the list.
    if req.headers.get("HX-Current-URL", "").rstrip("/").endswith(f"/c/{cid}"):
        return Redirect("/")
    return ""


# ---------------------------------------------------------------- scenarios


@app.get("/scenarios")
def scenarios(q: str = ""):
    """The scenarios page. The search box asks for this same page with ?q=... and swaps in just the cards."""
    q = q.strip()
    with session_scope() as s:
        found = search_scenarios(s, q) if q else all_scenarios(s)
        return ui.page("Scenarios", ui.scenarios_page(found, q), sidebar=sidebar(s, "scenarios"))


@app.get("/scenarios/edit")
def scenario_edit(sid: int | None = None):
    with session_scope() as s:
        sc = s.get(Scenario, sid) if sid else None
        editor = ui.scenario_editor(sc, known_categories(s), lore_ui.scenario_lore_section(sc))
        return ui.page("Scenario", editor, sidebar=sidebar(s, "scenarios"))


@app.post("/scenarios/save")
async def scenario_save(name: str, system_prompt: str = "", description: str = "", categories: str = "",
                        sid: str = ""):
    name, description, system_prompt = name.strip(), description.strip(), system_prompt.strip()
    with session_scope() as s:
        known = known_categories(s)
    chosen = category.clean(categories, known)
    if not chosen and (description or system_prompt):
        # Left empty: let the AI pick (if it fails, the scenario is just saved without categories).
        chosen = await category.suggest(name, description, system_prompt, known)
    with session_scope() as s:
        sc = s.get(Scenario, int(sid)) if sid.isdigit() else None
        if sc is None:
            sc = Scenario()
            s.add(sc)
        sc.name, sc.description, sc.system_prompt = name, description, system_prompt
        sc.categories = chosen
        sc.updated_at = utcnow()
    return Redirect("/scenarios")


@app.post("/scenarios/clone")
def scenario_clone(sid: int):
    """Copy a scenario and open the copy in the editor, ready to tweak."""
    with session_scope() as s:
        original = s.get(Scenario, sid)
        if original is None:
            return Redirect("/scenarios")
        copy = Scenario(name=f"{original.name} (copy)"[:200], description=original.description,
                        categories=original.categories, system_prompt=original.system_prompt)
        copy.lore = [ScenarioLore(name=e.name, keys=e.keys, text=e.text, enabled=e.enabled) for e in original.lore]
        s.add(copy)
        s.flush()
        copy_id = copy.id
    return Redirect(f"/scenarios/edit?sid={copy_id}")


@app.post("/scenarios/delete")
def scenario_delete(sid: int):
    with session_scope() as s:
        sc = s.get(Scenario, sid)
        if sc:
            s.delete(sc)
    return Redirect("/scenarios")


@app.get("/scenarios/export")
def scenario_export(req):
    """Download scenarios as JSON: ?sid=3 gives one object, ?sid=3&sid=5 a list, and no sid a list of all."""
    ids = [int(x) for x in req.query_params.getlist("sid") if x.isdigit()]
    with session_scope() as s:
        if ids:
            chosen = list(s.scalars(select(Scenario).where(Scenario.id.in_(ids)).order_by(Scenario.name)))
        else:
            chosen = all_scenarios(s)
        if not chosen:
            return Response("No scenarios to export", status_code=404)
        if len(ids) == 1:
            data, filename = scenario_json.to_dict(chosen[0]), scenario_json.filename_for(chosen[0].name)
        else:
            data, filename = [scenario_json.to_dict(sc) for sc in chosen], "zchat-scenarios.json"
    return Response(scenario_json.dumps(data), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/scenarios/import")
async def scenario_import(req):
    """Import one or more .json files (each holding one scenario or a list). Re-renders the scenarios page."""
    form = await req.form()
    imported, errors = [], []
    with session_scope() as s:
        taken = {sc.name for sc in all_scenarios(s)}
        for f in form.getlist("files"):
            if not getattr(f, "filename", None):
                continue
            try:
                items = scenario_json.parse(await f.read())
            except ValueError as e:
                errors.append(f"{f.filename}: {e}")
                continue
            for item in items:
                item["name"] = scenario_json.unique_name(item["name"], taken)
                taken.add(item["name"])
                lorebook = item.pop("lorebook")
                sc = Scenario(**item)
                sc.lore = [ScenarioLore(**entry) for entry in lorebook]
                s.add(sc)
                imported.append(item["name"])
        s.flush()
        page = ui.scenarios_page(all_scenarios(s))

    notice = None
    if imported:
        count = f"{len(imported)} scenario" + ("s" if len(imported) != 1 else "")
        notice = info_toast([f"Imported {count}: " + ", ".join(imported)])
    return page, notice, error_toast(errors)


# ---------------------------------------------------------------- scenario lorebooks (see zchat/lore.py)


def get_scenario(s: Session, sid: int) -> Scenario:
    sc = s.get(Scenario, sid)
    if sc is None:
        raise LookupError(f"Scenario {sid} not found")
    return sc


@app.get("/lore/scenario/list")
def scenario_lore_list(sid: int):
    with session_scope() as s:
        return lore_ui.scenario_lore_list(get_scenario(s, sid))


@app.get("/lore/scenario/form")
def scenario_lore_form(sid: int, lid: int | None = None):
    with session_scope() as s:
        entry = s.get(ScenarioLore, lid) if lid else None
        return lore_ui.scenario_lore_form(get_scenario(s, sid), entry)


@app.post("/lore/scenario/save")
def scenario_lore_save(sid: int, name: str, text: str = "", keys: str = "", enabled: str = "", lid: str = ""):
    with session_scope() as s:
        sc = get_scenario(s, sid)
        entry = s.get(ScenarioLore, int(lid)) if lid.isdigit() else None
        if entry is None or entry.scenario_id != sc.id:
            entry = ScenarioLore()
            sc.lore.append(entry)
        entry.name = scenario_json.clean(name, 200, one_line=True) or "Untitled"
        entry.keys = scenario_json.clean(keys, 500, one_line=True)
        entry.text = scenario_json.clean(text, 20_000)
        entry.enabled = bool(enabled)  # an unchecked checkbox isn't sent at all
        s.flush()
        return lore_ui.scenario_lore_list(sc)


@app.post("/lore/scenario/delete")
def scenario_lore_delete(lid: int):
    with session_scope() as s:
        entry = s.get(ScenarioLore, lid)
        if entry is None:  # already gone (e.g. deleted in another tab)
            return Response("Not found", status_code=404)
        sc = entry.scenario
        s.delete(entry)
        s.flush()
        s.refresh(sc)
        return lore_ui.scenario_lore_list(sc)


@app.post("/lore/scenario/import")
async def scenario_lore_import(req, sid: int):
    """Add the entries from one or more lorebook files. Existing entries are never overwritten."""
    form = await req.form()
    added, errors = 0, []
    with session_scope() as s:
        sc = get_scenario(s, sid)
        for f in form.getlist("files"):
            if not getattr(f, "filename", None):
                continue
            try:
                entries = lore_import.parse(await f.read())
            except ValueError as e:
                errors.append(f"{f.filename}: {e}")
                continue
            sc.lore.extend(ScenarioLore(**entry) for entry in entries)
            added += len(entries)
        s.flush()
        notice = info_toast([f"Imported {lore_ui.entry_count(added)}.", lore_import.IMPORT_NOTE]) if added else None
        return lore_ui.scenario_lore_list(sc), notice, error_toast(errors)


@app.post("/lore/scenario/draft")
async def scenario_lore_draft(sid: int):
    """"Draft from prompt": the AI writes starter entries from the system prompt. Names already there are skipped."""
    with session_scope() as s:
        system_prompt = get_scenario(s, sid).system_prompt.strip()
    if not system_prompt:
        with session_scope() as s:
            return lore_ui.scenario_lore_list(get_scenario(s, sid)), error_toast(
                ["Write a system prompt first: the draft is made from it."])
    try:
        drafted = await lore_update.draft_from_prompt(system_prompt, config.DEFAULT_MODEL)
    except Exception as e:
        with session_scope() as s:
            return lore_ui.scenario_lore_list(get_scenario(s, sid)), error_toast([f"Couldn't draft a lorebook: {e}"])
    with session_scope() as s:
        sc = get_scenario(s, sid)
        taken = {e.name.lower() for e in sc.lore}
        added = 0
        for entry in drafted:
            if entry["name"].lower() in taken:
                continue
            sc.lore.append(ScenarioLore(name=entry["name"], keys=entry["keys"], text=entry["text"]))
            taken.add(entry["name"].lower())
            added += 1
        s.flush()
        notice = info_toast([f"Drafted {lore_ui.entry_count(added)} from the system prompt. "
                             "Edit or delete any you don't want."])
        return lore_ui.scenario_lore_list(sc), notice


@app.get("/lore/scenario/export")
def scenario_lore_export(sid: int):
    with session_scope() as s:
        sc = get_scenario(s, sid)
        data = lore_import.to_export(sc.name, sc.lore)
        filename = scenario_json.filename_for(sc.name).removesuffix(".json") + "-lorebook.json"
    return Response(scenario_json.dumps(data), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ---------------------------------------------------------------- chat lorebooks


@app.get("/lore/chat/list")
def chat_lore_list(cid: int):
    with session_scope() as s:
        return render_chat_lore(s, get_conv(s, cid))


@app.get("/lore/chat/form")
def chat_lore_form(cid: int, vid: int | None = None):
    with session_scope() as s:
        version = s.get(LoreVersion, vid) if vid else None
        return lore_ui.chat_lore_form(cid, version)


@app.post("/lore/chat/save")
def chat_lore_save(cid: int, name: str, text: str = "", keys: str = "", enabled: str = "", vid: str = ""):
    """Edit the version shown (in place, like editing "Story so far"), or add a new entry for every branch."""
    with session_scope() as s:
        conv = get_conv(s, cid)
        version = s.get(LoreVersion, int(vid)) if vid.isdigit() else None
        if version is None or version.entry.conversation_id != conv.id:
            version = LoreVersion()
            s.add(LoreEntry(conversation_id=conv.id, versions=[version]))
        version.name = scenario_json.clean(name, 200, one_line=True) or "Untitled"
        version.keys = scenario_json.clean(keys, 500, one_line=True)
        version.text = scenario_json.clean(text, 20_000)
        version.enabled = bool(enabled)  # an unchecked checkbox isn't sent at all
        version.embedding = None  # embedded again (from the new text) before the next search
        s.flush()
        return render_chat_lore(s, conv)


@app.post("/lore/chat/delete")
def chat_lore_delete(vid: int):
    """Delete the whole entry (every version, on every branch)."""
    with session_scope() as s:
        version = s.get(LoreVersion, vid)
        if version is None:  # already gone (e.g. deleted in another tab)
            return Response("Not found", status_code=404)
        conv = version.entry.conversation
        s.delete(version.entry)
        s.flush()
        return render_chat_lore(s, conv)


@app.post("/lore/undo")
def lore_undo(mid: int):
    """Undo the lorebook update noted on reply `mid`."""
    with session_scope() as s:
        m = s.get(Message, mid)
        lore_update.undo_update(s, m)
        return render_messages(s, m.conversation), render_chat_lore(s, m.conversation, oob=True)


# ---------------------------------------------------------------- files


@app.get("/media/{aid}")
def media(aid: int):
    with session_scope() as s:
        a = s.get(Attachment, aid)
        if a is None:
            return Response("Not found", status_code=404)
        path = documents.abs_path(a.path)
        mime, name = a.mime, a.filename
    if not path.exists():
        return Response("File missing on disk", status_code=404)
    disposition = "inline" if mime.startswith(("image/", "text/", "application/pdf")) else "attachment"
    return FileResponse(path, media_type=mime, filename=name, content_disposition_type=disposition)


# 127.0.0.1: only this computer can open the app. The default (0.0.0.0) would let anyone on the same
# network (e.g. café Wi-Fi) use it, and your API key, since zchat has no login. It also makes uvicorn
# print a link that works; http://0.0.0.0:5001 doesn't open in a browser on Windows.
serve(host="127.0.0.1")
