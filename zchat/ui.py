"""HTML components (FastHTML FT functions). No database access here — routes pass data in."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import quote

from fasthtml.common import (A, Aside, Button, Details, Div, Form, H1, H2, Header, Img, Input, Label, Main,
                             NotStr, Option, P, Select, Small, Span, Summary, Textarea, Title)
from markdown_it import MarkdownIt

from . import category, config, pricing
from .chat import Usage
from .models import Attachment, Conversation, Message, Scenario
from .zai import PlanUsage

# html=False: raw HTML in model output is escaped, not rendered.
_md = MarkdownIt("commonmark", {"html": False, "breaks": True, "linkify": False}).enable(["table", "strikethrough"])


def markdown(text: str):
    return NotStr(_md.render(text or ""))


# Small inline SVG icons (Lucide-style strokes).
_ICON_PATHS = {
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "send": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "clip": '<path d="M21 11.5 12.5 20a5 5 0 0 1-7-7L14 4.5a3.5 3.5 0 0 1 5 5L10.5 18a2 2 0 0 1-3-3L15 7.5"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/>',
    "wand": '<path d="m15 4 1 2 2 1-2 1-1 2-1-2-2-1 2-1zM5 19 16 8M19 14l.7 1.3L21 16l-1.3.7L19 18l-.7-1.3L17 16l1.3-.7z"/>',
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16zM14 6l4 4"/>',
    "regen": '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
    "fork": '<circle cx="6" cy="5" r="2"/><circle cx="18" cy="5" r="2"/><circle cx="12" cy="19" r="2"/>'
            '<path d="M6 7v2a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3V7M12 12v5"/>',
    "copy": '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.7A8 8 0 1 1 21 12z"/>',
    "book": '<path d="M4 4h10a4 4 0 0 1 4 4v12H8a4 4 0 0 1-4-4z"/><path d="M4 16a4 4 0 0 1 4-4h10"/>',
    "menu": '<path d="M4 6h16M4 12h16M4 18h16"/>',
    "left": '<path d="m15 6-6 6 6 6"/>',
    "right": '<path d="m9 6 6 6-6 6"/>',
    "file": '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4"/>',
    "save": '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v5h7M8 21v-7h8v7"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4'
           'M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/>',
    "download": '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
    "upload": '<path d="M12 20V9M7 14l5-5 5 5M5 4h14"/>',
}


def icon(name: str, size: int = 18):
    return NotStr(f'<svg class="icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                  f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                  f'{_ICON_PATHS[name]}</svg>')


def fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1000:.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def fmt_cost(u: Usage) -> str:
    if config.USING_CODING_PLAN:
        return "Coding Plan"  # subscription: no per-token dollars to show
    text = f"${u.cost:.4f}" if u.cost < 1 else f"${u.cost:.2f}"
    return text if u.cost_known else text + "+?"


# ---------------------------------------------------------------- page shell


def page(title: str, *content, sidebar):
    return (Title(f"{title} · zchat"),
            Div(sidebar,
                Div(cls="scrim", onclick="toggleSidebar(false)"),
                Main(Button(icon("menu"), cls="icon-btn menu-btn", onclick="toggleSidebar()", title="Menu"),
                     *content, cls="main"),
                Div(id="toasts", cls="toasts"),
                cls="app"))


def sidebar(convs: list[Conversation], active: str, conv_id: int | None, total: Usage):
    return Aside(
        Div(A(Span("z", cls="logo"), "chat", href="/", cls="brand"),
            Button(Span(icon("moon", 17), cls="icon-moon"), Span(icon("sun", 17), cls="icon-sun"),
                   cls="icon-btn theme-toggle", onclick="toggleTheme()", title="Switch light / dark"),
            cls="sidebar-top"),
        A(icon("plus"), "New chat", href="/", cls="btn btn-primary new-chat-btn"),
        Div(A(icon("chat"), "Chats", href="/", cls="nav-link" + (" active" if active == "chats" else "")),
            A(icon("book"), "Scenarios", href="/scenarios",
              cls="nav-link" + (" active" if active == "scenarios" else "")),
            cls="nav"),
        conv_list(convs, conv_id),
        plan_usage_placeholder() if config.USING_CODING_PLAN else None,
        total_usage(total),
        id="sidebar", cls="sidebar")


def _plan_refresh():
    """The sidebar loads the meters after the page renders and re-asks after each reply ("refresh-plan" is
    fired by app.js). The server decides from config.yaml whether that actually calls Z.ai again."""
    trigger = "refresh-plan from:body"
    if config.PLAN_REFRESH_BY == "seconds":
        trigger += f", every {config.PLAN_EVERY_SECONDS}s"
    return dict(hx_get="/plan_usage", hx_swap="outerHTML", hx_trigger=trigger)


def plan_usage_placeholder():
    # fresh=1: every page load or reload asks Z.ai for current numbers
    return Div(Small("Plan usage…", cls="muted"), id="plan-usage", cls="plan-usage",
               hx_get="/plan_usage?fresh=1", hx_swap="outerHTML", hx_trigger="load")


def _reset_text(when: datetime) -> str:
    if when - datetime.now() < timedelta(hours=24):
        return "resets " + f"{when:%I:%M %p}".lstrip("0")
    return f"resets {when:%a %b} {when.day}"


def plan_usage(plan: PlanUsage | None, error: str = "", updated: datetime | None = None):
    """Coding Plan quota meters for the sidebar. `plan` is None when the API call failed."""
    if plan is None:
        return Div(Small("Plan usage unavailable", cls="muted"), id="plan-usage", cls="plan-usage",
                   title=error, **_plan_refresh())

    rows = []
    for limit in plan.limits:
        level = "high" if limit.percent >= 80 else ""
        rows.append(Div(
            Div(Span(limit.label), Span(f"{limit.percent}%"), cls="plan-row-head"),
            Div(Div(cls="meter-fill " + level, style=f"width:{min(limit.percent, 100)}%"), cls="meter"),
            Small(f"{limit.used:,} / {limit.total:,} · {_reset_text(limit.resets_at)}", cls="muted"),
            cls="plan-row"))

    week_total = sum(plan.week_tokens.values())
    per_model = " · ".join(f"{name} {fmt(n)}" for name, n in plan.week_tokens.items())
    return Div(
        Div(Span("Coding Plan"), Span(plan.level.title(), cls="plan-level"), cls="plan-row-head"),
        *rows,
        Small(f"Last 7 days: {fmt(week_total)} tokens", cls="muted", title=per_model) if week_total else None,
        id="plan-usage", cls="plan-usage",
        title=f"Updated {updated:%I:%M %p}".replace(" 0", " ") if updated else None, **_plan_refresh())


def total_usage(total: Usage):
    """All-time token/cost counter at the bottom of the sidebar."""
    return Div(Span("All-time"), Span(f"{fmt(total.total)} tokens · {fmt_cost(total)}"),
               id="total-usage", cls="sidebar-foot",
               title=f"in {total.prompt:,} (cached {total.cached:,}) · out {total.completion:,} · {total.images} images")


def conv_list(convs: list[Conversation], conv_id: int | None, oob: bool = False):
    items = []
    for c in convs:
        items.append(Div(
            A(Span(c.title, cls="conv-title"),
              Span(c.scenario.name, cls="conv-tag") if c.scenario else None,
              href=f"/c/{c.id}", cls="conv-link"),
            Button(icon("trash", 15), cls="icon-btn conv-del", title="Delete chat",
                   hx_post=f"/delete_conv?cid={c.id}", hx_confirm=f'Delete "{c.title}"?',
                   hx_target="closest .conv-item", hx_swap="outerHTML"),
            cls="conv-item" + (" active" if c.id == conv_id else "")))
    if not items:
        items = [P("No chats yet.", cls="muted small pad")]
    return Div(*items, id="conv-list", cls="conv-list", hx_swap_oob="true" if oob else None)


# ---------------------------------------------------------------- messages


@dataclass
class Branch:
    """Position of a message among its siblings (alternate versions)."""
    index: int
    count: int
    prev_id: int | None
    next_id: int | None


def attachment_view(a: Attachment):
    url = f"/media/{a.id}"
    if a.kind == "generated":
        return A(Img(src=url, alt=a.filename, loading="lazy"), href=url, target="_blank", cls="gen-image")
    if a.kind == "image":
        return A(Img(src=url, alt=a.filename, loading="lazy"), href=url, target="_blank", cls="thumb")
    detail = f"{len(a.text):,} chars" + (f" · {a.note}" if a.note else "")
    return A(icon("file", 16), Span(a.filename, cls="chip-name"), Small(detail), href=url, target="_blank",
             cls="chip", title=a.note or a.filename)


def branch_nav(b: Branch | None):
    if b is None or b.count < 2:
        return None
    return Div(
        Button(icon("left", 14), cls="icon-btn", disabled=b.prev_id is None, title="Previous version",
               hx_get=f"/switch?mid={b.prev_id}", hx_target="#messages"),
        Span(f"{b.index}/{b.count}"),
        Button(icon("right", 14), cls="icon-btn", disabled=b.next_id is None, title="Next version",
               hx_get=f"/switch?mid={b.next_id}", hx_target="#messages"),
        cls="branch-nav")


def usage_line(m: Message):
    if m.role != "assistant" or m.status in ("pending", "streaming"):
        return None
    parts = [config.model_info(m.model).label if m.model else ""]
    if m.prompt_tokens or m.completion_tokens:
        parts.append(f"in {fmt(m.prompt_tokens)}")
        if m.cached_tokens:
            parts.append(f"cached {fmt(m.cached_tokens)}")
        parts.append(f"out {fmt(m.completion_tokens)}")
    return Span(" · ".join(p for p in parts if p), cls="usage",
                title=f"prompt {m.prompt_tokens:,} · cached {m.cached_tokens:,} · "
                      f"completion {m.completion_tokens:,} (reasoning {m.reasoning_tokens:,})")


def lore_used_line(m: Message):
    """"Lore: Mire Hag, Gloomfang" under a reply: the lorebook entries that were sent with it."""
    if not m.lore_used:
        return None
    return Span("Lore: " + ", ".join(m.lore_used.splitlines()), cls="usage lore-used",
                title="Lorebook entries sent with this reply")


def action_btn(name: str, title: str, **kw):
    return Button(icon(name, 15), cls="icon-btn", title=title, **kw)


def message_view(m: Message, branch: Branch | None = None):
    pending = m.status in ("pending", "streaming")
    body = []

    if m.attachments:
        body.append(Div(*[attachment_view(a) for a in m.attachments], cls="attachments"))

    if m.role == "user":
        text = "🎨 Illustrate the current scene" if m.kind == "illustrate" else m.content
        body.append(Div(text, cls="content plain"))
    else:
        if m.reasoning or pending:
            body.append(Details(Summary("Reasoning"), Div(m.reasoning, cls="reasoning-text"),
                                cls="reasoning", hidden=not m.reasoning))
        if m.kind == "image":
            if m.content and not pending:
                body.append(Details(Summary("Image prompt"), P(m.content), cls="image-prompt"))
        elif pending:
            # Empty box that app.js fills with streamed text; replaced by rendered markdown when done.
            body.append(Div(cls="content plain streaming"))
        else:
            body.append(Div(markdown(m.content), cls="content md"))
        if pending:
            body.append(Div(Span(cls="dots"), Span(cls="status-line"), cls="pending-row"))
        if m.status == "stopped":
            body.append(Div("Stopped.", cls="note"))
        if m.status == "error":
            body.append(Div(m.error or "Something went wrong.", cls="error-box"))

    actions = []
    if not pending:
        actions.append(action_btn("copy", "Copy", onclick="copyMessage(this)"))
        if m.kind in ("text",) and m.status != "error":
            actions.append(action_btn("edit", "Edit" if m.role == "assistant" else "Edit (creates a new branch)",
                                      hx_get=f"/edit_form?mid={m.id}", hx_target=f"#msg-{m.id}",
                                      hx_swap="outerHTML"))
        if m.role == "assistant":
            actions.append(action_btn("regen", "Regenerate (creates a new branch)", hx_post=f"/regen?mid={m.id}",
                                      hx_target="#messages"))
        actions.append(action_btn("fork", "Fork into a new chat from here", hx_post=f"/fork?mid={m.id}"))

    return Div(
        Div(*body, cls="bubble"),
        Div(branch_nav(branch), usage_line(m), lore_used_line(m), Div(*actions, cls="actions"), cls="msg-meta"),
        id=f"msg-{m.id}",
        cls=f"msg msg-{m.role} status-{m.status}",
        data_stream_url=f"/stream?mid={m.id}" if m.status == "pending" else None,
        data_mid=m.id)


def summary_view(m: Message):
    """The "Story so far" box, shown after the last message it covers."""
    return Details(
        Summary(icon("book", 15), Span("Story so far"),
                Small("Everything above is sent to the model as this summary, not word for word.",
                      cls="muted")),
        Div(markdown(m.summary), cls="content md"),
        Div(Button(icon("edit", 15), "Edit", type="button", cls="btn btn-ghost",
                   hx_get=f"/summary_form?mid={m.id}", hx_target=f"#summary-{m.id}", hx_swap="outerHTML"),
            Button(icon("trash", 15), "Remove", type="button", cls="btn btn-ghost",
                   hx_post=f"/summary_remove?mid={m.id}", hx_target="#messages",
                   hx_confirm="Remove this summary? The messages above will be sent in full again."),
            cls="row gap end"),
        id=f"summary-{m.id}", cls="story-summary")


def summary_form(m: Message):
    return Form(
        Textarea(m.summary, name="summary", rows=16, required=True, cls="textarea"),
        Div(Button("Cancel", type="button", cls="btn btn-ghost", hx_get=f"/summary?mid={m.id}",
                   hx_target=f"#summary-{m.id}", hx_swap="outerHTML"),
            Button("Save", type="submit", cls="btn btn-primary"),
            cls="row gap end"),
        hx_post=f"/summary?mid={m.id}", hx_swap="outerHTML",
        id=f"summary-{m.id}", cls="story-summary stack tight")


def edit_form(m: Message):
    hint = "Saving creates a new branch and a new reply." if m.role == "user" else "Edits this reply in place."
    return Form(
        Textarea(m.content, name="content", rows=min(max(m.content.count("\n") + 3, 4), 20), cls="edit-area"),
        Div(Small(hint, cls="muted"),
            Div(Button("Cancel", type="button", cls="btn btn-ghost", hx_get=f"/msg?mid={m.id}",
                       hx_target=f"#msg-{m.id}", hx_swap="outerHTML"),
                Button("Save", type="submit", cls="btn btn-primary"), cls="row gap"),
            cls="row between"),
        hx_post=f"/edit?mid={m.id}",
        hx_target="#messages" if m.role == "user" else f"#msg-{m.id}",
        hx_swap="innerHTML" if m.role == "user" else "outerHTML",
        id=f"msg-{m.id}", cls=f"msg msg-{m.role} editing")


# ---------------------------------------------------------------- composer & settings


def model_select(selected: str, name: str = "model", **kw):
    selected = config.usable_model(selected)  # matches what chat.effective_model will actually use
    options = []
    for m in config.AVAILABLE_MODELS:
        tags = []
        if m.free:
            tags.append("free")
        if m.vision:
            tags.append("vision")
        label = m.label + (f"  ({', '.join(tags)})" if tags else "")
        options.append(Option(label, value=m.id, selected=m.id == selected))
    return Select(*options, name=name, cls="select", **kw)


def reasoning_control(model_id: str, thinking: bool, effort: str, oob: bool = False):
    """Effort dropdown for models that always think (GLM-5.3 family), on/off switch for the rest.

    It's swapped out whenever the model changes, so it always matches the selected model.
    """
    if config.uses_effort(model_id):
        options = [Option(level.capitalize(), value=level, selected=level == effort) for level in config.EFFORT_LEVELS]
        control = Label("Effort", Select(*options, name="reasoning_effort", cls="select effort-select"),
                        cls="effort", title="How hard the model thinks before answering. Higher is slower "
                                            "and uses more tokens.")
    else:
        control = Label(Input(type="checkbox", name="thinking", checked=thinking, role="switch"), "Thinking",
                        cls="switch", title="Let the model reason before answering (slower). "
                                            "The reasoning shows in a collapsible box.")
    return Span(control, id="reasoning-control", cls="reasoning-control", hx_swap_oob="true" if oob else None)


def lore_switch(on: bool, cid: int | None = None, oob: bool = False):
    """The Lorekeeper switch: whether this chat uses its lorebook. On the new-chat page it's a form field
    (`lore_on`); in a chat (`cid`) it saves as soon as it's flipped."""
    saves = dict(hx_post=f"/lore/switch?cid={cid}", hx_trigger="change", hx_swap="none") if cid else {}
    return Label(Input(type="checkbox", name="on" if cid else "lore_on", value="1", checked=on, role="switch",
                       **saves),
                 "Lorekeeper", id="lore-switch", cls="switch", hx_swap_oob="true" if oob else None,
                 title="The Lorekeeper looks up this chat's lorebook before each reply, sends the entries that "
                       "match the story, and keeps the lorebook up to date as the story goes. Off: the lorebook "
                       "is kept, but nothing is searched, sent or updated.")


def _usd(amount: float) -> str:
    return f"${amount:.2f}" if amount >= 0.1 else f"${amount:g}"  # $0.50, $1.40, but $0.075 stays exact


def model_price(model_id: str, oob: bool = False):
    """The selected model's per-token prices. Swapped together with reasoning_control."""
    p = pricing.price(model_id)
    if p.free:
        text = "Free"
    elif p.input is None or p.output is None:
        text = "Price unknown"
    else:
        cached = f" · cached {_usd(p.cached)}" if p.cached is not None else ""
        text = f"in {_usd(p.input)}{cached} · out {_usd(p.output)}"

    source = (f"docs.z.ai, checked {pricing.checked_at:%b} {pricing.checked_at.day}"
              if model_id in pricing.live_prices and pricing.checked_at else "built-in list")
    tip = f"Pay-as-you-go price per 1M tokens ({source})."
    if config.USING_CODING_PLAN:
        tip += " Your Coding Plan covers this model, so these aren't charged."
    return Span(text, Small(" /1M", cls="muted") if "$" in text else None,
                id="model-price", cls="model-price", title=tip, hx_swap_oob="true" if oob else None)


def composer_bar(illustrate_cid: int | None = None):
    """The message box + toolbar. Lives inside a Form built by the caller."""
    return Div(
        Div(id="attach-preview", cls="attach-preview"),
        Textarea(name="content", id="composer-input", rows=1, placeholder="Message…  (Shift+Enter for a new line)"),
        Input(type="hidden", name="mode", value="chat", id="mode-input"),
        Div(
            Label(icon("clip"), Input(type="file", name="files", multiple=True, id="file-input", hidden=True,
                                      accept=".pdf,.docx,.txt,.md,.csv,.json,.html,.py,.png,.jpg,.jpeg,.webp,.gif"),
                  cls="icon-btn", title="Attach documents or images"),
            Button(icon("image"), Span("Image", cls="btn-label"), type="button", id="mode-btn", cls="pill-btn",
                   onclick="toggleImageMode()", title="Image mode: send your text to glm-image"),
            Button(icon("wand"), Span("Illustrate", cls="btn-label"), type="button", cls="pill-btn",
                   hx_post=f"/illustrate?cid={illustrate_cid}", hx_target="#messages", hx_swap="beforeend",
                   title="Generate an image of the current scene") if illustrate_cid else None,
            Div(cls="grow"),
            Button(icon("stop"), type="button", id="stop-btn", cls="send-btn stop", onclick="stopStreaming()",
                   title="Stop"),
            Button(icon("send"), type="submit", id="send-btn", cls="send-btn", title="Send"),
            cls="composer-bar"),
        cls="composer-box")


def scenario_options(scenarios: list[Scenario], selected: int | None):
    return [Option("— None —", value="")] + [
        Option(s.name, value=s.id, selected=s.id == selected) for s in scenarios]


def system_prompt_area(text: str = "", rows: int = 6):
    return Textarea(text, name="system_prompt", id="system-prompt", rows=rows, cls="textarea",
                    placeholder="You are a helpful assistant…  (or pick a scenario above)")


def stats_view(conv: Conversation, usage: Usage, ctx_used: int, ctx_model: str | None, oob: bool = False):
    window = config.model_info(conv.model).context
    pct = min(100, round(100 * ctx_used / window)) if window else 0
    cache_pct = round(100 * usage.cached / usage.prompt) if usage.prompt else 0
    return Div(
        Div(Div(cls="ctx-fill", style=f"width:{pct}%"), cls="ctx-bar",
            title=f"Context: {ctx_used:,} of {window:,} tokens ({pct}%) on this branch"),
        Span(f"{fmt(ctx_used)} / {fmt(window)}", cls="ctx-label"),
        lore_switch(conv.lore_on, cid=conv.id),
        Span(f"in {fmt(usage.prompt)} · out {fmt(usage.completion)} · {cache_pct}% cached · {fmt_cost(usage)}",
             cls="spend",
             title=f"This chat, all branches: prompt {usage.prompt:,} (cached {usage.cached:,}), "
                   f"completion {usage.completion:,}, {usage.images} images"),
        Button(Span("Summarize", cls="idle-label"), Span("Summarizing…", cls="busy-label"),
               type="button", cls="pill-btn summarize-btn", hx_post=f"/summarize?cid={conv.id}",
               hx_target="#messages", hx_disabled_elt="this",
               title="Summarize older messages into a \"Story so far\" now. It also happens by itself "
                     "when the context gets big."),
        id="chat-stats", cls="chat-stats", hx_swap_oob="true" if oob else None)


def chat_header(conv: Conversation, stats):
    return Header(
        Div(Input(value=conv.title, name="title", cls="title-input", hx_post=f"/rename?cid={conv.id}",
                  hx_trigger="change", hx_swap="none", aria_label="Chat title"),
            Span(conv.scenario.name, cls="conv-tag") if conv.scenario else None,
            cls="row gap grow min0"),
        Form(model_select(conv.model),
             reasoning_control(config.usable_model(conv.model), conv.thinking, conv.reasoning_effort),
             model_price(config.usable_model(conv.model)),
             hx_post=f"/settings?cid={conv.id}", hx_trigger="change", hx_swap="none", cls="row gap"),
        stats,
        cls="chat-header")


def system_drawer(conv: Conversation):
    return Details(
        Summary("System prompt", Small(" · editing it mid-chat resets the prompt cache", cls="muted")),
        Form(system_prompt_area(conv.system_prompt, rows=8),
             Div(Span(id="system-saved", cls="muted small"),
                 Div(Input(name="name", placeholder="Scenario name", cls="input small-input"),
                     Button(icon("save", 15), "Save as scenario", type="submit", cls="btn btn-ghost",
                            hx_post=f"/save_scenario?cid={conv.id}", hx_target="#system-saved"),
                     Button("Apply", type="submit", cls="btn btn-primary"), cls="row gap"),
                 cls="row between"),
             hx_post=f"/system?cid={conv.id}", hx_target="#system-saved"),
        cls="system-drawer")


def chat_page(conv: Conversation, header, messages: list, lore_drawer=None):
    """`lore_drawer` comes from lore_ui.chat_lore_drawer (built by the route, so ui.py needn't import it)."""
    return Div(
        header,
        system_drawer(conv),
        lore_drawer,
        Div(Div(*messages, id="messages", cls="messages"), id="scroller", cls="scroller"),
        Form(composer_bar(illustrate_cid=conv.id), hx_post=f"/send?cid={conv.id}", hx_target="#messages",
             hx_swap="beforeend", id="composer", cls="composer"),
        cls="chat")


def new_chat_page(scenarios: list[Scenario], scenario: Scenario | None):
    return Div(
        Div(H1("What are we doing today?"),
            P("Pick a scenario or write a system prompt, choose a model, and start typing.", cls="muted"),
            cls="hero"),
        Form(
            Div(Label("Scenario", Select(*scenario_options(scenarios, scenario.id if scenario else None),
                                         name="scenario_id", cls="select", hx_get="/scenario_prompt",
                                         hx_target="#system-prompt", hx_swap="outerHTML")),
                # Changing the model swaps in the matching effort/thinking control (and its price).
                Label("Model", model_select(config.DEFAULT_MODEL, hx_get="/reasoning_control",
                                            hx_target="#reasoning-control", hx_swap="outerHTML")),
                reasoning_control(config.DEFAULT_MODEL, False, config.DEFAULT_EFFORT),
                model_price(config.DEFAULT_MODEL),
                lore_switch(on=scenario is not None),  # on for a scenario (a story), off for a plain chat
                cls="setup-row"),
            Details(Summary("System prompt"), system_prompt_area(scenario.system_prompt if scenario else ""),
                    open=True, cls="setup-prompt"),
            composer_bar(),
            hx_post="/new", hx_swap="none", id="composer", cls="composer new-composer"),
        cls="new-chat")


# ---------------------------------------------------------------- scenarios


def category_tags(s: Scenario):
    """A scenario's categories as small tags; clicking one searches for it.

    Every card gets this row, so the cards line up; one without categories gets a link to add some.
    """
    names = category.split(s.categories)
    if not names:
        return Div(A("+ Add categories", href=f"/scenarios/edit?sid={s.id}", cls="tag tag-empty",
                     title="No categories yet. Open the editor to add some."), cls="tags")
    return Div(*[A(name, href=f"/scenarios?q={quote(name)}", cls="tag", title=f"Show {name} scenarios")
                 for name in names], cls="tags")


def scenarios_page(scenarios: list[Scenario], q: str = ""):
    cards = [Div(
        Div(Div(H2(s.name), Input(type="checkbox", name="sid", value=s.id, cls="pick",
                                  title="Select for export", aria_label=f"Select {s.name} for export"),
                cls="row between"),
            category_tags(s),
            P(s.description, cls="muted") if s.description else None,
            P(s.system_prompt[:220] + ("…" if len(s.system_prompt) > 220 else ""), cls="preview"),
            cls="card-body"),
        Div(Div(delete_scenario_btn(s), clone_scenario_btn(s), export_scenario_btn(s), cls="row"),
            Div(A("Edit", href=f"/scenarios/edit?sid={s.id}", cls="btn btn-ghost"),
                A(icon("chat", 15), "Start chat", href=f"/?scenario={s.id}", cls="btn btn-primary"),
                cls="row gap"),
            cls="row between"),
        cls="card") for s in scenarios]
    if not cards and q:
        cards = [Div(P(f"No scenarios match “{q}”."), cls="card empty")]
    elif not cards:
        cards = [Div(P("No scenarios yet. A scenario is a reusable system prompt: a setting, a character, "
                       "a set of rules for a campaign."), cls="card empty")]
    # Import: picking files submits the form straight away (several files at once is fine).
    import_btn = Form(
        Label(icon("upload", 15), "Import",
              Input(type="file", name="files", accept=".json,application/json", multiple=True, hidden=True),
              cls="btn btn-ghost", title="Import scenarios from .json files"),
        hx_post="/scenarios/import", hx_encoding="multipart/form-data", hx_trigger="change",
        hx_target="#scenarios-page", hx_swap="outerHTML")
    # Export: submits the card checkboxes below; nothing checked means everything.
    export_btn = Button(icon("download", 15), "Export", type="submit", form="export-form", cls="btn btn-ghost",
                        title="Download the checked scenarios as JSON (all of them if none are checked)")
    return Div(
        Div(H1("Scenarios"),
            Div(import_btn, export_btn if scenarios else None,
                A(icon("plus"), "New scenario", href="/scenarios/edit", cls="btn btn-primary"),
                cls="row gap"),
            cls="row between page-head"),
        # Typing asks for the whole page again (?q=...), but only the cards are swapped in (hx-select),
        # so the box keeps focus. The URL updates too, so reload and Back keep the search.
        Input(type="search", name="q", value=q, cls="input search-input", aria_label="Search scenarios",
              placeholder="Search names, categories and text…", autocomplete="off",
              hx_get="/scenarios", hx_trigger="input changed delay:250ms, search",
              hx_target="#scenario-cards", hx_select="#scenario-cards", hx_swap="outerHTML",
              hx_push_url="true"),
        Form(Div(*cards, id="scenario-cards", cls="card-grid"), id="export-form", action="/scenarios/export",
             method="get"),
        id="scenarios-page", cls="page")


def export_scenario_btn(s: Scenario):
    return A(icon("download", 15), href=f"/scenarios/export?sid={s.id}", cls="icon-btn",
             title="Export as JSON")


def delete_scenario_btn(s: Scenario):
    return Button(icon("trash", 15), type="button", cls="icon-btn danger-hover", title="Delete scenario",
                  hx_post=f"/scenarios/delete?sid={s.id}",
                  hx_confirm=f'Delete the scenario "{s.name}"? Chats that used it are kept.')


def clone_scenario_btn(s: Scenario):
    return Button(icon("copy", 15), type="button", cls="icon-btn", title="Clone scenario",
                  hx_post=f"/scenarios/clone?sid={s.id}")


def category_picker(current: str, known: list[str]):
    """Text box for categories, plus one-click buttons for common and already-used ones (see app.js)."""
    chosen = [c.lower() for c in category.split(current)]
    buttons = [Button(name, type="button", cls="tag-btn" + (" on" if name.lower() in chosen else ""),
                      onclick="toggleCategory(this)") for name in known]
    return Div(
        Label("Categories",
              Input(name="categories", id="categories-input", value=current, cls="input", autocomplete="off",
                    placeholder="Comma separated, e.g. Fantasy, Mystery. Leave empty and the AI picks on save.")),
        Div(*buttons, cls="tags"),
        cls="stack tight")


def scenario_prompt_area(text: str, oob: bool = False):
    """The scenario editor's system prompt box (refreshed when "Move details into lorebook" is applied)."""
    return Textarea(text, name="system_prompt", id="scenario-prompt", rows=18, cls="textarea mono",
                    hx_swap_oob="true" if oob else None)


def scenario_editor(s: Scenario | None, known_categories: list[str], lore_section=None):
    """`lore_section` comes from lore_ui.scenario_lore_section (built by the route, so ui.py needn't import it)."""
    return Div(
        Div(H1("Edit scenario" if s else "New scenario"), A("Back", href="/scenarios", cls="btn btn-ghost"),
            cls="row between page-head"),
        Form(
            Label("Name", Input(name="name", value=s.name if s else "", required=True, cls="input")),
            Label("Short description", Input(name="description", value=s.description if s else "", cls="input",
                                             placeholder="Optional one-liner shown on the card")),
            category_picker(s.categories if s else "", known_categories),
            Label("System prompt", scenario_prompt_area(s.system_prompt if s else "")),
            Input(type="hidden", name="sid", value=s.id if s else ""),
            Div(Div(Button(icon("trash", 15), "Delete", type="button", cls="btn btn-danger",
                           hx_post=f"/scenarios/delete?sid={s.id}",
                           hx_confirm=f'Delete the scenario "{s.name}"? Chats that used it are kept.'),
                    Button(icon("copy", 15), "Clone", type="button", cls="btn btn-ghost",
                           hx_post=f"/scenarios/clone?sid={s.id}"),
                    A(icon("download", 15), "Export", href=f"/scenarios/export?sid={s.id}", cls="btn btn-ghost"),
                    cls="row gap") if s else Span(),
                Div(A("Start chat", href=f"/?scenario={s.id}", cls="btn btn-ghost") if s else None,
                    Button("Save", type="submit", cls="btn btn-primary"), cls="row gap"),
                cls="row between"),
            # Saving can take a few seconds when the AI picks categories; stop double-clicks meanwhile.
            hx_post="/scenarios/save", hx_disabled_elt="find button[type='submit']", cls="stack"),
        lore_section,
        cls="page narrow")
