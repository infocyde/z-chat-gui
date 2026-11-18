"""HTML for lorebooks: the entry list and form (used for both a scenario's lorebook and a chat's),
and the scenario editor's lorebook section. No database access here; routes pass data in."""

from __future__ import annotations

from fasthtml.common import (A, Button, Details, Div, Form, H2, Input, Label, P, Small, Span, Strong, Summary,
                             Textarea)

from .lore import split_keys
from .ui import icon

PREVIEW_CHARS = 160


def lore_row(e, *, edit_url: str, delete_url: str, target: str, delete_note: str = ""):
    """One entry: name, keys, the start of its text, and Edit / Delete. `e` has id, name, keys, text, enabled.
    `delete_note` is added to the delete confirm."""
    preview = e.text[:PREVIEW_CHARS] + ("…" if len(e.text) > PREVIEW_CHARS else "")
    return Div(
        Div(Div(Strong(e.name), Span("Disabled", cls="tag tag-off") if not e.enabled else None,
                cls="row gap min0"),
            Div(Button(icon("edit", 15), cls="icon-btn", title="Edit", type="button",
                       hx_get=f"{edit_url}{e.id}", hx_target=f"#lore-row-{e.id}", hx_swap="outerHTML"),
                Button(icon("trash", 15), cls="icon-btn danger-hover", title="Delete", type="button",
                       hx_post=f"{delete_url}{e.id}", hx_target=target, hx_swap="outerHTML",
                       hx_confirm=f'Delete "{e.name}" from the lorebook?{delete_note}'),
                cls="row"),
            cls="row between"),
        Div(*[Span(k, cls="tag tag-static") for k in split_keys(e.keys)], cls="tags") if e.keys else None,
        P(preview, cls="preview"),
        id=f"lore-row-{e.id}", cls="lore-row")


def lore_list(entries: list, *, list_id: str, edit_url: str, delete_url: str, empty_text: str,
              header=None, oob: bool = False, delete_note: str = ""):
    """The whole list. Adding, editing, deleting and importing all swap this element out for a fresh one.

    The first child is an empty slot where the "Add entry" form appears.
    """
    target = f"#{list_id}"
    rows = [lore_row(e, edit_url=edit_url, delete_url=delete_url, target=target, delete_note=delete_note)
            for e in entries]
    return Div(header, Div(id=f"{list_id}-new"),
               *(rows or [P(empty_text, cls="muted small")]),
               id=list_id, cls="lore-list", hx_swap_oob="true" if oob else None)


def lore_form(*, post_url: str, cancel_url: str, target: str, hidden: dict, entry_id: int | None = None,
              name: str = "", keys: str = "", text: str = "", enabled: bool = True):
    """Add or edit one entry. Saving or cancelling re-renders the whole list (`target`)."""
    return Form(
        Label("Name", Input(name="name", value=name, required=True, cls="input", placeholder="Mire Hag")),
        Label("Keys", Input(name="keys", value=keys, cls="input", placeholder="hag, witch, swamp")),
        Label("Text", Textarea(text, name="text", rows=6, required=True, cls="textarea",
                               placeholder="What the narrator should know about it.")),
        Div(Label(Input(type="checkbox", name="enabled", value="1", checked=enabled), " Enabled", cls="check"),
            Div(Button("Cancel", type="button", cls="btn btn-ghost", hx_get=cancel_url, hx_target=target,
                       hx_swap="outerHTML"),
                Button("Save", type="submit", cls="btn btn-primary"), cls="row gap"),
            cls="row between"),
        *[Input(type="hidden", name=k, value=v) for k, v in hidden.items() if v is not None],
        hx_post=post_url, hx_target=target, hx_swap="outerHTML",
        id=f"lore-row-{entry_id}" if entry_id else None, cls="lore-form stack tight")


# ---------------------------------------------------------------- scenario editor


def scenario_lore_list(scenario):
    return lore_list(scenario.lore, list_id="scenario-lore",
                     edit_url=f"/lore/scenario/form?sid={scenario.id}&lid=",
                     delete_url="/lore/scenario/delete?lid=",
                     empty_text="No entries yet. Add characters, places, monsters or items: only the ones the "
                                "story is touching are sent with each reply.")


def scenario_lore_form(scenario, entry=None):
    return lore_form(post_url="/lore/scenario/save", cancel_url=f"/lore/scenario/list?sid={scenario.id}",
                     target="#scenario-lore", hidden={"sid": scenario.id, "lid": entry.id if entry else None},
                     entry_id=entry.id if entry else None,
                     name=entry.name if entry else "", keys=entry.keys if entry else "",
                     text=entry.text if entry else "", enabled=entry.enabled if entry else True)


def scenario_lore_section(scenario):
    """The "Lorebook" part of the scenario editor, below the scenario form."""
    if scenario is None:
        return Div(H2("Lorebook"), P("Save the scenario first to add a lorebook.", cls="muted"),
                   cls="lore-section")
    import_btn = Form(
        Label(icon("upload", 15), "Import",
              Input(type="file", name="files", accept=".json,application/json", multiple=True, hidden=True),
              cls="btn btn-ghost", title="Import SillyTavern World Info, Character Card or zchat lorebook files"),
        hx_post=f"/lore/scenario/import?sid={scenario.id}", hx_encoding="multipart/form-data",
        hx_trigger="change", hx_target="#scenario-lore", hx_swap="outerHTML")
    return Div(
        Div(H2("Lorebook"),
            Div(import_btn,
                A(icon("download", 15), "Export", href=f"/lore/scenario/export?sid={scenario.id}",
                  cls="btn btn-ghost"),
                # Drafting takes a few seconds; stop double-clicks meanwhile.
                Button(icon("wand", 15), "Draft from prompt", type="button", cls="btn btn-ghost",
                       title="Let the AI pull characters, places and items out of the system prompt",
                       hx_post=f"/lore/scenario/draft?sid={scenario.id}", hx_target="#scenario-lore",
                       hx_swap="outerHTML", hx_disabled_elt="this"),
                Button(icon("plus", 15), "Add entry", type="button", cls="btn btn-primary",
                       hx_get=f"/lore/scenario/form?sid={scenario.id}", hx_target="#scenario-lore-new"),
                cls="row gap wrap"),
            cls="row between"),
        P("Copied into each new chat started from this scenario. Before every reply, only the entries "
          "that match what's happening are sent.", cls="muted small"),
        scenario_lore_list(scenario),
        cls="lore-section stack tight")


# ---------------------------------------------------------------- chat drawer


def entry_count(n: int) -> str:
    return f"{n} entry" if n == 1 else f"{n} entries"


def chat_lore_list(conv_id: int, versions: list, oob: bool = False):
    """The chat's lorebook as this branch sees it. `versions` come from lore.current_versions."""
    header = Div(Span(f"{entry_count(len(versions))} · only the entries that match what's happening are sent "
                      "with each reply", cls="muted small"),
                 Button(icon("plus", 15), "Add entry", type="button", cls="btn btn-ghost",
                        hx_get=f"/lore/chat/form?cid={conv_id}", hx_target="#chat-lore-new"),
                 cls="row between")
    return lore_list(versions, list_id="chat-lore", edit_url=f"/lore/chat/form?cid={conv_id}&vid=",
                     delete_url="/lore/chat/delete?vid=", header=header, oob=oob,
                     delete_note=" It goes on every branch.",
                     empty_text="No entries yet. Add characters, places, monsters or items the narrator "
                                "should remember.")


def chat_lore_form(conv_id: int, version=None):
    return lore_form(post_url="/lore/chat/save", cancel_url=f"/lore/chat/list?cid={conv_id}", target="#chat-lore",
                     hidden={"cid": conv_id, "vid": version.id if version else None},
                     entry_id=version.id if version else None,
                     name=version.name if version else "", keys=version.keys if version else "",
                     text=version.text if version else "", enabled=version.enabled if version else True)


def chat_lore_drawer(conv_id: int, versions: list):
    """The "Lorebook" drawer under the system prompt drawer on a chat page."""
    return Details(Summary("Lorebook", Small(" · this chat's own copy; edits here don't change the scenario",
                                             cls="muted")),
                   chat_lore_list(conv_id, versions),
                   cls="system-drawer lore-drawer")


# ---------------------------------------------------------------- lorebook update notes


def lore_note_view(m, versions: list):
    """The note after a reply when the lorebook updated itself. `versions` are the ones it saved."""
    if m.lore_note == "Undone.":
        return Div(icon("book", 15), Span("Lorebook update undone."), id=f"lore-note-{m.id}",
                   cls="lore-note muted small")
    if m.lore_note == "No changes.":
        return Div(icon("book", 15), Span("Lorebook checked: no changes."), id=f"lore-note-{m.id}",
                   cls="lore-note muted small")
    return Details(
        Summary(icon("book", 15), Span("Lorebook updated"),
                Small(" · " + " · ".join(m.lore_note.splitlines()), cls="muted")),
        Div(*[Div(Strong(v.name), P(v.text, cls="preview")) for v in versions], cls="stack tight"),
        Div(Button("Undo", type="button", cls="btn btn-ghost", hx_post=f"/lore/undo?mid={m.id}",
                   hx_target="#messages",
                   hx_confirm="Undo this lorebook update? The entries it added or changed go back to how "
                              "they were, including any edits you made to them since."),
            cls="row end"),
        id=f"lore-note-{m.id}", cls="lore-note")
