"""Runs a pending assistant message to completion, yielding progress events.

Kept separate from the web layer so it can be tested without HTTP. Each event is
(name, payload): "delta"/"reasoning" carry text, "status" a progress line, "title" means the
conversation got a title, and the stream always ends with "done" or "error".
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime
from pathlib import PurePosixPath
from typing import AsyncIterator
from urllib.parse import urlparse

from . import config, documents, zai
from .chat import build_messages, effective_model, path_to, touch
from .db import session_scope
from .models import Attachment, Conversation, Message

Event = tuple[str, dict]

# Message ids the user asked to stop; checked between streamed chunks.
STOP: set[int] = set()

ILLUSTRATE_SYSTEM = (
    "You write prompts for an image generation model. Given a story/chat transcript, describe the "
    "single most vivid moment of the LATEST scene as one image: subjects, setting, lighting, mood, "
    "composition, art style. Concrete visual details only, no dialogue, no meta commentary. "
    "Reply with the prompt only, under 120 words."
)
# The message goes inside the instruction (not as its own user turn); otherwise roleplay-tuned
# models tend to just continue the story instead of titling it.
TITLE_PROMPT = ("Write a short title (at most 6 words) for a chat that begins with the message below. "
                "Reply with the title only: no quotes, no punctuation at the end.\n\n<message>\n{seed}\n</message>")


async def run(mid: int) -> AsyncIterator[Event]:
    with session_scope() as s:
        m = s.get(Message, mid)
        if m is None:
            yield "error", {"msg": "Message not found."}
            return
        if m.status != "pending":
            yield "done", {}
            return
        m.status = "streaming"
        kind = m.kind
        conv = m.conversation
        path = path_to(s, m.parent_id)
        model = effective_model(conv, path)
        if kind != "image":
            m.model = model
        msgs = build_messages(conv, path, model)
        thinking, effort, temperature = conv.thinking, conv.reasoning_effort, conv.temperature

    if kind == "image":
        async for ev in _run_image(mid):
            yield ev
    else:
        async for ev in _run_text(mid, msgs, model, thinking, effort, temperature):
            yield ev


async def _run_text(mid: int, msgs: list[dict], model: str, thinking: bool, effort: str,
                    temperature: float | None) -> AsyncIterator[Event]:
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}
    status, error = "done", None
    try:
        async for ev in zai.stream_chat(msgs, model, thinking, temperature, effort=effort):
            if mid in STOP:
                status = "stopped"
                break
            if ev.type == "content":
                content.append(ev.text)
                yield "delta", {"t": ev.text}
            elif ev.type == "reasoning":
                reasoning.append(ev.text)
                yield "reasoning", {"t": ev.text}
            elif ev.type == "usage":
                usage = ev.data
            elif ev.type == "finish" and ev.text == "sensitive":
                status, error = "error", "Z.ai's safety filter stopped this reply."
            elif ev.type == "finish" and ev.text == "model_context_window_exceeded":
                status, error = "error", "The conversation no longer fits in this model's context window."
    except zai.ZaiError as e:
        status, error = "error", str(e)
    except (asyncio.CancelledError, GeneratorExit):
        status = "stopped"
        raise
    except Exception as e:  # network errors etc. — surface them instead of hanging the UI
        status, error = "error", f"{type(e).__name__}: {e}"
    finally:
        STOP.discard(mid)
        _save_text(mid, "".join(content), "".join(reasoning), usage, status, error)

    if status == "error":
        yield "error", {"msg": error}
        return
    if await _maybe_title(mid):
        yield "title", {}
    yield "done", {}


def _save_text(mid: int, content: str, reasoning: str, usage: dict, status: str, error: str | None) -> None:
    with session_scope() as s:
        m = s.get(Message, mid)
        if m is None:
            return
        m.content, m.reasoning, m.status, m.error = content, reasoning, status, error
        if status == "stopped" and not content and not reasoning:
            m.status, m.error = "error", "Stopped before any reply arrived."
        m.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        m.completion_tokens = int(usage.get("completion_tokens") or 0)
        m.cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        m.reasoning_tokens = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        touch(m.conversation)


def _transcript(path: list[Message], conv: Conversation, limit: int = 8) -> str:
    lines = []
    if conv.system_prompt.strip():
        lines.append(f"[Setting]\n{conv.system_prompt.strip()[:2000]}\n")
    for m in [m for m in path if m.kind == "text" and m.content][-limit:]:
        who = "User" if m.role == "user" else "Narrator"
        lines.append(f"{who}: {m.content[:3000]}")
    return "\n\n".join(lines)


async def _run_image(mid: int) -> AsyncIterator[Event]:
    with session_scope() as s:
        m = s.get(Message, mid)
        parent = s.get(Message, m.parent_id) if m.parent_id else None
        illustrate = parent is not None and parent.kind == "illustrate"
        prompt = (parent.content if parent else "").strip()
        transcript = _transcript(path_to(s, m.parent_id), m.conversation) if illustrate else ""
    size = "1568x1056" if illustrate else "1280x1280"
    try:
        if illustrate:
            yield "status", {"t": "Writing an image prompt from the scene…"}
            prompt = await zai.complete([{"role": "system", "content": ILLUSTRATE_SYSTEM},
                                         {"role": "user", "content": transcript or "An empty stage."}])
        if not prompt:
            raise zai.ZaiError("Empty image prompt.")
        yield "status", {"t": "Generating image… (10–30s)"}
        url = await zai.generate_image(prompt, size=size)
        suffix = PurePosixPath(urlparse(url).path).suffix.lower()
        suffix = suffix if suffix in documents.IMAGE_EXTS else ".png"
        dest = config.IMAGE_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{mid}_{secrets.token_hex(3)}{suffix}"
        size_bytes = await zai.download(url, dest)
    except (asyncio.CancelledError, GeneratorExit):
        _fail(mid, "Stopped.", prompt)
        raise
    except Exception as e:
        msg = str(e) if isinstance(e, zai.ZaiError) else f"{type(e).__name__}: {e}"
        _fail(mid, msg, prompt)
        yield "error", {"msg": msg}
        return
    with session_scope() as s:
        m = s.get(Message, mid)
        m.content, m.status, m.error = prompt, "done", None
        s.add(Attachment(message_id=mid, kind="generated", filename=dest.name, path=documents.rel(dest),
                         mime=f"image/{suffix.lstrip('.').replace('jpg', 'jpeg')}", size=size_bytes))
        touch(m.conversation)
    STOP.discard(mid)
    if await _maybe_title(mid):
        yield "title", {}
    yield "done", {}


def _fail(mid: int, msg: str, prompt: str) -> None:
    STOP.discard(mid)
    with session_scope() as s:
        m = s.get(Message, mid)
        if m is not None:
            m.status, m.error, m.content = "error", msg, prompt or m.content


async def _maybe_title(mid: int) -> bool:
    """Give a brand-new conversation a title after its first reply."""
    with session_scope() as s:
        m = s.get(Message, mid)
        conv = m.conversation
        if conv.title != "New chat":
            return False
        cid = conv.id
        # Use the first user message as the basis for the title (or its file name if it was just a file).
        seed = ""
        user_messages = [x for x in path_to(s, mid) if x.role == "user"]
        if user_messages:
            first = user_messages[0]
            seed = first.content
            if not seed and first.attachments:
                seed = first.attachments[0].filename
    title = ""
    if seed.strip():
        try:
            title = await asyncio.wait_for(zai.complete(
                [{"role": "user", "content": TITLE_PROMPT.format(seed=seed[:2000])}], max_tokens=30), timeout=30)
        except Exception:
            title = ""
    title = title.strip().strip('"').strip("*#").strip()
    if not title or len(title.split()) > 8:  # empty, or the model ignored the instructions
        title = seed.strip()[:60] or "Untitled"
    with session_scope() as s:
        s.get(Conversation, cid).title = title.splitlines()[0][:120]
    return True
