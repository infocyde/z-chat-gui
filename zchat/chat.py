"""Conversation-tree operations and API context building (pure DB logic, no HTTP)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import config, documents, pricing
from .models import Attachment, Conversation, Message, utcnow

# ---------------------------------------------------------------- tree navigation


def path_to(s: Session, leaf_id: int | None) -> list[Message]:
    """Messages from the root down to `leaf_id` (inclusive)."""
    out: list[Message] = []
    mid = leaf_id
    seen: set[int] = set()
    while mid is not None and mid not in seen:
        seen.add(mid)
        m = s.get(Message, mid)
        if m is None:
            break
        out.append(m)
        mid = m.parent_id
    out.reverse()
    return out


def active_path(s: Session, conv: Conversation) -> list[Message]:
    return path_to(s, conv.active_leaf_id)


def children(s: Session, conv_id: int, parent_id: int | None) -> list[Message]:
    q = select(Message).where(Message.conversation_id == conv_id)
    q = q.where(Message.parent_id.is_(None) if parent_id is None else Message.parent_id == parent_id)
    return list(s.scalars(q.order_by(Message.id)))


def siblings(s: Session, m: Message) -> list[Message]:
    return children(s, m.conversation_id, m.parent_id)


def deepest_leaf(s: Session, m: Message) -> Message:
    """Follow the most recent child down to a leaf (used when switching branches)."""
    cur = m
    while True:
        kids = children(s, cur.conversation_id, cur.id)
        if not kids:
            return cur
        cur = kids[-1]


def touch(conv: Conversation) -> None:
    conv.updated_at = utcnow()


def add_message(s: Session, conv: Conversation, role: str, content: str = "", *, parent_id: int | None,
                kind: str = "text", status: str = "done", model: str | None = None) -> Message:
    m = Message(conversation_id=conv.id, parent_id=parent_id, role=role, kind=kind, content=content,
                status=status, model=model)
    s.add(m)
    s.flush()
    conv.active_leaf_id = m.id
    touch(conv)
    return m


def add_attachment(s: Session, msg: Message, p: documents.Processed) -> Attachment:
    a = Attachment(message_id=msg.id, kind=p.kind, filename=p.filename, path=p.path, mime=p.mime,
                   size=p.size, text=p.text, note=p.note)
    s.add(a)
    s.flush()
    return a


def copy_attachments(s: Session, src: Message, dst: Message) -> None:
    for a in src.attachments:
        s.add(Attachment(message_id=dst.id, kind=a.kind, filename=a.filename, path=a.path, mime=a.mime,
                         size=a.size, text=a.text, note=a.note))
    s.flush()


def send_user_message(s: Session, conv: Conversation, content: str, *, kind: str = "text",
                      image_mode: bool = False) -> tuple[Message, Message]:
    """Append a user turn plus a pending assistant reply to the end of the active branch."""
    user = add_message(s, conv, "user", content, parent_id=conv.active_leaf_id, kind=kind)
    reply = add_message(s, conv, "assistant", parent_id=user.id, status="pending",
                        kind="image" if image_mode else "text",
                        model=config.IMAGE_MODEL if image_mode else conv.model)
    return user, reply


def edit_user_message(s: Session, original: Message, content: str) -> tuple[Message, Message]:
    """Editing a user message creates a sibling branch and a fresh reply; the original is kept."""
    conv = original.conversation
    user = add_message(s, conv, "user", content, parent_id=original.parent_id, kind=original.kind)
    copy_attachments(s, original, user)
    old_reply = next(iter(children(s, conv.id, original.id)), None)
    image_mode = old_reply is not None and old_reply.kind == "image"
    reply = add_message(s, conv, "assistant", parent_id=user.id, status="pending",
                        kind="image" if image_mode else "text",
                        model=config.IMAGE_MODEL if image_mode else conv.model)
    return user, reply


def regenerate(s: Session, assistant: Message) -> Message:
    conv = assistant.conversation
    image = assistant.kind == "image"
    return add_message(s, conv, "assistant", parent_id=assistant.parent_id, status="pending",
                       kind=assistant.kind, model=config.IMAGE_MODEL if image else conv.model)


def switch_to(s: Session, m: Message) -> None:
    m.conversation.active_leaf_id = deepest_leaf(s, m).id


def fork(s: Session, conv: Conversation, upto: Message) -> Conversation:
    """New conversation containing the branch from the root through `upto`."""
    new = Conversation(title=f"{conv.title} (fork)"[:200], scenario_id=conv.scenario_id,
                       system_prompt=conv.system_prompt, model=conv.model, thinking=conv.thinking,
                       reasoning_effort=conv.reasoning_effort, temperature=conv.temperature,
                       forked_from_message_id=upto.id)
    s.add(new)
    s.flush()
    parent: int | None = None
    for m in path_to(s, upto.id):
        if m.status in ("pending", "streaming"):
            break
        c = Message(conversation_id=new.id, parent_id=parent, role=m.role, kind=m.kind, content=m.content,
                    reasoning=m.reasoning, model=m.model, status=m.status, error=m.error,
                    created_at=m.created_at)
        s.add(c)
        s.flush()
        copy_attachments(s, m, c)
        parent = c.id
    new.active_leaf_id = parent
    return new


# ---------------------------------------------------------------- API context


def _doc_block(a: Attachment) -> str:
    note = f' note="{a.note}"' if a.note else ""
    return f'<document name="{a.filename}"{note}>\n{a.text}\n</document>'


def user_content(m: Message, *, vision: bool) -> str | list[dict]:
    docs = [_doc_block(a) for a in m.attachments if a.kind == "document"]
    images = [a for a in m.attachments if a.kind == "image"]
    text = m.content
    if m.kind == "illustrate":
        text = "(The user asked for an illustration of the current scene.)"
    if docs:
        text = "\n\n".join(docs + [text]) if text else "\n\n".join(docs)
    if not images:
        return text
    if not vision:
        names = ", ".join(a.filename for a in images)
        return f"{text}\n\n[Attached image(s) not visible to this model: {names}]".strip()
    parts: list[dict] = []
    for a in images:
        p = documents.abs_path(a.path)
        if p.exists():
            parts.append({"type": "image_url", "image_url": {"url": documents.data_url(p, a.mime)}})
    parts.append({"type": "text", "text": text or "Describe this image."})
    return parts


def assistant_content(m: Message) -> str | None:
    if m.status == "error" or (m.status == "pending"):
        return None
    if m.kind == "image":
        if not any(a.kind == "generated" for a in m.attachments):
            return None
        return f"[Generated an image: {m.content}]"
    return m.content or None


def build_messages(conv: Conversation, path: list[Message], model_id: str) -> list[dict]:
    """Chat-completions `messages` for the given branch (excluding the pending reply itself).

    The prefix is deterministic for a given branch, so Z.ai's automatic context cache can hit.
    Images are only sent as pixels to vision models, and only for the newest user turn, to keep
    old base64 blobs from bloating every request.
    """
    vision = config.model_info(model_id).vision
    out: list[dict] = []
    if conv.system_prompt.strip():
        out.append({"role": "system", "content": conv.system_prompt.strip()})
    last_user_idx = max((i for i, m in enumerate(path) if m.role == "user"), default=-1)
    for i, m in enumerate(path):
        if m.role == "user":
            out.append({"role": "user", "content": user_content(m, vision=vision and i == last_user_idx)})
        else:
            c = assistant_content(m)
            if c:
                out.append({"role": "assistant", "content": c})
    return out


def needs_vision(path: list[Message]) -> bool:
    last_user = next((m for m in reversed(path) if m.role == "user"), None)
    return bool(last_user and any(a.kind == "image" for a in last_user.attachments))


def effective_model(conv: Conversation, path: list[Message]) -> str:
    """The model that will actually answer: the chat's model, unless it can't do the job."""
    # e.g. an old chat that used a free model, opened after switching to a Coding Plan key
    model = config.usable_model(conv.model)
    if needs_vision(path) and not config.model_info(model).vision:
        model = config.FALLBACK_VISION_MODEL
    return model


# ---------------------------------------------------------------- token accounting


@dataclass
class Usage:
    prompt: int = 0
    cached: int = 0
    completion: int = 0
    images: int = 0
    cost: float = 0.0
    cost_known: bool = True

    @property
    def total(self) -> int:
        return self.prompt + self.completion


def usage_for(s: Session, conv_id: int | None = None) -> Usage:
    """Total spend across all branches (what you were billed), per conversation or overall."""
    q = select(Message.model, func.sum(Message.prompt_tokens), func.sum(Message.cached_tokens),
               func.sum(Message.completion_tokens)).where(Message.role == "assistant")
    if conv_id is not None:
        q = q.where(Message.conversation_id == conv_id)
    u = Usage()
    for model, p, c, o in s.execute(q.group_by(Message.model)):
        p, c, o = int(p or 0), int(c or 0), int(o or 0)
        u.prompt, u.cached, u.completion = u.prompt + p, u.cached + c, u.completion + o
        if p or o:
            cost = pricing.cost(model, p, c, o)
            if cost is None:
                u.cost_known = False
            else:
                u.cost += cost
    iq = select(func.count(Attachment.id)).join(Message).where(Attachment.kind == "generated")
    if conv_id is not None:
        iq = iq.where(Message.conversation_id == conv_id)
    u.images = int(s.scalar(iq) or 0)
    u.cost += u.images * config.IMAGE_PRICE
    return u


def context_used(path: list[Message]) -> tuple[int, str | None]:
    """Prompt tokens of the latest reply on this branch (= how full the context window is)."""
    for m in reversed(path):
        if m.role == "assistant" and m.prompt_tokens:
            return m.prompt_tokens + m.completion_tokens, m.model
    return 0, None
