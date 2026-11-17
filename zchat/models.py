"""SQLAlchemy ORM models.

Conversations are stored as a *tree* of messages (parent_id), so editing a message or
regenerating a reply creates a sibling branch instead of overwriting history.
`Conversation.active_leaf_id` marks the branch currently shown.

RAG-readiness: every message has a stable id, a timestamp and a conversation/scenario link,
so a future memory index (Chroma) can key embeddings by message id and filter by scenario.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(String(500), default="")
    # Comma-separated, e.g. "Fantasy, Mystery" (see category.py)
    categories: Mapped[str] = mapped_column(String(300), default="", server_default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    conversations: Mapped[list[Conversation]] = relationship(back_populates="scenario")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    scenario_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenarios.id", ondelete="SET NULL"), nullable=True, index=True)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(64))
    thinking: Mapped[bool] = mapped_column(Boolean, default=False)  # on/off, for models with a thinking switch
    # low | high | max, for models that always think (GLM-5.3 family)
    reasoning_effort: Mapped[str] = mapped_column(String(16), default="low", server_default="low")
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Not a FK on purpose: avoids a circular constraint with messages.
    active_leaf_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    forked_from_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    scenario: Mapped[Scenario | None] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=True, index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    # text | image (assistant reply is a generated image) | illustrate (user asked to illustrate the scene)
    kind: Mapped[str] = mapped_column(String(16), default="text")
    content: Mapped[str] = mapped_column(Text, default="")
    reasoning: Mapped[str] = mapped_column(Text, default="")
    # "Story so far": a summary of the branch from the start up to and including this message.
    # Replies after it are sent this summary instead of those older messages (see story.py).
    summary: Mapped[str] = mapped_column(Text, default="", server_default="")
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # done | pending | streaming | stopped | error
    status: Mapped[str] = mapped_column(String(16), default="done")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    attachments: Mapped[list[Attachment]] = relationship(
        back_populates="message", cascade="all, delete-orphan", passive_deletes=True,
        order_by="Attachment.id")


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # document | image | generated
    filename: Mapped[str] = mapped_column(String(300))  # display name
    path: Mapped[str] = mapped_column(String(500))  # path relative to project root
    mime: Mapped[str] = mapped_column(String(100), default="application/octet-stream")
    size: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, default="")  # extracted document text
    note: Mapped[str] = mapped_column(String(300), default="")  # e.g. "OCR", "truncated"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    message: Mapped[Message] = relationship(back_populates="attachments")
