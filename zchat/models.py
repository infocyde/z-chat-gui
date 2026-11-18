"""SQLAlchemy ORM models.

Conversations are stored as a *tree* of messages (parent_id), so editing a message or
regenerating a reply creates a sibling branch instead of overwriting history.
`Conversation.active_leaf_id` marks the branch currently shown.

RAG-readiness: every message has a stable id, a timestamp and a conversation/scenario link,
so a future memory index (Chroma) can key embeddings by message id and filter by scenario.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
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
    # The starting lorebook: copied into each chat started from this scenario (see lore.seed_chat).
    lore: Mapped[list[ScenarioLore]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan", passive_deletes=True,
        order_by="ScenarioLore.id")


class ScenarioLore(Base):
    """One entry of a scenario's starting lorebook."""

    __tablename__ = "scenario_lore"

    id: Mapped[int] = mapped_column(primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    keys: Mapped[str] = mapped_column(String(500), default="")  # comma-separated, e.g. "hag, witch"
    text: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    scenario: Mapped[Scenario] = relationship(back_populates="lore")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    scenario_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenarios.id", ondelete="SET NULL"), nullable=True, index=True)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(64))
    thinking: Mapped[bool] = mapped_column(Boolean, default=False)
    # The Lorekeeper switch: search, send and update the lorebook for this chat.
    lore_on: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")  # on/off, for models with a thinking switch
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
    lore_entries: Mapped[list[LoreEntry]] = relationship(
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
    # Names of the lorebook entries sent with this reply, one per line (shown as "Lore: ...").
    lore_used: Mapped[str] = mapped_column(Text, default="", server_default="")
    # Set when the lorebook was updated right after this reply, e.g. "Added: Lantern\nUpdated: Gloomfang".
    lore_note: Mapped[str] = mapped_column(Text, default="", server_default="")
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


class LoreEntry(Base):
    """One thing in a chat's lorebook (a character, a place, a monster...).

    Its name and text live in versions: the lorebook updates itself as the story goes, and each
    branch of the chat sees the newest version made on that branch (see lore.current_versions).
    """

    __tablename__ = "lore_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)

    conversation: Mapped[Conversation] = relationship(back_populates="lore_entries")
    versions: Mapped[list[LoreVersion]] = relationship(
        back_populates="entry", cascade="all, delete-orphan", passive_deletes=True, order_by="LoreVersion.id")


class LoreVersion(Base):
    __tablename__ = "lore_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("lore_entries.id", ondelete="CASCADE"), index=True)
    # None: applies to every branch (copied from the scenario, or added/edited by hand).
    # Set: made by the lorebook update after that reply, so only branches through it see it.
    message_id: Mapped[int | None] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    keys: Mapped[str] = mapped_column(String(500), default="")  # comma-separated
    text: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # What the entry means, as 384 float32 numbers (see embeddings.py). None until it's been embedded;
    # cleared when the name, keys or text change.
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    entry: Mapped[LoreEntry] = relationship(back_populates="versions")
