"""Lorebooks: facts about the world (characters, places, monsters...) that are sent to the narrator
only when the story touches them.

A scenario holds the starting lorebook. Starting a chat copies it into the chat (seed_chat), and
from then on the chat's lorebook is its own. Each entry's text lives in versions (models.LoreVersion):
a version with no message applies to every branch; one tied to a message was made by the lorebook
update after that reply, so only branches through that reply see it, the same way "Story so far"
summaries work.

Before each reply, find_for_reply searches the branch's current entries two ways: by words, with
SQLite's full-text index (FTS5, ranked by BM25), and by meaning, comparing embeddings (see
embeddings.py) with sqlite-vec. The two rankings are merged (fuse) and the best few are picked.
lore_block turns them into the note that chat.build_messages puts in front of the newest user message.

This module must not import zchat.chat (chat imports this one).
"""

from __future__ import annotations

import logging
import re

from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from . import config, embeddings
from .models import Conversation, LoreEntry, LoreVersion, Message, Scenario

# Common words that would match almost every entry. BM25 already counts them for little,
# but leaving them out keeps the search query short.
STOPWORDS = {
    "the", "and", "you", "your", "are", "was", "were", "with", "that", "this", "for", "but", "not",
    "have", "has", "had", "his", "her", "him", "she", "they", "them", "their", "its", "from", "into",
    "what", "when", "where", "who", "how", "all", "any", "can", "could", "would", "should", "will",
    "just", "then", "than", "there", "here", "out", "about", "over", "our", "one", "some", "now",
}
MAX_QUERY_WORDS = 64
MAX_RESULTS = 20
MAX_QUERY_CHARS = 2000  # what's embedded for the meaning search: the end of the recent messages
RRF_K = 60  # reciprocal rank fusion constant (the usual value)

log = logging.getLogger(__name__)

LORE_HEADING = ("Background from the lorebook, for the narrator. Not written by the user: use what fits "
                "the scene and never mention these notes.")


def split_keys(keys: str) -> list[str]:
    """'hag, witch' -> ['hag', 'witch']"""
    return [part.strip() for part in keys.split(",") if part.strip()]


def seed_chat(s: Session, conv: Conversation, scenario: Scenario) -> None:
    """Copy the scenario's lorebook into a new chat. These versions apply to every branch."""
    for item in scenario.lore:
        entry = LoreEntry(conversation_id=conv.id)
        entry.versions.append(LoreVersion(name=item.name, keys=item.keys, text=item.text, enabled=item.enabled))
        s.add(entry)
    s.flush()


def current_versions(s: Session, conv_id: int, path: list[Message]) -> list[LoreVersion]:
    """The version of each entry that is in effect on this branch (disabled ones included)."""
    position = {m.id: i for i, m in enumerate(path)}
    versions = s.scalars(select(LoreVersion).join(LoreEntry).where(LoreEntry.conversation_id == conv_id))
    newest: dict[int, tuple[int, LoreVersion]] = {}  # entry id -> (position on the branch, version)
    for v in versions:
        if v.message_id is None:
            pos = -1  # applies from the start
        elif v.message_id in position:
            pos = position[v.message_id]
        else:
            continue  # made on another branch
        best = newest.get(v.entry_id)
        if best is None or (pos, v.id) > (best[0], best[1].id):
            newest[v.entry_id] = (pos, v)
    return sorted((v for _, v in newest.values()), key=lambda v: v.name.lower())


def query_words(story_text: str) -> list[str]:
    """The words worth searching for. Only letters, digits and _ survive, so FTS5 syntax can't sneak in."""
    words = []
    for word in re.findall(r"\w+", story_text.lower()):
        if len(word) < 3 or word in STOPWORDS or word in words:
            continue
        words.append(word)
    return words[:MAX_QUERY_WORDS]


def search_fts(s: Session, version_ids: list[int], words: list[str]) -> list[int]:
    """Version ids that share a word with `words`, best first (BM25; a match in the name counts most)."""
    if not version_ids or not words:
        return []
    # Any word may match (OR); FTS5 would otherwise require all of them.
    match = " OR ".join(f'"{word}"' for word in words)
    query = text("SELECT rowid FROM lore_fts WHERE lore_fts MATCH :match AND rowid IN :ids "
                 "ORDER BY bm25(lore_fts, 10.0, 5.0, 1.0) LIMIT :limit"
                 ).bindparams(bindparam("ids", expanding=True))
    return list(s.scalars(query, {"match": match, "ids": version_ids, "limit": MAX_RESULTS}))


def ensure_embeddings(s: Session, versions: list[LoreVersion]) -> None:
    """Embed the versions that don't have an embedding yet (new, or edited since), in one call."""
    missing = [v for v in versions if v.embedding is None]
    if not missing:
        return
    vectors = embeddings.embed_passages([embeddings.passage_text(v.name, v.keys, v.text) for v in missing])
    for v, vector in zip(missing, vectors):
        v.embedding = vector
    s.flush()


def search_vectors(s: Session, version_ids: list[int], query_text: str) -> list[int]:
    """Version ids whose meaning is close to `query_text`, best first, at least LORE_MIN_SIMILARITY alike."""
    if not version_ids or not query_text.strip():
        return []
    query = text("SELECT id FROM lore_versions "
                 "WHERE id IN :ids AND embedding IS NOT NULL AND vec_distance_cosine(embedding, :q) <= :max_distance "
                 "ORDER BY vec_distance_cosine(embedding, :q) LIMIT :limit"
                 ).bindparams(bindparam("ids", expanding=True))
    params = {"ids": version_ids, "q": embeddings.embed_query(query_text),
              "max_distance": 1 - config.LORE_MIN_SIMILARITY,  # cosine distance = 1 - cosine similarity
              "limit": MAX_RESULTS}
    return list(s.scalars(query, params))


def fuse(rankings: list[list[int]]) -> list[int]:
    """Merge several best-first lists into one (reciprocal rank fusion).

    Each id scores 1 / (60 + its rank) in every list it appears in, and the scores are added, so an
    entry that ranks well in both the word search and the meaning search comes out on top.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)  # sorted() is stable: ties keep order


def recent_text(path: list[Message]) -> str:
    """The text of the newest messages, which is what the lorebook is searched with."""
    return "\n".join(m.content for m in path[-config.LORE_SCAN_MESSAGES:] if m.content)


def estimate_tokens(v: LoreVersion) -> int:
    return len(v.name + v.text) // 4


def find_for_reply(s: Session, conv: Conversation, path: list[Message]) -> list[LoreVersion]:
    """The lorebook entries to send with the next reply: the best matches, within the caps."""
    candidates = {v.id: v for v in current_versions(s, conv.id, path) if v.enabled}
    story_text = recent_text(path)
    rankings = [search_fts(s, list(candidates), query_words(story_text))]
    if config.LORE_VECTORS and candidates:
        try:
            ensure_embeddings(s, list(candidates.values()))
            rankings.append(search_vectors(s, list(candidates), story_text[-MAX_QUERY_CHARS:]))
        except Exception as e:  # e.g. the model couldn't be downloaded: words alone still work
            log.warning("Lorebook meaning search failed, using word search only: %s", e)
    ranked = fuse(rankings)
    chosen, tokens = [], 0
    for version_id in ranked:
        if len(chosen) >= config.LORE_MAX_ENTRIES:
            break
        v = candidates[version_id]
        if tokens + estimate_tokens(v) > config.LORE_MAX_TOKENS:
            continue  # too big to fit; a smaller one further down may still fit
        chosen.append(v)
        tokens += estimate_tokens(v)
    return chosen


def lore_block(versions: list[LoreVersion]) -> str:
    """The note sent in front of the newest user message, or "" when there's nothing to send."""
    if not versions:
        return ""
    entries = "\n\n".join(f"### {v.name}\n{v.text}" for v in versions)
    return f"<world_notes>\n{LORE_HEADING}\n\n{entries}\n</world_notes>"


def copy_to_fork(s: Session, old_conv_id: int, new_conv_id: int, message_ids: dict[int, int]) -> None:
    """Copy a chat's lorebook into its fork: the versions on the forked branch, pointed at the copied messages."""
    old_entries = s.scalars(select(LoreEntry).where(LoreEntry.conversation_id == old_conv_id).order_by(LoreEntry.id))
    for old in old_entries:
        copies = []
        # Queried rather than old.versions: that list may have been loaded before newer versions were added.
        for v in s.scalars(select(LoreVersion).where(LoreVersion.entry_id == old.id).order_by(LoreVersion.id)):
            if v.message_id is not None and v.message_id not in message_ids:
                continue  # not on the forked branch
            copies.append(LoreVersion(message_id=message_ids.get(v.message_id), name=v.name, keys=v.keys,
                                      text=v.text, enabled=v.enabled, embedding=v.embedding,
                                      created_at=v.created_at))
        if copies:
            s.add(LoreEntry(conversation_id=new_conv_id, versions=copies))
    s.flush()
