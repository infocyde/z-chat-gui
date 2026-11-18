"""Local embeddings (fastembed) and the sqlite-vec extension. The real model is never loaded here."""

import struct

import pytest
from sqlalchemy import text

from zchat import embeddings
from zchat.models import LoreVersion


def test_sqlite_vec_is_loaded(s):
    a = struct.pack("3f", 1, 0, 0)
    b = struct.pack("3f", 0, 1, 0)
    assert s.execute(text("SELECT vec_distance_cosine(:a, :b)"), {"a": a, "b": b}).scalar() == pytest.approx(1.0)


def test_lore_versions_can_store_an_embedding(s):
    assert "embedding" in LoreVersion.__table__.columns


def test_passage_text():
    assert embeddings.passage_text("Mire Hag", "hag", "Swamp witch.") == "Mire Hag. hag. Swamp witch."
    assert embeddings.passage_text("Mire Hag", "", "Swamp witch.") == "Mire Hag. Swamp witch."


def test_model_settings():
    assert embeddings.MODEL_NAME == "BAAI/bge-small-en-v1.5" and embeddings.DIMENSIONS == 384


def test_is_ready_once_the_model_has_loaded(monkeypatch):
    monkeypatch.setattr(embeddings, "_model", None)
    assert not embeddings.is_ready()
    monkeypatch.setattr(embeddings, "_model", "loaded")
    assert embeddings.is_ready()
