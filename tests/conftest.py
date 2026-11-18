"""Point the app at a throwaway data directory *before* anything imports zchat.config."""

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="zchat-test-"))
os.environ["ZCHAT_DATA_DIR"] = str(_tmp / "data")
os.environ["ZCHAT_IMAGE_DIR"] = str(_tmp / "image_output")
os.environ["ZCHAT_UPLOAD_DIR"] = str(_tmp / "uploads")
# Never hit the real API from tests, and don't depend on what's in the real .env.
os.environ["ZCHAT_ZAI_API_KEY"] = "test-key"
os.environ["ZCHAT_ZAI_BASE_URL"] = "https://api.z.ai/api/paas/v4"

import pytest  # noqa: E402

from zchat.db import upgrade_db  # noqa: E402

upgrade_db()  # builds the schema through the real Alembic migrations


@pytest.fixture
def s():
    from zchat.db import session_scope

    with session_scope() as session:
        yield session


# Words the fake embedder treats as meaning the same thing.
FAKE_SYNONYMS = {"bog": "swamp", "marsh": "swamp", "fen": "swamp"}


def fake_vector(text: str) -> bytes:
    """A stand-in for the embedding model: each word lights up one of 384 slots (synonyms share a slot)."""
    import re
    import zlib

    import numpy as np

    v = np.zeros(384, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        v[zlib.crc32(FAKE_SYNONYMS.get(word, word).encode()) % 384] += 1
    norm = np.linalg.norm(v)
    return (v / norm if norm else v).astype(np.float32).tobytes()


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """No test ever downloads or loads the real embedding model."""
    from zchat import embeddings

    calls = {"passages": 0, "queries": 0}

    def embed_passages(texts):
        calls["passages"] += len(texts)
        return [fake_vector(t) for t in texts]

    def embed_query(text):
        calls["queries"] += 1
        return fake_vector(text)

    monkeypatch.setattr(embeddings, "_model", "fake model")  # counts as loaded
    monkeypatch.setattr(embeddings, "embed_passages", embed_passages)
    monkeypatch.setattr(embeddings, "embed_query", embed_query)
    return calls
