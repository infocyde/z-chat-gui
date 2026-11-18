"""Turning text into vectors, so the lorebook can be searched by meaning as well as by words.

Uses fastembed with a small English model that runs on the CPU (no GPU, no PyTorch). The model is
downloaded once (about 70 MB) into data/models and loaded the first time it's needed; main.py warms
it up in the background at startup so the first reply doesn't wait for the download.

Vectors are stored as float32 bytes, the format sqlite-vec's vec_distance_cosine reads.
"""

from __future__ import annotations

import threading

import numpy as np

from . import config

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384

_model = None
_model_lock = threading.Lock()  # warm-up runs in a thread; don't load the model twice


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding  # imported here: it's slow to import and tests never need it

            _model = TextEmbedding(MODEL_NAME, cache_dir=str(config.DATA_DIR / "models"))
        return _model


def _to_bytes(vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def passage_text(name: str, keys: str, text: str) -> str:
    """What gets embedded for a lorebook entry: 'Mire Hag. hag, witch. Swamp witch...'"""
    return ". ".join(part for part in (name, keys, text) if part)


def embed_passages(texts: list[str]) -> list[bytes]:
    """Vectors for lorebook entries (the things being searched)."""
    return [_to_bytes(v) for v in _get_model().passage_embed(texts)]


def embed_query(text: str) -> bytes:
    """The vector for what's being searched for (bge models add a short instruction to queries)."""
    return _to_bytes(next(iter(_get_model().query_embed([text]))))


def warm_up() -> None:
    """Load (and on first run, download) the model."""
    embed_query("warm up")
