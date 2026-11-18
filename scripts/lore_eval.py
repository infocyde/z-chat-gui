"""Does searching by meaning help? Compares how well the lorebook search finds the right entry for a
scene, using word search only (BM25), meaning only (vectors), and both merged (what zchat uses).

    uv run python scripts/lore_eval.py

Uses the real embedding model (downloaded once into data/models) and a throwaway in-memory database,
so it never touches data/zchat.db. The sample world is scripts/lore_eval_world.json: 15 entries and
20 scenes, most of which describe the thing without using its name or keys.

hit@1: the right entry ranked first. hit@4: in the top 4 (zchat sends up to 4 entries by default).

Results (2026-11-18, BAAI/bge-small-en-v1.5, 15 entries, 20 scenes):

    method               hit@1   hit@4   extra entries sent per scene
    BM25 (words only)    15/20   18/20   1.5
    hybrid, min 0.4      17/20   19/20   3.0
    hybrid, min 0.45     17/20   19/20   2.9
    hybrid, min 0.5      17/20   19/20   1.9   <- the default in config.yaml
    hybrid, min 0.55     15/20   18/20   1.6
    hybrid, min 0.6      15/20   18/20   1.5   (vectors add nothing: bge-small scores most text 0.5-0.65)
    hybrid, min 0.7      15/20   18/20   1.5

So meaning search helps (2 more scenes get the right entry first, 1 more gets it at all), and 0.5 is
where it helps without dragging in much noise. Both methods still miss "Something with long fingers
rises out of the bog" (Mire Hag). A small world and few scenes: treat the numbers as a direction,
not a benchmark. Re-run after changing the model or the search.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import sqlite_vec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `zchat` imports when run as a script

from zchat import embeddings, lore  # noqa: E402

WORLD = Path(__file__).with_name("lore_eval_world.json")
THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
TOP = 4


def build_db(entries: list[dict]) -> sqlite3.Connection:
    """The same FTS5 table and weights as zchat, plus the entry embeddings, in memory."""
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.execute("CREATE VIRTUAL TABLE lore_fts USING fts5(name, keys, text, "
               "tokenize='porter unicode61 remove_diacritics 2')")
    db.execute("CREATE TABLE vectors (id INTEGER PRIMARY KEY, embedding BLOB)")
    vectors = embeddings.embed_passages([embeddings.passage_text(e["name"], e["keys"], e["text"]) for e in entries])
    for i, (e, vector) in enumerate(zip(entries, vectors), start=1):
        db.execute("INSERT INTO lore_fts(rowid, name, keys, text) VALUES (?, ?, ?, ?)", (i, e["name"], e["keys"], e["text"]))
        db.execute("INSERT INTO vectors VALUES (?, ?)", (i, vector))
    return db


def bm25_ranking(db: sqlite3.Connection, scene: str) -> list[int]:
    words = lore.query_words(scene)
    if not words:
        return []
    match = " OR ".join(f'"{w}"' for w in words)
    rows = db.execute("SELECT rowid FROM lore_fts WHERE lore_fts MATCH ? ORDER BY bm25(lore_fts, 10.0, 5.0, 1.0) "
                      "LIMIT 20", (match,))
    return [r[0] for r in rows]


def vector_ranking(db: sqlite3.Connection, scene: str, min_similarity: float) -> list[int]:
    q = embeddings.embed_query(scene)
    rows = db.execute("SELECT id FROM vectors WHERE vec_distance_cosine(embedding, ?) <= ? "
                      "ORDER BY vec_distance_cosine(embedding, ?) LIMIT 20", (q, 1 - min_similarity, q))
    return [r[0] for r in rows]


def score(rankings: list[list[int]], expected: list[int]) -> tuple[int, int]:
    hit1 = sum(1 for r, want in zip(rankings, expected) if r[:1] == [want])
    hit4 = sum(1 for r, want in zip(rankings, expected) if want in r[:TOP])
    return hit1, hit4


def main() -> None:
    world = json.loads(WORLD.read_text(encoding="utf-8"))
    entries, scenes = world["entries"], world["scenes"]
    names = {i: e["name"] for i, e in enumerate(entries, start=1)}
    ids = {e["name"]: i for i, e in enumerate(entries, start=1)}
    expected = [ids[s["expected"]] for s in scenes]
    db = build_db(entries)
    n = len(scenes)

    bm25 = [bm25_ranking(db, s["scene"]) for s in scenes]
    print(f"{len(entries)} entries, {n} scenes\n")
    print(f"{'method':<28}{'hit@1':>8}{'hit@4':>8}")
    h1, h4 = score(bm25, expected)
    print(f"{'BM25 (words only)':<28}{h1:>5}/{n}{h4:>5}/{n}")

    results = {}
    for t in THRESHOLDS:
        vec = [vector_ranking(db, s["scene"], t) for s in scenes]
        hybrid = [lore.fuse([b, v]) for b, v in zip(bm25, vec)]
        results[t] = (vec, hybrid)
        v1, v4 = score(vec, expected)
        f1, f4 = score(hybrid, expected)
        print(f"{f'vectors, min {t}':<28}{v1:>5}/{n}{v4:>5}/{n}")
        print(f"{f'hybrid,  min {t}':<28}{f1:>5}/{n}{f4:>5}/{n}")

    # How much noise comes along: entries sent per scene (up to TOP) that aren't the expected one.
    print(f"\nExtra entries sent per scene (top {TOP}, not the expected one), average:")
    print(f"  BM25: {sum(len([x for x in r[:TOP] if x != w]) for r, w in zip(bm25, expected)) / n:.1f}")
    for t, (_, hybrid) in results.items():
        extra = sum(len([x for x in r[:TOP] if x != w]) for r, w in zip(hybrid, expected)) / n
        print(f"  hybrid, min {t}: {extra:.1f}")

    print("\nMissed (not in the top 4):")
    for label, rankings in [("BM25", bm25), ("hybrid, min 0.5", results[0.5][1])]:
        missed = [s["scene"] for s, r, want in zip(scenes, rankings, expected) if want not in r[:TOP]]
        print(f"  {label}: {len(missed)}")
        for scene in missed:
            print(f"    - {scene}")


if __name__ == "__main__":
    main()
