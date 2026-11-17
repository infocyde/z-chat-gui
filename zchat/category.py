"""Scenario categories: cleaning what the user typed, and asking the AI to pick some.

Categories are stored on a scenario as one comma-separated string, e.g. "Fantasy, Mystery".
"""

from __future__ import annotations

import asyncio

from . import zai

# Offered as one-click suggestions in the scenario editor, and steered towards when the AI picks.
COMMON = ["Fantasy", "Sci-Fi", "Horror", "Mystery", "Romance", "Adventure", "Comedy", "Drama",
          "Thriller", "Historical", "Post-Apocalyptic", "Cyberpunk", "Slice of Life", "Superhero",
          "Western", "Dungeon Crawl", "Survival", "Noir"]

MAX_LENGTH = 300  # the database column size
MAX_CATEGORY_LENGTH = 40

SUGGEST_PROMPT = (
    "Pick 1 to 3 genre categories for the story scenario below. Use these when they fit: {known}. "
    "Reply with the categories only, separated by commas, nothing else.\n\n"
    "<scenario>\nName: {name}\nDescription: {description}\n\n{system_prompt}\n</scenario>"
)


def split(text: str) -> list[str]:
    """'Fantasy, Horror' -> ['Fantasy', 'Horror']"""
    return [part.strip() for part in text.split(",") if part.strip()]


def clean(text: str, known: list[str]) -> str:
    """Tidy a typed list: trims spaces, drops repeats, and matches the spelling of a known category
    ('sci-fi' becomes 'Sci-Fi'), so the same category isn't stored three different ways."""
    spelling = {k.lower(): k for k in known}
    result = []
    for name in split(text):
        name = " ".join(name.split())[:MAX_CATEGORY_LENGTH]
        name = spelling.get(name.lower(), name)
        if name.lower() not in [r.lower() for r in result]:
            result.append(name)
    joined = ", ".join(result)
    # Never cut a category in half to fit the column: drop whole ones from the end instead.
    while len(joined) > MAX_LENGTH:
        result.pop()
        joined = ", ".join(result)
    return joined


async def suggest(name: str, description: str, system_prompt: str, known: list[str]) -> str:
    """Ask the AI for categories. Returns "" if it fails or replies with something that isn't a list."""
    prompt = SUGGEST_PROMPT.format(known=", ".join(known), name=name, description=description,
                                   system_prompt=system_prompt[:3000])
    try:
        reply = await asyncio.wait_for(zai.complete([{"role": "user", "content": prompt}], max_tokens=40),
                                       timeout=30)
    except Exception:
        return ""
    reply = reply.strip().rstrip(".")
    picked = split(reply)
    # A sentence instead of a list means the model ignored the instructions.
    if not picked or len(picked) > 5 or any(len(c) > MAX_CATEGORY_LENGTH for c in picked):
        return ""
    return clean(reply, known)
