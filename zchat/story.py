"""Story so far: keeping long chats inside the model's context window.

When a branch gets long, the older messages are summarized into a "Story so far" note, saved on
the last message it covers (Message.summary). From then on, chat.build_messages sends that note
instead of those messages. Because the note lives on a message in the tree, branching just works:
branches that split off before that message never see it, and everything after it inherits it.

Each new summary rolls the previous one forward (previous summary + the messages since), so the
note always covers the whole story up to its message.
"""

from __future__ import annotations

from . import config, zai
from .chat import assistant_content, context_used, latest_summary, path_to
from .db import session_scope
from .models import Message

SUMMARY_SYSTEM = (
    "You keep the running summary of a long interactive story between a user and a narrator. "
    "Write the \"Story so far\" that replaces the older messages, so the narrator can carry on "
    "without them. Cover: what has happened (key events, in order); characters (names, roles, "
    "relationships, what they want, how they feel about the user's character); places; important "
    "items and facts; promises, secrets and open threads; and exactly where the story stands now. "
    "Keep every name exact. Be concrete and leave out flavor that won't matter later. Use short "
    "headings and plain notes, under 1000 words. Reply with the summary only."
)

SUMMARY_REQUEST = """<setting>
{setting}
</setting>

<previous_summary>
{previous}
</previous_summary>

<new_messages>
{transcript}
</new_messages>

Write the updated Story so far: the previous summary (if any) plus everything in the new messages."""


def summary_due(path: list[Message], model: str) -> bool:
    """True when the branch is close enough to the context limit that older messages should be summarized."""
    if config.SUMMARY_AT_TOKENS == 0:
        return False
    used, _ = context_used(path)
    limit = min(config.SUMMARY_AT_TOKENS, 0.8 * config.model_info(model).context)
    return used >= limit


def pick_cutoff(path: list[Message]) -> int | None:
    """Index of the last message to summarize, or None if there's nothing new to summarize.

    The newest `keep_recent` messages stay word for word, and the cut lands on a narrator reply so
    the messages that are still sent start with the user's turn.
    """
    checkpoint = latest_summary(path)
    already_summarized = path.index(checkpoint) if checkpoint else -1
    cutoff = len(path) - config.SUMMARY_KEEP_RECENT - 1
    while cutoff > already_summarized and path[cutoff].role != "assistant":
        cutoff -= 1
    if cutoff <= already_summarized:
        return None
    return cutoff


def transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        if m.role == "user":
            if m.kind == "illustrate":
                continue
            text = m.content
            if m.attachments:
                text += "\n[Attached: " + ", ".join(a.filename for a in m.attachments) + "]"
            lines.append(f"User: {text.strip()}")
        else:
            text = assistant_content(m)
            if text:
                lines.append(f"Narrator: {text}")
    return "\n\n".join(lines)


async def summarize_branch(leaf_id: int | None) -> int | None:
    """Summarize the older messages on the branch ending at `leaf_id` and save the summary.

    Returns the id of the message the summary was saved on, or None if there was nothing to
    summarize yet. Raises (zai.ZaiError, network errors) if the model call fails.
    """
    with session_scope() as s:
        path = [m for m in path_to(s, leaf_id) if m.status not in ("pending", "streaming")]
        cutoff = pick_cutoff(path)
        if cutoff is None:
            return None
        previous = latest_summary(path[:cutoff + 1])
        start = path.index(previous) + 1 if previous else 0
        request = SUMMARY_REQUEST.format(
            setting=path[0].conversation.system_prompt.strip()[:4000] or "(none)",
            previous=previous.summary if previous else "(none: this is the first summary)",
            transcript=transcript(path[start:cutoff + 1]))
        # The chat's own model: its context window has already held all of this text.
        model = config.usable_model(path[0].conversation.model)
        cutoff_id = path[cutoff].id

    summary = await zai.complete([{"role": "system", "content": SUMMARY_SYSTEM},
                                  {"role": "user", "content": request}],
                                 model=model, max_tokens=2500, timeout=300)
    if not summary.strip():
        raise zai.ZaiError("The model returned an empty summary.")

    with session_scope() as s:
        s.get(Message, cutoff_id).summary = summary.strip()
    return cutoff_id
