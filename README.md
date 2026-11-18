# zchat

A personal web chat client for [Z.ai](https://docs.z.ai) (GLM models), built with FastHTML.

- **Chats** stream replies with optional "thinking" shown in a collapsible box.
- **Branching:** editing one of your messages or regenerating a reply creates a new branch (`‹ 2/3 ›`); nothing is overwritten. **Fork** copies a chat up to any message into a new chat. Assistant replies can be edited in place to steer a story.
- **Long stories (Story so far):** when a chat's context reaches 100k tokens (or 80% of the model's window, if smaller), the older messages are summarized before the next reply, and from then on the summary is sent instead of them. The newest 10 messages are always sent word for word. The summary shows as a "Story so far" box in the chat, where you can edit or remove it; the **Summarize** button in the header does it on demand. Each new summary builds on the previous one. It's saved on the last message it covers, so branches split off earlier don't see it and forks keep it. Both numbers are in `config.yaml`. Summary calls aren't counted in the token totals.
- **Models:** pick per chat and switch at any time; each reply records which model wrote it.
- **Scenarios:** reusable system prompts, managed on the Scenarios tab and picked on the new-chat page. You can also type a one-off system prompt.
- **Lorebooks:** each scenario can have a lorebook: entries for characters, places, monsters and items (a name, keys like `hag, witch`, the text, and on/off). Keep always-true basics in the system prompt; the lorebook is for everything else. Starting a chat copies the scenario's lorebook into the chat, and the chat's copy is its own (the **Lorebook** drawer under the system prompt). Before every reply, the newest messages are searched and only the best matches are sent, in front of your newest message. They aren't saved into the history, so long chats don't fill up with lore, and the cached part of the prompt stays the same. Each reply shows which entries went with it ("Lore: Mire Hag"). Limits are in `config.yaml` under `lorebook:`.
  - **Search** works two ways, merged: by words (SQLite full-text search, BM25; a match in the name or keys counts most) and by meaning, with a small local embedding model (`BAAI/bge-small-en-v1.5` via fastembed, CPU only, downloaded once into `data/models`, about 65 MB). So "something rises from the bog" can bring up the swamp witch. If the model can't load, word search carries on alone. `scripts/lore_eval.py` measures how well each way finds the right entry.
  - **It updates itself:** every `update_every_replies` replies (10 by default; 0 = never), before the next reply, the chat's model reads the lorebook and the story since the last update and adds or updates entries (it never deletes). A note after the reply shows what changed, with **Undo**. Updates follow branches like "Story so far": a branch that split off earlier doesn't see them, and forks keep them. Your own edits in the drawer apply to every branch.
  - **Draft from prompt** (scenario editor): the AI pulls the characters, places, monsters and items out of the system prompt as starter entries for you to edit.
  - **Import** SillyTavern World Info files, Character Card V2/V3 (JSON) lorebooks, and zchat's own lorebook exports. Only name, keys, text and on/off are used for now. Other lorebook settings (secondary keys, always-on, position, probability, …) are ignored. PNG character cards and NovelAI lorebooks aren't supported yet. Scenario export/import includes the lorebook.
- **Categories & search:** each scenario has comma-separated categories (Fantasy, Mystery, …). The editor offers common ones and every one you've used as one-click buttons; leave the box empty and the AI picks 1–3 when you save. The search box on the Scenarios page matches names, categories, descriptions and prompt text (every word must match); clicking a category tag on a card searches for it.
- **Images:** the *Image* toggle sends your text to `glm-image`; *Illustrate* writes an image prompt from the current scene. Images are downloaded to `image_output/` and shown in the chat (Z.ai's links expire after 30 days).
- **Documents:** attach PDF, DOCX, text/markdown, or images. Text is extracted locally; scanned PDFs fall back to `glm-ocr`. Images go to a vision model (switches to the free GLM-4.6V Flash automatically if needed).
- **Tokens:** per-reply input / cached / output counts, per-chat totals with estimated cost, a context-window bar, and an all-time total in the sidebar.

## Run it

```powershell
uv sync
uv run main.py          # http://localhost:5001
```

Settings are read from `..\.env` (the folder *above* this project), by design:

```ini
ZCHAT_ZAI_API_KEY=...                                     # zchat's own key (falls back to ZAI_API_KEY)
ZCHAT_ZAI_BASE_URL=https://api.z.ai/api/coding/paas/v4    # GLM Coding Plan key
# or https://api.z.ai/api/paas/v4 for a standard pay-as-you-go key (the default)
```

Z.ai keeps two kinds of keys apart: **Coding Plan** keys (monthly subscription, created under
GLM Coding Plan → Personal) only work on the `/coding/` URL; **standard** keys (API Keys page, billed
from the cash balance) only work on the normal URL. With a Coding Plan key the model list shows only
models the plan includes, and costs show "Coding Plan" instead of dollars. Image generation and OCR
are not part of the Coding Plan; they need pay-as-you-go credit and a standard key.

On the Coding Plan the sidebar shows the plan's 5-hour and weekly credit limits (with reset times) and
the last 7 days of tokens, from `api.z.ai/api/monitor/usage` (the same data as z.ai's usage page).
It asks Z.ai on every page load or reload, then by default after every 5 replies; set
`plan_usage` in `config.yaml` to change the count or to refresh every N seconds instead.

Z.ai has no pricing API, so prices (input, cached input, output per 1M tokens) are read from
`docs.z.ai/guides/overview/pricing.md` once a day and saved to `data/prices.json`. If that page can't be
read, the prices typed into `zchat/config.py` are used. The selected model's price shows next to the
model picker; hover it to see where it came from.

Set `ZCHAT_ENV_FILE` to read a different file.

App behavior (not secrets) lives in `config.yaml` in the project folder; restart the server after
editing it. Set `ZCHAT_SETTINGS_FILE` to use a different file.

## Database & migrations

SQLite lives at `data/zchat.db` (git-ignored). Schema is managed by Alembic; the app runs
`alembic upgrade head` automatically on startup.

After changing `zchat/models.py`:

```powershell
uv run alembic revision --autogenerate -m "describe the change"
uv run alembic upgrade head      # or just restart the app
```

## Tests

```powershell
uv run pytest
```

Tests use a temporary database (built through the real migrations) and fake the Z.ai API, so they cost nothing.

## Code map

| File | What it does |
|---|---|
| `main.py` | Routes (pages, sending, streaming, editing, scenarios, files) |
| `config.yaml` | App settings you can edit (e.g. how often the plan meters refresh) |
| `zchat/config.py` | Paths, API settings, `config.yaml` loading, model list with prices/context sizes |
| `zchat/models.py` | SQLAlchemy tables: scenarios, conversations, messages (a tree), attachments |
| `zchat/chat.py` | Conversation tree logic, building the API `messages`, token/cost totals |
| `zchat/generate.py` | Runs a pending reply (text or image) and yields progress events |
| `zchat/zai.py` | Z.ai HTTP client: streaming chat, image generation, OCR, Coding Plan quota |
| `zchat/story.py` | Story so far: when to summarize, what to summarize, and saving the summary |
| `zchat/category.py` | Scenario categories: the common list, tidying typed ones, asking the AI to pick |
| `zchat/scenario_json.py` | Scenario import/export: the JSON format and cleanup of imported text |
| `zchat/pricing.py` | Per-token prices, read daily from the docs pricing page (cached in `data/prices.json`) |
| `zchat/documents.py` | Upload handling and text extraction |
| `zchat/ui.py` | HTML components |
| `static/` | `app.css`, `app.js` (streaming + composer), favicon |

## Notes

- Default model: GLM-5.3 Flash on the Coding Plan (it always reasons; the Thinking switch sets low vs
  high effort), GLM-4.7 Flash (free) with a standard key.
- Z.ai caches repeated prompt prefixes automatically. Changing the system prompt mid-chat resets that cache.
- Prices in `config.py` are only a fallback for when the docs page can't be read; models with no known price show `$…+?`.

## Scenario import/export

On the Scenarios page: **Export** downloads the checked scenarios (all of them if none are checked), the
download icon on a card exports just that one, and **Import** takes one or more `.json` files. The format
is plain: one scenario is `{"name": "...", "description": "...", "categories": ["..."], "system_prompt": "..."}`
(categories optional) and several
are a list of those. Imports never overwrite: a name that already exists becomes "Name (2)". Files in
UTF-8, UTF-16 or old Windows encodings all work; invisible control characters are removed.

## Planned: campaign memory (RAG)

Not built yet. The intended shape: embed finished messages into a local **Chroma** collection keyed by
message id with `scenario_id` / `conversation_id` metadata, then retrieve relevant past events into the
prompt for long-running campaigns. Z.ai has no embeddings endpoint, so embeddings will be local.
