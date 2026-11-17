"""Per-token prices for each model.

Z.ai has no pricing API, but its docs serve the pricing page as Markdown for programs to read. We
fetch that once a day and cache it in data/prices.json. If the fetch or the parsing ever fails, the
prices typed into config.MODELS are used instead, so costs never disappear.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import httpx

from . import config

log = logging.getLogger(__name__)

PRICING_URL = "https://docs.z.ai/guides/overview/pricing.md"
CACHE_FILE = config.DATA_DIR / "prices.json"
MAX_AGE = timedelta(days=1)


@dataclass
class Price:
    """USD per 1M tokens. None = not listed."""
    input: float | None
    cached: float | None
    output: float | None

    @property
    def free(self) -> bool:
        return self.input == 0 and self.output == 0

    def cost(self, prompt: int, cached: int, completion: int) -> float | None:
        if self.input is None or self.output is None:
            return None
        # No cached price listed means no caching discount: cached tokens cost the normal input price.
        cached_price = self.cached if self.cached is not None else self.input
        uncached = max(prompt - cached, 0)
        return (uncached * self.input + cached * cached_price + completion * self.output) / 1_000_000


# Prices read from the docs, keyed by model id ("glm-5.3"). Empty until loaded.
live_prices: dict[str, Price] = {}
checked_at: datetime | None = None


# ---------------------------------------------------------------- reading the docs page

# A model row in the docs tables:  | GLM-5.3 | \$1.4 | \$0.26 | Limited-time Free | \$4.4 |
# (rows inside the "Other Pricing" accordion start with "> ")
_ROW = re.compile(r"^[>\s]*\|\s*(GLM-[^|]+?)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|\s*$", re.MULTILINE)


def _money(cell: str) -> float | None:
    """'\\$1.4' -> 1.4, 'Free' -> 0.0, '-' or '\\' -> None."""
    text = cell.replace("\\", "").replace("$", "").strip()
    if text.lower() == "free":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def parse(markdown: str) -> dict[str, Price]:
    prices = {}
    for name, price_in, price_cached, _storage, price_out in _ROW.findall(markdown):
        prices[name.lower()] = Price(_money(price_in), _money(price_cached), _money(price_out))
    return prices


def load_cache() -> None:
    global checked_at
    try:
        saved = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        live_prices.clear()
        live_prices.update({model: Price(**p) for model, p in saved["prices"].items()})
        checked_at = datetime.fromisoformat(saved["checked_at"])
    except (OSError, ValueError, KeyError, TypeError):
        pass  # no cache yet (or a broken one): config prices are used until the first refresh


async def refresh(force: bool = False) -> None:
    """Re-read the docs pricing page if our copy is more than a day old."""
    global checked_at
    if not force and checked_at and datetime.now() - checked_at < MAX_AGE:
        return
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(PRICING_URL)
        r.raise_for_status()
    prices = parse(r.text)
    if not prices:
        raise ValueError("No model prices found on the docs page (did its layout change?)")

    live_prices.clear()
    live_prices.update(prices)
    checked_at = datetime.now()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"checked_at": checked_at.isoformat(),
                                      "prices": {m: asdict(p) for m, p in prices.items()}}, indent=1),
                          encoding="utf-8")
    log.info("Loaded %d model prices from %s", len(prices), PRICING_URL)


async def refresh_forever() -> None:
    """Started with the app: keeps prices at most a day old."""
    while True:
        try:
            await refresh()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("Could not refresh prices, using saved ones: %s", e)
        await asyncio.sleep(6 * 3600)


# ---------------------------------------------------------------- lookups


def price(model_id: str | None) -> Price:
    """Docs price if we have one, otherwise the price typed into config.MODELS."""
    if model_id in live_prices:
        return live_prices[model_id]
    m = config.model_info(model_id)
    if m.free:
        return Price(0.0, 0.0, 0.0)
    return Price(m.price_in, m.price_cached, m.price_out)


def cost(model_id: str | None, prompt: int, cached: int, completion: int) -> float | None:
    return price(model_id).cost(prompt, cached, completion)


load_cache()
