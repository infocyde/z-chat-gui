import httpx
import pytest
import respx

from zchat import pricing

# Trimmed copy of https://docs.z.ai/guides/overview/pricing.md (Nov 2026)
DOCS = r"""
| Model | Input | Cached Input | Cached Input Storage | Output |
| :- | :- | :- | :- | :- |
| GLM-5.3 | \$1.4 | \$0.26 | Limited-time Free | \$4.4 |
| GLM-5.3-Flash | \$0.15 | \$0.03 | Limited-time Free | \$0.50 |

| Tool | Cost |
| :- | :- |
| Web Search | \$0.01 / use |

> | GLM-4-32B-0414-128K | \$0.1 | - | - | \$0.1 |
> | GLM-4.7-Flash | Free | Free | Free | Free |
> | GLM-OCR | \$0.03 | \\ | \\ | \$0.03 |
"""


@pytest.fixture(autouse=True)
def no_live_prices(monkeypatch):
    monkeypatch.setattr(pricing, "live_prices", {})
    monkeypatch.setattr(pricing, "checked_at", None)


def test_parse_reads_model_rows_only():
    prices = pricing.parse(DOCS)
    assert prices["glm-5.3"] == pricing.Price(1.4, 0.26, 4.4)
    assert prices["glm-5.3-flash"] == pricing.Price(0.15, 0.03, 0.5)
    assert prices["glm-4-32b-0414-128k"] == pricing.Price(0.1, None, 0.1)
    assert prices["glm-4.7-flash"].free
    assert prices["glm-ocr"] == pricing.Price(0.03, None, 0.03)
    assert "web search" not in prices


def test_cost_uses_cached_price_and_falls_back_to_config():
    # glm-4.7 isn't loaded from the docs here, so the config price is used: 0.6 in, 0.11 cached, 2.2 out
    assert pricing.cost("glm-4.7", 1_000_000, 500_000, 1_000_000) == pytest.approx(0.3 + 0.055 + 2.2)
    assert pricing.cost("glm-4.7-flash", 1000, 0, 1000) == 0.0
    assert pricing.cost("mystery-model", 1000, 0, 1000) is None


@respx.mock
async def test_refresh_replaces_config_prices_and_saves_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "CACHE_FILE", tmp_path / "prices.json")
    respx.get(pricing.PRICING_URL).mock(return_value=httpx.Response(200, text=DOCS.replace("1.4", "2.0")))

    await pricing.refresh()

    assert pricing.price("glm-5.3").input == 2.0
    assert pricing.checked_at is not None
    pricing.live_prices.clear()
    pricing.load_cache()  # what the next app start does
    assert pricing.price("glm-5.3").input == 2.0


@respx.mock
async def test_refresh_rejects_a_page_without_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "CACHE_FILE", tmp_path / "prices.json")
    respx.get(pricing.PRICING_URL).mock(return_value=httpx.Response(200, text="# Pricing moved"))

    with pytest.raises(ValueError):
        await pricing.refresh()
    assert pricing.price("glm-5.3").input == 1.4  # config fallback still there
