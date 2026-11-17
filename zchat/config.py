"""App configuration: paths, API settings, and the model catalog."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# The API key lives in a .env *outside* the project (one level up) by design.
# ZCHAT_ENV_FILE can point somewhere else; a project-local .env is an optional fallback.
ENV_FILE = Path(os.environ.get("ZCHAT_ENV_FILE", ROOT.parent / ".env"))
load_dotenv(ENV_FILE)
load_dotenv(ROOT / ".env")

# zchat-specific names win, so the shared .env can hold a different ZAI_API_KEY for other projects.
# Coding Plan keys only work with the coding base URL: https://api.z.ai/api/coding/paas/v4
ZAI_API_KEY = os.environ.get("ZCHAT_ZAI_API_KEY") or os.environ.get("ZAI_API_KEY", "")
ZAI_BASE_URL = (os.environ.get("ZCHAT_ZAI_BASE_URL") or os.environ.get("ZAI_BASE_URL")
                or "https://api.z.ai/api/paas/v4").rstrip("/")
# Coding Plan = monthly subscription: different model list, and usage isn't billed per token.
USING_CODING_PLAN = "/coding/" in ZAI_BASE_URL
# Coding Plan quota (5-hour and weekly limits) lives on the same host: https://api.z.ai/api/monitor/usage
ZAI_USAGE_URL = ZAI_BASE_URL.split("/api/")[0] + "/api/monitor/usage"

# ---------------------------------------------------------------- config.yaml (app behavior, no secrets)

SETTINGS_FILE = Path(os.environ.get("ZCHAT_SETTINGS_FILE", ROOT / "config.yaml"))
try:
    _settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8")) or {}
except FileNotFoundError:
    _settings = {}  # every setting has a default below

_plan = _settings.get("plan_usage") or {}
PLAN_REFRESH_BY = _plan.get("refresh_by", "replies")  # "replies" or "seconds"
PLAN_EVERY_REPLIES = max(1, int(_plan.get("every_replies", 5)))
PLAN_EVERY_SECONDS = max(10, int(_plan.get("every_seconds", 60)))
if PLAN_REFRESH_BY not in ("replies", "seconds"):
    raise ValueError(f"{SETTINGS_FILE}: plan_usage.refresh_by must be 'replies' or 'seconds', "
                     f"not {PLAN_REFRESH_BY!r}")

DATA_DIR = Path(os.environ.get("ZCHAT_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "zchat.db"
DATABASE_URL = os.environ.get("ZCHAT_DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")
IMAGE_DIR = Path(os.environ.get("ZCHAT_IMAGE_DIR", ROOT / "image_output"))
UPLOAD_DIR = Path(os.environ.get("ZCHAT_UPLOAD_DIR", ROOT / "uploads"))
ALEMBIC_INI = ROOT / "alembic.ini"

IMAGE_MODEL = "glm-image"
IMAGE_PRICE = 0.015  # USD per image
IMAGE_SIZES = ["1280x1280", "1568x1056", "1056x1568", "1728x960", "960x1728"]
OCR_MODEL = "glm-ocr"
# Image generation and OCR are NOT included in the Coding Plan; they need pay-as-you-go credit.

# Model used for chat titles and for turning a scene into an image prompt.
UTILITY_MODEL = "glm-5.3-flash" if USING_CODING_PLAN else "glm-4.7-flash"
# Vision model used automatically when a message includes images and the chosen model can't see.
FALLBACK_VISION_MODEL = "glm-5.3-flash" if USING_CODING_PLAN else "glm-4.6v-flash"

MAX_DOC_CHARS = 200_000

# reasoning_effort values Z.ai documents for models that always think (GLM-5.3 family)
EFFORT_LEVELS = ["low", "high", "max"]
DEFAULT_EFFORT = "low"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    context: int
    vision: bool = False
    # "toggle": thinking can be switched on/off. "always": model always reasons; you pick an effort level.
    thinking: str = "toggle"
    # USD per 1M tokens, used only until zchat/pricing.py has read the live docs page. None = unknown.
    price_in: float | None = None
    price_cached: float | None = None
    price_out: float | None = None
    free: bool = False
    # Tested Nov 2026 with a GLM Coding Lite key: the free Flash models and 5.3 FlashX aren't included.
    on_coding_plan: bool = True


# Context sizes and fallback prices from docs.z.ai (Nov 16, 2026).
MODELS: list[ModelInfo] = [
    ModelInfo("glm-4.7-flash", "GLM-4.7 Flash", 200_000, free=True, on_coding_plan=False),
    ModelInfo("glm-4.5-flash", "GLM-4.5 Flash", 128_000, free=True, on_coding_plan=False),
    ModelInfo("glm-4.6v-flash", "GLM-4.6V Flash", 128_000, vision=True, free=True, on_coding_plan=False),
    ModelInfo("glm-5.3", "GLM-5.3", 1_000_000, thinking="always", price_in=1.4, price_cached=0.26, price_out=4.4),
    ModelInfo("glm-5.3-flash", "GLM-5.3 Flash", 1_000_000, vision=True, thinking="always",
              price_in=0.15, price_cached=0.03, price_out=0.5),
    ModelInfo("glm-5.3-flashx", "GLM-5.3 FlashX", 1_000_000, vision=True, thinking="always",
              price_in=0.37, price_cached=0.075, price_out=1.25, on_coding_plan=False),
    ModelInfo("glm-5.2", "GLM-5.2", 1_000_000, price_in=1.4, price_cached=0.26, price_out=4.4),
    ModelInfo("glm-5.1", "GLM-5.1", 200_000, price_in=1.4, price_cached=0.26, price_out=4.4),
    ModelInfo("glm-5", "GLM-5", 200_000, price_in=1.0, price_cached=0.2, price_out=3.2),
    ModelInfo("glm-4.7", "GLM-4.7", 200_000, price_in=0.6, price_cached=0.11, price_out=2.2),
    ModelInfo("glm-4.6", "GLM-4.6", 200_000, price_in=0.6, price_cached=0.11, price_out=2.2),
    ModelInfo("glm-4.6v", "GLM-4.6V", 128_000, vision=True, price_in=0.3, price_cached=0.05, price_out=0.9),
    ModelInfo("glm-4.5-air", "GLM-4.5 Air", 128_000, price_in=0.2, price_cached=0.03, price_out=1.1),
    ModelInfo("glm-4.5", "GLM-4.5", 128_000, price_in=0.6, price_cached=0.11, price_out=2.2),
]
MODEL_MAP = {m.id: m for m in MODELS}

# The models you can actually pick, given which kind of key is configured.
if USING_CODING_PLAN:
    AVAILABLE_MODELS = [m for m in MODELS if m.on_coding_plan]
    DEFAULT_MODEL = "glm-5.3-flash"
else:
    AVAILABLE_MODELS = MODELS
    DEFAULT_MODEL = "glm-4.7-flash"


def is_available(model_id: str | None) -> bool:
    return any(m.id == model_id for m in AVAILABLE_MODELS)


def usable_model(model_id: str | None) -> str:
    """The model a chat really runs on: an old chat whose model isn't on this key falls back to the default."""
    return model_id if is_available(model_id) else DEFAULT_MODEL


def uses_effort(model_id: str | None) -> bool:
    """True for models that always think and take an effort level instead of an on/off switch."""
    return model_info(model_id).thinking == "always"


def model_info(model_id: str | None) -> ModelInfo:
    if model_id and model_id in MODEL_MAP:
        return MODEL_MAP[model_id]
    return ModelInfo(model_id or "unknown", model_id or "unknown", 128_000)
