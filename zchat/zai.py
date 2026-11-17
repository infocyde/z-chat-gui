"""Thin async client for the Z.ai API (OpenAI-compatible chat, plus image + OCR endpoints)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from . import config

FRIENDLY = {
    "1113": "Your Z.ai key has no balance or plan that covers this. Image generation and OCR need "
            "pay-as-you-go credit (z.ai → Billing); the GLM Coding Plan only covers chat models.",
    "1210": "The model rejected a parameter.",
    "1261": "The prompt is too long for this model's context window.",
    "1301": "The content was blocked by Z.ai's safety filter.",
}


class ZaiError(Exception):
    def __init__(self, message: str, status: int = 0, code: str = ""):
        self.status, self.code = status, code
        friendly = FRIENDLY.get(code)
        super().__init__(f"{friendly} ({message})" if friendly and friendly != message else message)


@dataclass
class StreamEvent:
    type: str  # "content" | "reasoning" | "usage" | "finish"
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _headers() -> dict[str, str]:
    if not config.ZAI_API_KEY:
        raise ZaiError(f"ZAI_API_KEY is not set (looked in {config.ENV_FILE}).")
    return {"Authorization": f"Bearer {config.ZAI_API_KEY}", "Accept-Language": "en-US,en"}


def _raise_for(status: int, body: bytes) -> None:
    """Turn an error response into a ZaiError. Z.ai errors look like {"error": {"code", "message"}}."""
    text = body.decode("utf-8", "replace")
    try:
        err = json.loads(text)["error"]
        message = err.get("message") or text[:300]
        code = str(err.get("code", ""))
    except (ValueError, KeyError, TypeError, AttributeError):
        message = f"HTTP {status}: {text[:300]}"
        code = ""
    raise ZaiError(message, status, code)


def thinking_params(model_id: str, thinking: bool, effort: str = config.DEFAULT_EFFORT) -> dict[str, Any]:
    """Models that always think take an effort level; the rest take an on/off switch."""
    if config.uses_effort(model_id):
        if effort not in config.EFFORT_LEVELS:
            effort = config.DEFAULT_EFFORT
        return {"reasoning_effort": effort}
    return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def chat_body(messages: list[dict], model: str, thinking: bool, temperature: float | None,
              stream: bool = True, max_tokens: int | None = None,
              effort: str = config.DEFAULT_EFFORT) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream,
                            **thinking_params(model, thinking, effort)}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens:
        body["max_tokens"] = max_tokens
    return body


async def stream_chat(messages: list[dict], model: str, thinking: bool = False,
                      temperature: float | None = None,
                      effort: str = config.DEFAULT_EFFORT) -> AsyncIterator[StreamEvent]:
    """Yield content/reasoning deltas, then usage and finish events."""
    body = chat_body(messages, model, thinking, temperature, effort=effort)
    timeout = httpx.Timeout(connect=15, read=300, write=60, pool=15)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{config.ZAI_BASE_URL}/chat/completions",
                                 headers=_headers(), json=body) as r:
            if r.status_code != 200:
                _raise_for(r.status_code, await r.aread())
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if "error" in chunk:
                    err = chunk["error"]
                    raise ZaiError(err.get("message", str(err)), 200, str(err.get("code", "")))
                if chunk.get("usage"):
                    yield StreamEvent("usage", data=chunk["usage"])
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        yield StreamEvent("reasoning", delta["reasoning_content"])
                    if delta.get("content"):
                        yield StreamEvent("content", delta["content"])
                    if choice.get("finish_reason"):
                        yield StreamEvent("finish", choice["finish_reason"])


async def complete(messages: list[dict], model: str = config.UTILITY_MODEL, max_tokens: int = 300) -> str:
    """One-shot, non-streaming completion (titles, image prompts)."""
    if config.model_info(model).thinking == "always":
        # These models reason before answering and the reasoning counts toward max_tokens,
        # so leave room for it or the answer comes back empty.
        max_tokens += 2000
    body = chat_body(messages, model, thinking=False, temperature=None, stream=False, max_tokens=max_tokens)
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{config.ZAI_BASE_URL}/chat/completions", headers=_headers(), json=body)
    if r.status_code != 200:
        _raise_for(r.status_code, r.content)
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


async def generate_image(prompt: str, size: str = "1280x1280", quality: str = "hd") -> str:
    """Return the (temporary, 30-day) URL of a generated image."""
    body = {"model": config.IMAGE_MODEL, "prompt": prompt, "size": size, "quality": quality}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{config.ZAI_BASE_URL}/images/generations", headers=_headers(), json=body)
    if r.status_code != 200:
        _raise_for(r.status_code, r.content)
    data = r.json().get("data") or []
    if not data or not data[0].get("url"):
        raise ZaiError(f"No image returned: {r.text[:300]}")
    return data[0]["url"]


async def download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
    dest.write_bytes(r.content)
    return len(r.content)


async def ocr(data_url: str) -> str:
    """Parse a PDF/image (base64 data URL or http URL) into Markdown with glm-ocr."""
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{config.ZAI_BASE_URL}/layout_parsing", headers=_headers(),
                              json={"model": config.OCR_MODEL, "file": data_url})
    if r.status_code != 200:
        _raise_for(r.status_code, r.content)
    j = r.json()
    md = j.get("md_results")
    if isinstance(md, list):
        md = "\n\n".join(str(x) for x in md)
    return (md or "").strip()


# ---------------------------------------------------------------- Coding Plan quota


@dataclass
class PlanLimit:
    label: str            # "5-hour", "Weekly"
    used: int             # credits used in the current window
    total: int            # credits allowed per window
    percent: int
    resets_at: datetime   # local time


@dataclass
class PlanUsage:
    level: str                                                # "lite", "pro", ...
    limits: list[PlanLimit]
    week_tokens: dict[str, int] = field(default_factory=dict)  # model name -> tokens, last 7 days


def _limit_label(unit: int, number: int) -> str:
    """Z.ai describes each window as (unit, number). Seen so far: (3, 5) = 5 hours, (6, 1) = 1 week."""
    if unit == 3:
        return f"{number}-hour"
    if unit == 6:
        return "Weekly" if number == 1 else f"{number}-week"
    return "Limit"


async def plan_usage() -> PlanUsage:
    """Coding Plan quota windows plus per-model tokens for the last 7 days (same data as z.ai's usage page)."""
    now = datetime.now()
    week = {"startTime": f"{now - timedelta(days=7):%Y-%m-%d %H:%M:%S}", "endTime": f"{now:%Y-%m-%d %H:%M:%S}"}
    async with httpx.AsyncClient(timeout=20) as client:
        quota = await client.get(f"{config.ZAI_USAGE_URL}/quota/limit", headers=_headers())
        models = await client.get(f"{config.ZAI_USAGE_URL}/model-usage", headers=_headers(), params=week)
    if quota.status_code != 200:
        _raise_for(quota.status_code, quota.content)

    data = quota.json()["data"]
    limits = [PlanLimit(label=_limit_label(x["unit"], x["number"]),
                        used=x.get("currentValue", 0),
                        total=x.get("usage", 0),
                        percent=x.get("percentage", 0),
                        resets_at=datetime.fromtimestamp(x["nextResetTime"] / 1000))
              for x in data.get("limits", [])]

    week_tokens = {}
    if models.status_code == 200:  # nice to have; the limits above are what matter
        for m in models.json()["data"].get("modelSummaryList") or []:
            week_tokens[m["modelName"]] = m.get("totalTokens", 0)

    return PlanUsage(level=data.get("level", ""), limits=limits, week_tokens=week_tokens)
