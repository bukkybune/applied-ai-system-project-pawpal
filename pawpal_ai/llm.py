"""Language model clients.

The advisor only needs one method, `generate(system, prompt, json_mode)`, so the
real model (Gemini, free tier) and the fakes used in tests are interchangeable.
With no API key there is no client at all and the advisor runs in offline mode.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Protocol

log = logging.getLogger("pawpal_ai.llm")

DEFAULT_MODEL = "gemini-3.8-flash"
REQUEST_TIMEOUT_MS = 45_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMError(Exception):
    """The model could not be reached or returned something unusable."""


class LLMClient(Protocol):
    name: str

    def generate(self, system: str, prompt: str, json_mode: bool = False) -> str: ...


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env reader so setup needs no extra package. Never overrides real env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_json(text: str):
    """Pull a JSON value out of a model reply, tolerating ```json fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    raise LLMError("model did not return valid JSON")


class GeminiClient:
    """Google Gemini through the official google-genai SDK (free tier works)."""

    def __init__(self, api_key: str, model: str | None = None):
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as error:  # pragma: no cover - depends on install
            raise LLMError(
                "google-genai is not installed; run: pip install -r requirements.txt"
            ) from error
        self._types = types
        self._api_error = errors.APIError
        # A busy free tier can stall; cap each call so the app falls back to
        # offline mode instead of freezing. One retry rides out brief 503s.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=2),
            ),
        )
        self.model = model or DEFAULT_MODEL
        self.name = f"Gemini ({self.model})"

    def generate(self, system: str, prompt: str, json_mode: bool = False) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,  # low, so the same question gives the same advice
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except self._api_error as error:
            # 429 = free-tier quota used up, 403 = bad key, 404 = model not available.
            log.warning("Gemini API error %s: %s", getattr(error, "code", "?"), error)
            raise LLMError(f"Gemini API error {getattr(error, 'code', '?')}: {error}") from error
        except Exception as error:
            # Network problems surface as transport errors; the advisor only
            # needs to know the call failed so it can fall back.
            log.warning("Gemini call failed: %s", error)
            raise LLMError(f"{type(error).__name__}: {error}") from error
        text = response.text
        if not text:
            raise LLMError("Gemini returned an empty response")
        return text


def make_client() -> LLMClient | None:
    """The live client if a key is configured, otherwise None (offline mode)."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.info("no GEMINI_API_KEY; running in offline mode")
        return None
    try:
        return GeminiClient(api_key, os.environ.get("PAWPAL_MODEL") or None)
    except LLMError as error:
        log.error("could not start Gemini client: %s", error)
        return None
