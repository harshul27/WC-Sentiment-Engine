"""Shared test setup: make src/ importable and isolate LLM env keys."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def no_external_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests deterministic and offline: no keyed external services."""
    for key in (
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "COMMENTARY_FEED_URL",
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "YOUTUBE_API_KEY",
        "XAI_API_KEY",
        "ENABLE_SOFASCORE",
    ):
        monkeypatch.delenv(key, raising=False)
