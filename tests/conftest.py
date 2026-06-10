"""Shared test setup: make src/ importable and isolate LLM env keys."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def no_llm_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests deterministic and offline: never call live LLM endpoints."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("COMMENTARY_FEED_URL", raising=False)
