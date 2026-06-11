"""Offline tests for the multi-source reaction aggregator in src/sources.py."""
from __future__ import annotations

import pandas as pd
import pytest
import requests

import sources


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_fetch_bluesky_parses_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "posts": [
            {
                "uri": "at://1",
                "record": {"createdAt": "2026-06-11T19:05:00Z", "text": "we are choking"},
            },
            {
                "uri": "at://1",
                "record": {"createdAt": "2026-06-11T19:05:00Z", "text": "we are choking"},
            },
        ]
    }
    monkeypatch.setattr(
        sources.requests, "get", lambda *a, **k: FakeResponse(payload)
    )
    frame = sources.fetch_bluesky(["Mexico"])
    assert len(frame) == 1
    assert list(frame.columns) == sources.REACTION_COLUMNS
    assert frame.loc[0, "source"] == "bluesky"


def test_fetch_mastodon_strips_html(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "id": "1",
            "created_at": "2026-06-11T19:10:00.000Z",
            "content": "<p>This is <b>panic</b> stations!</p>",
        }
    ]
    monkeypatch.setattr(
        sources.requests, "get", lambda *a, **k: FakeResponse(payload)
    )
    frame = sources.fetch_mastodon(["worldcup"])
    assert len(frame) == 1
    assert "<" not in frame.loc[0, "message"]
    assert "panic" in frame.loc[0, "message"]
    assert frame.loc[0, "source"] == "mastodon"


def test_reddit_and_youtube_skip_without_credentials() -> None:
    assert sources.fetch_reddit(["Mexico"]).empty
    assert sources.fetch_youtube("Mexico vs South Africa live").empty


def test_fetchers_survive_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(sources.requests, "get", boom)
    monkeypatch.setattr(sources.requests, "post", boom)
    assert sources.fetch_bluesky(["Mexico"]).empty
    assert sources.fetch_mastodon(["worldcup"]).empty


def test_gather_reactions_caps_window_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_source(n: int, source: str):
        return pd.DataFrame(
            {
                "created_utc": pd.date_range(
                    "2026-06-11T19:00:00Z", periods=n, freq="10s"
                ),
                "message": [f"{source} comment {i}" for i in range(n)],
                "source": source,
            }
        )

    big = fake_source(250, "bluesky")
    duplicate = big.copy()
    monkeypatch.setattr(sources, "fetch_bluesky", lambda terms: big)
    monkeypatch.setattr(sources, "fetch_mastodon", lambda tags: duplicate)
    monkeypatch.setattr(sources, "fetch_reddit", lambda terms: sources._empty())
    monkeypatch.setattr(sources, "fetch_youtube", lambda query: sources._empty())
    merged = sources.gather_reactions(["Mexico", "South Africa"])
    assert len(merged) == sources.COMMENT_WINDOW
    assert merged["created_utc"].is_monotonic_increasing
    assert merged["message"].is_unique


def test_gather_reactions_all_sources_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("fetch_bluesky", "fetch_mastodon", "fetch_reddit", "fetch_youtube"):
        monkeypatch.setattr(sources, name, lambda *a, **k: sources._empty())
    assert sources.gather_reactions(["Mexico"]).empty
