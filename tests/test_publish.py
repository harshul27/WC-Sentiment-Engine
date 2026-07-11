"""Tests for the Bluesky insight publisher in src/publish.py (offline, key-gated)."""
from __future__ import annotations

import pandas as pd
import pytest
import requests

import publish


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def _state() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["ESPN-760415"],
            "minute": [78],
            "dominant_emotion": ["panic"],
            "arbitrage_index": [0.71],
            "situation": ["panic_divergence"],
        }
    )


def test_draft_is_labelled_and_within_limit() -> None:
    draft = publish.draft_post(_state(), "Fans anxious while the match is calm.")
    assert publish.LABEL in draft
    assert len(draft) <= publish.MAX_CHARS
    assert "78'" in draft and "Panic" in draft


def test_draft_empty_state() -> None:
    draft = publish.draft_post(pd.DataFrame())
    assert publish.LABEL in draft


def test_post_disabled_without_credentials() -> None:
    assert publish.enabled() is False
    assert publish.post_insight("hello") is None  # no creds -> no network


def test_post_insight_creates_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_HANDLE", "me.bsky.social")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "app-pw")
    calls: list[str] = []

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        calls.append(url)
        if url.endswith("createSession"):
            return FakeResponse({"accessJwt": "jwt", "did": "did:plc:x"})
        body = kwargs.get("json", {})
        assert body["record"]["text"]  # posts exactly what was passed
        return FakeResponse({"uri": "at://did:plc:x/app.bsky.feed.post/1"})

    monkeypatch.setattr(publish.requests, "post", fake_post)
    uri = publish.post_insight("⚽ crowd mood insight\n🤖 automated analytics")
    assert uri == "at://did:plc:x/app.bsky.feed.post/1"
    assert any("createSession" in c for c in calls)
    assert any("createRecord" in c for c in calls)


def test_post_insight_survives_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_HANDLE", "me.bsky.social")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "app-pw")

    def boom(*a: object, **k: object) -> None:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(publish.requests, "post", boom)
    assert publish.post_insight("hello") is None


def test_recent_engagement_parses_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "feed": [
            {
                "post": {
                    "record": {"text": "insight A", "createdAt": "2026-07-11T10:00:00Z"},
                    "likeCount": 5,
                    "repostCount": 2,
                    "replyCount": 3,
                }
            }
        ]
    }
    monkeypatch.setattr(publish.requests, "get", lambda *a, **k: FakeResponse(payload))
    frame = publish.recent_engagement("me.bsky.social")
    assert len(frame) == 1
    assert frame.loc[0, "likes"] == 5 and frame.loc[0, "replies"] == 3


def test_recent_engagement_without_handle() -> None:
    assert publish.recent_engagement("").empty
