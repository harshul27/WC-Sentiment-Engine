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


def test_reddit_youtube_x_skip_without_credentials() -> None:
    assert sources.fetch_reddit(["Mexico"]).empty
    assert sources.fetch_youtube("Mexico vs South Africa live").empty
    assert sources.fetch_x(["Mexico", "South Africa"]).empty


def test_fetch_x_parses_grok_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    body = (
        'Here are the posts:\n[{"text": "we are choking again", '
        '"created_at": "2026-06-11T19:20:00Z"}, '
        '{"text": "what a save!!", "created_at": "2026-06-11T19:21:00Z"}]'
    )
    payload = {
        "output": [
            {"type": "x_search_call"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": body}],
            },
        ]
    }
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return FakeResponse(payload)

    monkeypatch.setattr(sources.requests, "post", fake_post)
    frame = sources.fetch_x(["Mexico", "South Africa"])
    assert len(frame) == 2
    assert list(frame.columns) == sources.REACTION_COLUMNS
    assert (frame["source"] == "x").all()
    assert "choking" in frame.loc[0, "message"]
    assert captured["url"] == sources.XAI_RESPONSES
    assert captured["json"]["tools"][0]["type"] == "x_search"


def test_fetch_x_handles_missing_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    payload = {"output_text": '[{"text": "no timestamp here"}]'}
    monkeypatch.setattr(
        sources.requests, "post", lambda *a, **k: FakeResponse(payload)
    )
    frame = sources.fetch_x(["Mexico"])
    assert len(frame) == 1
    assert frame.loc[0, "message"] == "no timestamp here"
    assert pd.notna(frame.loc[0, "created_utc"])


def test_fetch_x_survives_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    payload = {"output_text": "the model rambled without returning any array"}
    monkeypatch.setattr(
        sources.requests, "post", lambda *a, **k: FakeResponse(payload)
    )
    assert sources.fetch_x(["Mexico"]).empty


def test_fetchers_survive_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(sources.requests, "get", boom)
    monkeypatch.setattr(sources.requests, "post", boom)
    assert sources.fetch_bluesky(["Mexico"]).empty
    assert sources.fetch_mastodon(["worldcup"]).empty


def _reaction_frame(n: int, source: str, **over) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "created_utc": pd.date_range("2026-06-11T19:00:00Z", periods=n, freq="10s"),
            "message": over.get(
                "messages", [f"what a goal from Mexico, take {i}" for i in range(n)]
            ),
            "source": source,
            "author": over.get("author", ""),
        }
    )


def test_gather_reactions_caps_window_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = _reaction_frame(1200, "bluesky")
    duplicate = big.copy()
    monkeypatch.setattr(sources, "fetch_bluesky", lambda *a, **k: big)
    monkeypatch.setattr(sources, "fetch_mastodon", lambda *a, **k: duplicate)
    monkeypatch.setattr(sources, "fetch_reddit", lambda *a, **k: sources._empty())
    monkeypatch.setattr(sources, "fetch_youtube", lambda *a, **k: sources._empty())
    monkeypatch.setattr(sources, "fetch_x", lambda *a, **k: sources._empty())
    merged = sources.gather_reactions(["Mexico", "South Africa"])
    assert len(merged) == sources.COMMENT_WINDOW == 1000
    assert merged["created_utc"].is_monotonic_increasing
    assert merged["message"].is_unique


def test_gather_reactions_all_sources_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("fetch_bluesky", "fetch_mastodon", "fetch_reddit", "fetch_youtube", "fetch_x"):
        monkeypatch.setattr(sources, name, lambda *a, **k: sources._empty())
    assert sources.gather_reactions(["Mexico"]).empty


def test_clean_reactions_filters_jargon_and_bots() -> None:
    frame = pd.DataFrame(
        {
            "created_utc": pd.date_range("2026-06-11T19:00Z", periods=6, freq="5s"),
            "message": [
                "Mexico are choking, what a disaster",  # keep
                "⚽⚽⚽",  # drop: emoji-only / too short
                "https://t.co/x",  # drop: link-only
                "the weather is nice today and warm",  # drop: off-topic (no team/football)
                "great goal by South Africa there",  # keep (football keyword)
                "MATCH THREAD: Mexico vs South Africa",  # keep but from a bot below
            ],
            "source": ["bluesky"] * 6,
            "author": ["fan1", "fan2", "fan3", "fan4", "fan5", "AutoModerator"],
        }
    )
    cleaned = sources.clean_reactions(frame, ["Mexico", "South Africa"])
    msgs = cleaned["message"].tolist()
    assert "Mexico are choking, what a disaster" in msgs
    assert "great goal by South Africa there" in msgs
    assert "⚽⚽⚽" not in msgs
    assert "https://t.co/x" not in msgs
    assert "the weather is nice today and warm" not in msgs
    assert not cleaned["author"].str.lower().eq("automoderator").any()


def test_clean_reactions_caps_per_author_flood() -> None:
    frame = _reaction_frame(
        20, "youtube", messages=[f"Mexico goal moment {i}" for i in range(20)],
        author="spammer",
    )
    cleaned = sources.clean_reactions(frame, ["Mexico"])
    assert len(cleaned) == sources.MAX_PER_AUTHOR


def test_merge_window_accumulates_and_caps() -> None:
    prior = _reaction_frame(600, "bluesky")  # 19:00 .. 19:49:55 (5s spacing)
    new = _reaction_frame(
        600, "reddit", messages=[f"South Africa shot number {i}" for i in range(600)]
    )
    new["created_utc"] = pd.date_range(  # strictly later than every prior post
        "2026-06-11T21:00Z", periods=600, freq="5s"
    )
    merged = sources.merge_window(prior, new, window=1000)
    assert len(merged) == 1000
    assert merged["created_utc"].is_monotonic_increasing
    # every newest (reddit) reaction is retained; the oldest prior ones drop off
    merged_keys = set(merged["message"].map(sources._normalise))
    new_keys = set(new["message"].map(sources._normalise))
    assert new_keys <= merged_keys


def test_merge_window_dedupes_repeated_reactions() -> None:
    prior = _reaction_frame(10, "bluesky")
    merged = sources.merge_window(prior, prior.copy(), window=1000)
    assert len(merged) == 10
