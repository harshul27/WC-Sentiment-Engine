"""Tests for the Supabase mirror in src/warehouse.py (offline, key-gated)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import requests

import warehouse


class FakeResponse:
    def __init__(self, status: int = 201) -> None:
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError("boom")


def test_disabled_without_secrets() -> None:
    assert warehouse.enabled() is False
    # push is a no-op (returns 0) and makes no request when disabled
    assert warehouse.push_records("match_archive", [{"a": 1}]) == 0


def test_records_are_json_safe() -> None:
    frame = pd.DataFrame(
        {
            "match_id": ["ESPN-1"],
            "minute": [np.int64(7)],
            "value": [np.float64(0.5)],
            "ts": [pd.Timestamp("2026-06-11T19:00:00Z")],
            "missing": [np.nan],
        }
    )
    rows = warehouse.records(frame)
    assert rows[0]["minute"] == 7 and isinstance(rows[0]["minute"], int)
    assert rows[0]["value"] == 0.5
    assert rows[0]["ts"].startswith("2026-06-11T19:00:00")
    assert rows[0]["missing"] is None


def test_push_records_upserts_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "service-key")
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["data"] = kwargs.get("data")
        return FakeResponse(201)

    monkeypatch.setattr(warehouse.requests, "post", fake_post)
    sent = warehouse.push_records("match_archive", [{"match_id": "ESPN-1", "minute": 1}])
    assert sent == 1
    assert captured["url"] == "https://proj.supabase.co/rest/v1/match_archive"
    assert captured["headers"]["Prefer"].startswith("resolution=merge-duplicates")
    assert "ESPN-1" in str(captured["data"])


def test_push_records_survives_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "service-key")
    monkeypatch.setattr(
        warehouse.requests, "post", lambda *a, **k: FakeResponse(500)
    )
    assert warehouse.push_records("match_archive", [{"match_id": "x", "minute": 1}]) == 0


def test_push_reactions_keys_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "service-key")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        warehouse.requests,
        "post",
        lambda url, **k: captured.update(url=url, data=k.get("data")) or FakeResponse(201),
    )
    chat = pd.DataFrame(
        {
            "minute": [10, 10, 11],
            "message": ["we are choking", "we are choking", "what a goal"],
            "source": ["bluesky", "mastodon", "reddit"],
            "team": ["home", "home", "away"],
            "author": ["a", "b", "c"],
        }
    )
    sent = warehouse.push_reactions(chat, "ESPN-1")
    assert sent == 2  # duplicate message collapsed by (match_id, message_hash)
    assert captured["url"].endswith("/rest/v1/reactions")
    payload = json.loads(captured["data"])
    assert {r["message_hash"] for r in payload}  # hashes present
    assert all(r["match_id"] == "ESPN-1" for r in payload)


def test_push_reactions_disabled_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    assert warehouse.push_reactions(pd.DataFrame({"message": ["x"]}), "ESPN-1") == 0  # no keys
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    assert warehouse.push_reactions(pd.DataFrame(), "ESPN-1") == 0  # empty


def test_push_status_drops_nested_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "service-key")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        warehouse.requests,
        "post",
        lambda url, **k: captured.update(data=k.get("data")) or FakeResponse(201),
    )
    status = {"match_id": "ESPN-1", "live": True, "reactions_by_source": {"x": 3}}
    assert warehouse.push_status(status) == 1
    assert "reactions_by_source" not in str(captured["data"])
