"""Offline tests for the advanced match metrics layer in src/matchstats.py."""
from __future__ import annotations

import pytest
import requests

import matchstats

BOXSCORE_FIXTURE = {
    "boxscore": {
        "teams": [
            {
                "team": {"displayName": "Argentina"},
                "statistics": [
                    {"name": "possessionPct", "displayValue": "54"},
                    {"name": "totalShots", "displayValue": "20"},
                    {"name": "shotsOnTarget", "displayValue": "10"},
                    {"name": "wonCorners", "displayValue": "6"},
                    {"name": "saves", "displayValue": "2"},
                    {"name": "passPct", "displayValue": "0.8"},
                ],
            },
            {
                "team": {"displayName": "France"},
                "statistics": [
                    {"name": "possessionPct", "displayValue": "46"},
                    {"name": "totalShots", "displayValue": "10"},
                    {"name": "shotsOnTarget", "displayValue": "5"},
                    {"name": "wonCorners", "displayValue": "5"},
                    {"name": "saves", "displayValue": "7"},
                ],
            },
        ]
    }
}


def test_parse_boxscore_extracts_key_stats_only() -> None:
    stats = matchstats.parse_boxscore(BOXSCORE_FIXTURE)
    assert set(stats) == {"Argentina", "France"}
    assert stats["Argentina"]["possessionPct"] == "54"
    assert stats["Argentina"]["shotsOnTarget"] == "10"
    assert "passPct" not in stats["Argentina"]


def test_parse_boxscore_empty_payload() -> None:
    assert matchstats.parse_boxscore({}) == {}


def test_control_index_blends_shares() -> None:
    stats = matchstats.parse_boxscore(BOXSCORE_FIXTURE)
    control = matchstats.control_index(stats)
    assert control is not None
    assert 0.5 < control < 0.75
    expected = (0.45 * 0.54 + 0.35 * (10 / 15) + 0.20 * (20 / 30)) / 1.0
    assert control == pytest.approx(expected, rel=1e-6)


def test_control_index_handles_missing_stats() -> None:
    assert matchstats.control_index({}) is None
    partial = {"A": {"saves": "1"}, "B": {"saves": "2"}}
    assert matchstats.control_index(partial) is None


def test_fetch_boxscore_empty_on_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(matchstats.requests, "get", boom)
    assert matchstats.fetch_boxscore("760415") == {}


def test_sofascore_momentum_disabled_without_flag() -> None:
    frame = matchstats.fetch_sofascore_momentum("Mexico", "South Africa")
    assert frame.empty
    assert list(frame.columns) == ["minute", "momentum"]


def test_match_context_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        matchstats,
        "fetch_boxscore",
        lambda event_id: matchstats.parse_boxscore(BOXSCORE_FIXTURE),
    )
    context = matchstats.match_context("633850", "Argentina", "France")
    assert context["control_index"] is not None
    assert "Argentina" in context["stats"]
    assert context["momentum"].empty
