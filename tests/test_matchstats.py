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


def test_parse_boxscore_extracts_key_and_advanced_stats() -> None:
    stats = matchstats.parse_boxscore(BOXSCORE_FIXTURE)
    assert set(stats) == {"Argentina", "France"}
    assert stats["Argentina"]["possessionPct"] == "54"
    assert stats["Argentina"]["shotsOnTarget"] == "10"
    assert stats["Argentina"]["passPct"] == "0.8"  # advanced tier now carried


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


DETAIL_FIXTURE = {
    **BOXSCORE_FIXTURE,
    "rosters": [
        {
            "team": {"displayName": "Argentina"},
            "roster": [
                {
                    "athlete": {"displayName": "E. Martinez"},
                    "position": {"abbreviation": "G"},
                    "starter": True,
                    "stats": [
                        {"name": "saves", "value": 5.0},
                        {"name": "shotsFaced", "value": 7.0},
                        {"name": "goalsConceded", "value": 1.0},
                    ],
                },
                {
                    "athlete": {"displayName": "L. Messi"},
                    "position": {"abbreviation": "F"},
                    "starter": True,
                    "stats": [
                        {"name": "totalGoals", "value": 2.0},
                        {"name": "totalShots", "value": 5.0},
                        {"name": "shotsOnTarget", "value": 3.0},
                    ],
                },
                {  # unused sub: no stats block -> skipped
                    "athlete": {"displayName": "Bench Player"},
                    "position": {"abbreviation": "M"},
                    "starter": False,
                    "stats": [],
                },
            ],
        }
    ],
    "leaders": [
        {
            "team": {"displayName": "Argentina"},
            "leaders": [
                {
                    "name": "totalShots",
                    "leaders": [
                        {"athlete": {"displayName": "L. Messi"}, "displayValue": "5"}
                    ],
                },
                {
                    "name": "saves",
                    "leaders": [
                        {"athlete": {"displayName": "E. Martinez"}, "displayValue": "5"}
                    ],
                },
            ],
        }
    ],
    "keyEvents": [
        {
            "type": {"type": "goal"},
            "clock": {"value": 1380.0},  # ESPN clock is seconds -> 23'
            "team": {"displayName": "Argentina"},
            "text": "Goal! Lionel Messi scores",
        },
        {
            "type": {"type": "yellow-card"},
            "clock": {"value": 3033.0},
            "team": {"displayName": "France"},
            "text": "Booking",
        },
        {"type": {}, "clock": {"value": 0.0}},  # untyped -> dropped
    ],
}


def test_parse_player_stats_extracts_and_skips_unused() -> None:
    players = matchstats.parse_player_stats(DETAIL_FIXTURE)
    assert "Argentina" in players
    names = [p["name"] for p in players["Argentina"]]
    assert "L. Messi" in names and "E. Martinez" in names
    assert "Bench Player" not in names
    messi = next(p for p in players["Argentina"] if p["name"] == "L. Messi")
    assert messi["totalGoals"] == 2.0 and messi["position"] == "F"


def test_parse_leaders_lines() -> None:
    leaders = matchstats.parse_leaders(DETAIL_FIXTURE)
    assert leaders["Argentina"]["totalShots"] == "L. Messi (5)"
    lines = matchstats.top_performers(leaders)
    assert "L. Messi (5) shots" in lines["Argentina"]
    assert "E. Martinez (5) saves" in lines["Argentina"]


def test_parse_key_events_converts_seconds_to_minutes() -> None:
    events = matchstats.parse_key_events(DETAIL_FIXTURE)
    assert len(events) == 2  # untyped entry dropped
    goal = events[0]
    assert goal["minute"] == 23 and goal["type"] == "goal"
    goals = matchstats.goal_scorers(events)
    assert goals == ["23' Argentina — Goal! Lionel Messi scores"]


def test_keeper_pressure_picks_goalkeeper() -> None:
    players = matchstats.parse_player_stats(DETAIL_FIXTURE)
    keeper = matchstats.keeper_pressure(players)
    assert keeper["Argentina"]["keeper"] == "E. Martinez"
    assert keeper["Argentina"]["saves"] == 5.0
    assert keeper["Argentina"]["shots_faced"] == 7.0
    assert matchstats.keeper_pressure({}) == {}


def test_fetch_match_detail_empty_on_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(matchstats.requests, "get", boom)
    detail = matchstats.fetch_match_detail("760415")
    assert detail == {"stats": {}, "players": {}, "leaders": {}, "key_events": []}


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
