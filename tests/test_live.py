"""Offline tests for the live data connectors in src/live.py.

All parsing is exercised against canned fixtures mirroring the real ESPN
and Bluesky payload shapes; network failure paths are tested by patching
requests.get. No test here performs real network IO.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import requests

import live

SCOREBOARD_FIXTURE = {
    "events": [
        {
            "id": "760415",
            "name": "South Africa at Mexico",
            "shortName": "RSA @ MEX",
            "date": "2026-06-11T19:00Z",
            "competitions": [
                {
                    "status": {
                        "displayClock": "57'",
                        "type": {"state": "in"},
                    },
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "1",
                            "team": {"displayName": "Mexico"},
                        },
                        {
                            "homeAway": "away",
                            "score": "0",
                            "team": {"displayName": "South Africa"},
                        },
                    ],
                }
            ],
        },
        {
            "id": "760414",
            "name": "Czechia at South Korea",
            "shortName": "CZE @ KOR",
            "date": "2026-06-12T02:00Z",
            "competitions": [
                {
                    "status": {"displayClock": "0'", "type": {"state": "pre"}},
                    "competitors": [],
                }
            ],
        },
    ]
}

COMMENTARY_FIXTURE = {
    "commentary": [
        {"text": "Match ends, Mexico 1, South Africa 0.", "time": {"displayValue": "90'+5'"}},
        {"text": "Shot on target by Raul Jimenez, forces a save.", "time": {"displayValue": "57'"}},
        {"text": "Halftime notes and lineups.", "time": {"displayValue": "HT"}},
        {"text": "", "time": {"displayValue": "12'"}},
        {"text": "Corner conceded by South Africa.", "time": {"displayValue": "12'"}},
    ]
}


def test_parse_scoreboard_normalizes_fixture() -> None:
    board = live.parse_scoreboard(SCOREBOARD_FIXTURE)
    assert len(board) == 2
    first = board.iloc[0]
    assert first["event_id"] == "760415"
    assert first["state"] == "in"
    assert first["clock_minute"] == 57
    assert first["home_team"] == "Mexico"
    assert first["away_team"] == "South Africa"
    assert first["score"] == "1-0"
    assert first["kickoff_utc"] == datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)


def test_parse_scoreboard_handles_empty_payload() -> None:
    board = live.parse_scoreboard({})
    assert board.empty
    assert list(board.columns) == live.SCOREBOARD_COLUMNS


def test_parse_commentary_payload_formats_minute_lines() -> None:
    lines = live.parse_commentary_payload(COMMENTARY_FIXTURE)
    assert lines.tolist() == [
        "90' Match ends, Mexico 1, South Africa 0.",
        "57' Shot on target by Raul Jimenez, forces a save.",
        "12' Corner conceded by South Africa.",
    ]


def test_commentary_lines_feed_the_model_parser() -> None:
    from model import parse_commentary

    lines = live.parse_commentary_payload(COMMENTARY_FIXTURE)
    events = parse_commentary(lines)
    assert len(events) == 3
    assert "shot_on_target" in set(events["event_type"])
    assert "corner" in set(events["event_type"])


def test_posts_to_chat_maps_minutes_and_drops_stale_posts() -> None:
    kickoff = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)
    posts = pd.DataFrame(
        {
            "created_utc": [
                kickoff - timedelta(hours=2),
                kickoff - timedelta(minutes=10),
                kickoff + timedelta(minutes=57, seconds=30),
            ],
            "message": ["old preview post", "pre-match nerves", "we are choking again"],
        }
    )
    chat = live.posts_to_chat(posts, kickoff)
    assert chat["message"].tolist() == ["pre-match nerves", "we are choking again"]
    assert chat["minute"].tolist() == [0, 57]


def test_posts_to_chat_handles_mixed_precision_string_timestamps() -> None:
    kickoff = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)
    posts = pd.DataFrame(
        {
            "created_utc": [
                "2026-06-11T19:05:43.630Z",
                "2026-06-11T19:30:37.467994Z",
                "not a timestamp",
            ],
            "message": ["early nerves", "we are choking", "junk"],
        }
    )
    chat = live.posts_to_chat(posts, kickoff)
    assert chat["minute"].tolist() == [5, 30]
    assert chat["minute"].dtype == "int64"


def test_posts_to_chat_empty_inputs() -> None:
    chat = live.posts_to_chat(pd.DataFrame(columns=["created_utc", "message"]), live.utc_now())
    assert chat.empty
    assert list(chat.columns) == ["minute", "message"]


def test_current_live_match_picks_in_progress_fixture() -> None:
    board = live.parse_scoreboard(SCOREBOARD_FIXTURE)
    match = live.current_live_match(board)
    assert match is not None
    assert match["event_id"] == "760415"


def test_current_live_match_none_when_nothing_in_progress() -> None:
    board = live.parse_scoreboard(SCOREBOARD_FIXTURE)
    assert live.current_live_match(board.loc[board["state"] == "pre"]) is None
    assert live.current_live_match(live.parse_scoreboard({})) is None


def test_capture_phase_lifecycle() -> None:
    kickoff = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)
    during = kickoff + timedelta(minutes=60)
    just_finished = kickoff + timedelta(minutes=120)
    assert live.capture_phase("pre", kickoff, now=kickoff) == "pre"
    assert live.capture_phase("in", kickoff, now=during) == "live"
    assert (
        live.capture_phase("post", kickoff, now=just_finished, post_first_seen=just_finished)
        == "post-window"
    )
    assert (
        live.capture_phase(
            "post",
            kickoff,
            now=just_finished + timedelta(minutes=16),
            post_first_seen=just_finished,
        )
        == "frozen"
    )


def test_capture_phase_freezes_long_finished_matches() -> None:
    kickoff = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)
    hours_later = kickoff + timedelta(hours=5)
    assert live.capture_phase("post", kickoff, now=hours_later) == "frozen"


def test_current_capture_match_prefers_live_then_recent_post() -> None:
    board = live.parse_scoreboard(SCOREBOARD_FIXTURE)
    picked = live.current_capture_match(board)
    assert picked is not None
    assert picked["state"] == "in"
    finished = board.iloc[[0]].copy()
    finished["state"] = "post"
    kickoff = finished.iloc[0]["kickoff_utc"]
    inside_window = live.current_capture_match(
        finished, now=kickoff + timedelta(minutes=150)
    )
    assert inside_window is not None
    outside_window = live.current_capture_match(
        finished, now=kickoff + timedelta(minutes=200)
    )
    assert outside_window is None


def test_fetchers_return_empty_on_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(live.requests, "get", boom)
    assert live.fetch_scoreboard().empty
    assert live.fetch_match_commentary("760415").empty
    assert live.fetch_crowd_posts(["Mexico"]).empty
