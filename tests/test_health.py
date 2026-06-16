"""Tests for the run heartbeat / freshness signal in src/health.py."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import health


def _chat() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "minute": [0, 1],
            "message": ["we are done", "golazo"],
            "source": ["bluesky", "mastodon"],
        }
    )


def test_stream_health_live_match() -> None:
    commentary = pd.Series(["1' shot on target"], dtype="str")
    status = health.stream_health(_chat(), commentary, "ESPN-760415")
    assert status["source"] == "live"
    assert status["live"] is True
    assert status["fetch_ok"] is True
    assert status["n_reactions"] == 2
    assert status["reactions_by_source"] == {"bluesky": 1, "mastodon": 1}


def test_stream_health_simulator_is_not_live() -> None:
    status = health.stream_health(_chat(), pd.Series(dtype="str"), "SIM-20260613")
    assert status["source"] == "simulator"
    assert status["live"] is False
    assert status["fetch_ok"] is False


def test_write_and_load_status_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run_status.json"
    status = health.stream_health(_chat(), pd.Series(["1' x"], dtype="str"), "ESPN-1")
    health.write_status(status, path)
    assert health.load_status(path) == status


def test_load_status_missing_file(tmp_path: Path) -> None:
    assert health.load_status(tmp_path / "absent.json") is None


def test_freshness_levels() -> None:
    now = datetime(2026, 6, 13, 20, 0, tzinfo=timezone.utc)

    def status(source: str, age_min: int, reactions: int = 50) -> dict[str, object]:
        return {
            "last_run_utc": (now - timedelta(minutes=age_min)).isoformat(),
            "source": source,
            "live": source == "live",
            "n_reactions": reactions,
            "reactions_by_source": {"bluesky": reactions},
        }

    assert health.freshness(status("live", 5), now=now)["level"] == "live"
    assert health.freshness(status("live", 40), now=now)["level"] == "stale"
    assert health.freshness(status("simulator", 1), now=now)["level"] == "degraded"
    assert health.freshness(status("live", 1, reactions=0), now=now)["level"] == "no-data"
    assert health.freshness(None, now=now)["level"] == "no-data"
