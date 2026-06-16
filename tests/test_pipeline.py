"""Integration tests for the DuckDB ingestion pipeline in src/pipeline.py."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import pipeline
from model import parse_commentary


def test_simulate_streams_is_deterministic() -> None:
    chat_a, commentary_a = pipeline.simulate_streams(seed=42)
    chat_b, commentary_b = pipeline.simulate_streams(seed=42)
    pd.testing.assert_frame_equal(chat_a, chat_b)
    pd.testing.assert_series_equal(commentary_a, commentary_b)


def test_simulate_streams_schema_and_coverage() -> None:
    chat, commentary = pipeline.simulate_streams(seed=1, minutes=45)
    assert list(chat.columns) == ["minute", "message", "source"]
    assert (chat["source"] == "simulator").all()
    assert int(chat["minute"].max()) == 45
    assert len(commentary) == 46
    assert commentary.str.match(r"^\d+' ").all()


def test_simulate_streams_contains_panic_decoupling_window() -> None:
    chat, _ = pipeline.simulate_streams(seed=7)
    panic_start = int(90 * 0.6)
    window = chat.loc[chat["minute"].between(panic_start, panic_start + 11), "message"]
    panic_hits = window.isin(pipeline.PANIC_CHAT).mean()
    assert panic_hits > 0.5


def test_fetch_live_commentary_returns_empty_on_failure() -> None:
    lines = pipeline.fetch_live_commentary("http://127.0.0.1:9", timeout=2.0)
    assert lines.empty


def test_persist_to_duckdb_writes_tables_and_parquet(tmp_path: Path) -> None:
    chat, commentary = pipeline.simulate_streams(seed=3, minutes=20)
    events = parse_commentary(commentary)
    state = pd.DataFrame(
        {
            "minute": [0, 1],
            "crowd_panic_score": [0.1, 0.9],
            "rolling_xg": [0.0, 0.3],
            "delta_xg_10min": [0.0, 0.29],
            "arbitrage_index": [0.1, 0.64],
            "flagged": [False, True],
        }
    )
    db_path = tmp_path / "test.duckdb"
    state_path = tmp_path / "state.parquet"
    pipeline.persist_to_duckdb(chat, commentary, events, state, db_path, state_path)
    assert state_path.exists()
    round_trip = pd.read_parquet(state_path)
    assert len(round_trip) == 2
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    finally:
        connection.close()
    assert {"raw_chat", "raw_commentary", "match_events", "arbitrage_state"} <= tables


def test_derive_overreaction_truth_logic() -> None:
    state = pd.DataFrame(
        {"minute": [10, 50], "crowd_panic_score": [0.9, 0.9]}
    )
    events = pd.DataFrame(
        {"minute": [15], "team": ["Brazil"], "event_type": ["goal"], "xg_value": [0.4]}
    )
    truth = pipeline.derive_overreaction_truth(state, events, horizon=15)
    assert truth.tolist() == [0.0, 1.0]


def test_gather_streams_prefers_live_match(monkeypatch: pytest.MonkeyPatch) -> None:
    match = pd.Series(
        {
            "event_id": "760415",
            "home_team": "Mexico",
            "away_team": "South Africa",
            "kickoff_utc": pd.Timestamp("2026-06-11T19:00Z"),
        }
    )
    live_chat = pd.DataFrame({"minute": [1], "message": ["we are choking"]})
    live_commentary = pd.Series(["1' shot on target"], dtype="str", name="line")
    monkeypatch.setattr(
        pipeline.live, "current_capture_match", lambda scoreboard=None, now=None: match
    )
    monkeypatch.setattr(pipeline.live, "live_streams", lambda m: (live_chat, live_commentary))
    chat, commentary, match_id, match_row = pipeline.gather_streams()
    assert match_id == "ESPN-760415"
    assert commentary.tolist() == ["1' shot on target"]
    assert match_row is not None


def test_gather_streams_falls_back_when_live_commentary_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    match = pd.Series({"event_id": "1", "home_team": "A", "away_team": "B"})
    empty = pd.Series(dtype="str", name="line")
    monkeypatch.setattr(
        pipeline.live, "current_capture_match", lambda scoreboard=None, now=None: match
    )
    monkeypatch.setattr(
        pipeline.live, "live_streams", lambda m: (pd.DataFrame(), empty)
    )
    _, commentary, match_id, match_row = pipeline.gather_streams()
    assert match_id.startswith("SIM-")
    assert not commentary.empty
    assert match_row is None


def _patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_path = tmp_path / "model_config.json"
    monkeypatch.setattr(pipeline, "DB_PATH", tmp_path / "database.duckdb")
    monkeypatch.setattr(pipeline, "STATE_PATH", tmp_path / "state.parquet")
    monkeypatch.setattr(pipeline, "STATUS_PATH", tmp_path / "run_status.json")
    monkeypatch.setattr(pipeline, "CONFIG_PATH", config_path)
    return config_path


def test_run_ingest_simulator_writes_state_and_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipeline.live, "current_capture_match", lambda scoreboard=None, now=None: None
    )

    state = pipeline.run_ingest()
    assert (tmp_path / "state.parquet").exists()
    assert (tmp_path / "run_status.json").exists()
    assert len(state) == 91
    assert state["arbitrage_index"].between(0.0, 1.0).all()
    assert "dominant_emotion" in state.columns
    assert "emo_panic" in state.columns
    assert not state["emo_panic"].isna().any()
    assert "situation" in state.columns
    assert state["situation"].isin(list(__import__("situation").PROTOTYPES)).all()
    assert state["situation_confidence"].between(0, 1).all()
    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status["source"] == "simulator"
    assert status["live"] is False


def test_run_optimize_skips_without_real_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulator must never tune the live threshold: with no archived
    real matches, optimize is a no-op that leaves the config untouched."""
    config_path = _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipeline.live, "current_capture_match", lambda scoreboard=None, now=None: None
    )
    pipeline.run_ingest()  # simulator state only, no archive written

    result = pipeline.run_optimize()
    assert result["status"] == "skipped"
    assert result["arbitrage_flag_threshold"] == 0.65  # default, untouched
    # A true no-op: skipping never rewrites the config.
    assert not config_path.exists()


def test_run_optimize_trains_on_archive_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With enough archived real-match minutes spanning both outcome classes,
    optimize fits the threshold and records the run."""
    config_path = _patch_paths(tmp_path, monkeypatch)
    archive_path = tmp_path / "match_archive.parquet"

    def _match_state(panic_block: range) -> pd.DataFrame:
        minutes = 100
        panic = np.array(
            [0.8 if m in panic_block else 0.0 for m in range(minutes)], dtype=float
        )
        return pd.DataFrame(
            {
                "minute": range(minutes),
                "crowd_panic_score": panic,
                "rolling_xg": np.full(minutes, 0.05),  # flat: no threat arrives
                "delta_xg_10min": np.full(minutes, 0.05),
                "arbitrage_index": np.abs(panic) * 0.95,
                "flagged": panic > 0.4,
                "dominant_emotion": ["panic" if p > 0.4 else "neutral" for p in panic],
                "emotional_volatility": np.full(minutes, 0.2),
                "comment_volume": np.full(minutes, 4),
                **{col: np.full(minutes, 0.1) for col in __import__("archive").EMOTION_COLUMNS},
            }
        )

    archive_mod = __import__("archive")
    base_meta = {
        "home_team": "A",
        "away_team": "B",
        "kickoff_utc": pd.Timestamp("2026-06-11T19:00Z"),
        "final_score": "1-0",
        "state": "post",
    }
    for mid, block in (("ESPN-1", range(40, 60)), ("ESPN-2", range(50, 70))):
        archive_mod.archive_match(
            _match_state(block),
            {**base_meta, "match_id": mid},
            db_path=tmp_path / "database.duckdb",
            archive_path=archive_path,
            results_path=tmp_path / "match_results.parquet",
        )

    result = pipeline.run_optimize(min_minutes=180)
    assert result.get("status") != "skipped"
    assert 0.05 <= result["arbitrage_flag_threshold"] <= 0.95
    assert np.isfinite(result["log_loss"])
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert len(saved["log_loss_history"]) == 1
    assert saved["log_loss_history"][0]["evaluated_minutes"] == 200
    assert saved["log_loss_history"][0]["matches"] == 2


def test_gather_streams_live_only_returns_none_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline.live, "current_capture_match", lambda scoreboard=None, now=None: None
    )
    chat, commentary, match_id, match_row = pipeline.gather_streams(allow_simulator=False)
    assert match_id.startswith("NONE-")
    assert commentary.empty
    assert chat.empty
    assert match_row is None


def test_run_ingest_live_only_skips_when_no_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipeline.live, "current_capture_match", lambda scoreboard=None, now=None: None
    )
    state = pipeline.run_ingest(allow_simulator=False)
    assert state.empty
    assert not (tmp_path / "state.parquet").exists()
    assert not (tmp_path / "run_status.json").exists()
