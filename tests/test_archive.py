"""Tests for the validated match archive in src/archive.py."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import archive
from emotion import EMOTION_COLUMNS


def _raw_state(minutes: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "minute": range(minutes),
            "crowd_panic_score": np.linspace(-0.5, 1.5, minutes),
            "rolling_xg": np.linspace(0, 2, minutes),
            "delta_xg_10min": np.linspace(0, 0.9, minutes),
            "arbitrage_index": np.linspace(0, 0.8, minutes),
            "flagged": [False, True] * (minutes // 2) + [False] * (minutes % 2),
            "dominant_emotion": ["panic", None, "joy", None, "neutral"][:minutes],
            "emotional_volatility": [0.1, np.nan, 0.3, 0.2, 0.4][:minutes],
            "comment_volume": [3, 5, np.nan, 2, 1][:minutes],
        }
    )
    for column in EMOTION_COLUMNS:
        frame[column] = [0.2, np.nan, 0.4, 0.1, 0.0][:minutes]
    return frame


META = {
    "match_id": "ESPN-760415",
    "home_team": "Mexico",
    "away_team": "South Africa",
    "kickoff_utc": pd.Timestamp("2026-06-11T19:00Z"),
    "final_score": "2-1",
    "state": "post",
}


def test_validate_state_enforces_not_null_and_ranges() -> None:
    rows = archive.validate_state(_raw_state(), "ESPN-1")
    assert not rows.isna().any().any()
    assert float(rows["crowd_panic_score"].max()) <= 1.0
    assert float(rows["crowd_panic_score"].min()) >= -1.0
    assert (rows["dominant_emotion"] != "").all()
    assert rows["comment_volume"].dtype == "int64"
    assert list(rows.columns) == archive.ARCHIVE_COLUMNS


def test_validate_state_drops_bad_minutes_and_dedupes() -> None:
    state = _raw_state()
    state.loc[2, "minute"] = None
    state = pd.concat([state, state.iloc[[0]]], ignore_index=True)
    rows = archive.validate_state(state, "ESPN-1")
    assert rows["minute"].is_unique
    assert rows["minute"].notna().all()


def test_archive_match_writes_tables_and_parquet(tmp_path: Path) -> None:
    paths = {
        "db_path": tmp_path / "db.duckdb",
        "archive_path": tmp_path / "match_archive.parquet",
        "results_path": tmp_path / "match_results.parquet",
    }
    count = archive.archive_match(_raw_state(), META, **paths)
    assert count == 5
    assert paths["archive_path"].exists()
    assert paths["results_path"].exists()
    connection = duckdb.connect(str(paths["db_path"]), read_only=True)
    try:
        archived = connection.execute("SELECT COUNT(*) FROM match_archive").fetchone()[0]
        not_null_flags = connection.execute(
            "SELECT COUNT(*) FROM pragma_table_info('match_archive') WHERE \"notnull\""
        ).fetchone()[0]
        result = connection.execute(
            "SELECT home_team, final_score FROM match_results"
        ).fetchone()
    finally:
        connection.close()
    assert archived == 5
    assert not_null_flags == len(archive.ARCHIVE_COLUMNS)
    assert result == ("Mexico", "2-1")


def test_archive_match_upsert_is_idempotent(tmp_path: Path) -> None:
    paths = {
        "db_path": tmp_path / "db.duckdb",
        "archive_path": tmp_path / "match_archive.parquet",
        "results_path": tmp_path / "match_results.parquet",
    }
    archive.archive_match(_raw_state(), META, **paths)
    archive.archive_match(_raw_state(), META, **paths)
    rows = archive.load_archive("ESPN-760415", archive_path=paths["archive_path"])
    assert len(rows) == 5
    results = archive.load_results(results_path=paths["results_path"])
    assert len(results) == 1


def test_archive_survives_fresh_database(tmp_path: Path) -> None:
    """Ephemeral runners start with an empty DuckDB: the Parquet mirror
    must seed the tables so earlier matches are never clobbered."""
    paths = {
        "db_path": tmp_path / "db.duckdb",
        "archive_path": tmp_path / "match_archive.parquet",
        "results_path": tmp_path / "match_results.parquet",
    }
    archive.archive_match(_raw_state(), META, **paths)
    paths["db_path"].unlink()
    second_meta = {**META, "match_id": "ESPN-760414", "home_team": "South Korea"}
    archive.archive_match(_raw_state(), second_meta, **paths)
    rows = archive.load_archive(archive_path=paths["archive_path"])
    assert set(rows["match_id"]) == {"ESPN-760415", "ESPN-760414"}
    results = archive.load_results(results_path=paths["results_path"])
    assert len(results) == 2


def test_archive_migrates_old_schema_parquet(tmp_path: Path) -> None:
    """Pre-classifier archives lack the situation columns; upserting a new
    match must keep the old rows, defaulting them to 'unknown'."""
    paths = {
        "db_path": tmp_path / "db.duckdb",
        "archive_path": tmp_path / "match_archive.parquet",
        "results_path": tmp_path / "match_results.parquet",
    }
    old_rows = archive.validate_state(_raw_state(), "ESPN-OLD").drop(
        columns=["situation", "situation_confidence"]
    )
    old_rows.to_parquet(paths["archive_path"])
    archive.archive_match(_raw_state(), META, **paths)
    rows = archive.load_archive(archive_path=paths["archive_path"])
    assert set(rows["match_id"]) == {"ESPN-OLD", "ESPN-760415"}
    migrated = rows.loc[rows["match_id"] == "ESPN-OLD"]
    assert (migrated["situation"] == "unknown").all()


def test_load_archive_missing_file() -> None:
    rows = archive.load_archive(archive_path=Path("does/not/exist.parquet"))
    assert rows.empty
    assert list(rows.columns) == archive.ARCHIVE_COLUMNS


def test_archive_match_empty_state(tmp_path: Path) -> None:
    count = archive.archive_match(
        pd.DataFrame(columns=["minute"]),
        META,
        db_path=tmp_path / "db.duckdb",
        archive_path=tmp_path / "a.parquet",
        results_path=tmp_path / "r.parquet",
    )
    assert count == 0


def test_validate_state_rejects_unfixable_nulls() -> None:
    state = _raw_state()
    state["minute"] = None
    rows = archive.validate_state(state, "ESPN-1")
    assert rows.empty
