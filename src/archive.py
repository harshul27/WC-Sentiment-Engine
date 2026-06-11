"""Durable match archive: validated, NOT NULL schema for completed matches.

Two DuckDB tables (mirrored to committed Parquet files so the archive
survives ephemeral GitHub Action runners and is readable from the cloud):

  match_archive  - one row per (match_id, minute): the full scored state
                   including emotion distribution, xG stability, and the
                   arbitrage outputs. PRIMARY KEY (match_id, minute), every
                   column NOT NULL with range checks - the training corpus
                   for the nightly threshold self-correction.
  match_results  - one row per match: fixture metadata and final score,
                   the ground-truth side for outcome evaluation.

Writes are idempotent upserts (delete + insert per match_id), so re-running
the pipeline against a finished match simply refreshes its rows.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from emotion import EMOTION_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "database.duckdb"
ARCHIVE_PARQUET = DATA_DIR / "match_archive.parquet"
RESULTS_PARQUET = DATA_DIR / "match_results.parquet"

ARCHIVE_DDL = """
CREATE TABLE IF NOT EXISTS match_archive (
    match_id             VARCHAR   NOT NULL,
    minute               INTEGER   NOT NULL CHECK (minute >= 0),
    crowd_panic_score    DOUBLE    NOT NULL CHECK (crowd_panic_score BETWEEN -1.0 AND 1.0),
    emo_panic            DOUBLE    NOT NULL CHECK (emo_panic BETWEEN 0.0 AND 1.0),
    emo_anger            DOUBLE    NOT NULL CHECK (emo_anger BETWEEN 0.0 AND 1.0),
    emo_joy              DOUBLE    NOT NULL CHECK (emo_joy BETWEEN 0.0 AND 1.0),
    emo_confidence       DOUBLE    NOT NULL CHECK (emo_confidence BETWEEN 0.0 AND 1.0),
    emo_despair          DOUBLE    NOT NULL CHECK (emo_despair BETWEEN 0.0 AND 1.0),
    emo_surprise         DOUBLE    NOT NULL CHECK (emo_surprise BETWEEN 0.0 AND 1.0),
    dominant_emotion     VARCHAR   NOT NULL,
    emotional_volatility DOUBLE    NOT NULL CHECK (emotional_volatility BETWEEN 0.0 AND 1.0),
    comment_volume       INTEGER   NOT NULL CHECK (comment_volume >= 0),
    rolling_xg           DOUBLE    NOT NULL CHECK (rolling_xg >= 0.0),
    delta_xg_10min       DOUBLE    NOT NULL CHECK (delta_xg_10min BETWEEN 0.0 AND 1.0),
    arbitrage_index      DOUBLE    NOT NULL CHECK (arbitrage_index BETWEEN 0.0 AND 1.0),
    flagged              BOOLEAN   NOT NULL,
    situation            VARCHAR   NOT NULL,
    situation_confidence DOUBLE    NOT NULL CHECK (situation_confidence BETWEEN 0.0 AND 1.0),
    archived_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (match_id, minute)
)
"""

RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS match_results (
    match_id    VARCHAR   NOT NULL PRIMARY KEY,
    home_team   VARCHAR   NOT NULL,
    away_team   VARCHAR   NOT NULL,
    kickoff_utc TIMESTAMP NOT NULL,
    final_score VARCHAR   NOT NULL,
    state       VARCHAR   NOT NULL,
    archived_at TIMESTAMP NOT NULL
)
"""

ARCHIVE_COLUMNS = [
    "match_id",
    "minute",
    "crowd_panic_score",
    *EMOTION_COLUMNS,
    "dominant_emotion",
    "emotional_volatility",
    "comment_volume",
    "rolling_xg",
    "delta_xg_10min",
    "arbitrage_index",
    "flagged",
    "situation",
    "situation_confidence",
    "archived_at",
]


def validate_state(state: pd.DataFrame, match_id: str) -> pd.DataFrame:
    """Coerce a scored state frame into the strict archive schema.

    Guarantees: no nulls anywhere, all numeric fields clipped into their
    valid ranges, integer minutes >= 0, dominant_emotion defaulting to
    'neutral'. Rows without a usable minute are dropped.
    """
    frame = state.copy()
    frame["match_id"] = str(match_id)
    if "minute" not in frame.columns:
        frame["minute"] = np.nan
    frame["minute"] = pd.to_numeric(frame["minute"], errors="coerce")
    frame = frame.dropna(subset=["minute"])
    if frame.empty:
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    frame["minute"] = frame["minute"].astype("int64").clip(lower=0)
    frame = frame.drop_duplicates(subset="minute", keep="last")

    def numeric(column: str) -> pd.Series:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    bounded = {
        "crowd_panic_score": (-1.0, 1.0),
        "emotional_volatility": (0.0, 1.0),
        "delta_xg_10min": (0.0, 1.0),
        "arbitrage_index": (0.0, 1.0),
        **{col: (0.0, 1.0) for col in EMOTION_COLUMNS},
    }
    for column, (low, high) in bounded.items():
        frame[column] = (
            numeric(column).ffill().fillna(0.0).clip(low, high).astype("float64")
        )
    frame["rolling_xg"] = (
        numeric("rolling_xg").ffill().fillna(0.0).clip(lower=0.0).astype("float64")
    )
    frame["comment_volume"] = (
        numeric("comment_volume").ffill().fillna(0).clip(lower=0).astype("int64")
    )
    frame["dominant_emotion"] = (
        frame.get("dominant_emotion", pd.Series(index=frame.index, dtype="object"))
        .ffill()
        .fillna("neutral")
        .astype(str)
    )
    frame["situation"] = (
        frame.get("situation", pd.Series(index=frame.index, dtype="object"))
        .ffill()
        .fillna("unknown")
        .astype(str)
    )
    frame["situation_confidence"] = (
        numeric("situation_confidence").ffill().fillna(0.0).clip(0.0, 1.0).astype("float64")
    )
    flagged = frame.get("flagged")
    frame["flagged"] = (
        flagged.fillna(False).astype(bool) if flagged is not None else False
    )
    frame["archived_at"] = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    result = frame[ARCHIVE_COLUMNS].sort_values("minute").reset_index(drop=True)
    if result.isna().any().any():
        raise ValueError("archive validation failed: null values remain")
    return result


def archive_match(
    state: pd.DataFrame,
    match_meta: dict[str, object],
    db_path: Path = DB_PATH,
    archive_path: Path = ARCHIVE_PARQUET,
    results_path: Path = RESULTS_PARQUET,
) -> int:
    """Upsert one match into the archive tables and mirror to Parquet.

    match_meta requires: match_id, home_team, away_team, kickoff_utc,
    final_score, state. Returns the number of minute-rows archived.
    """
    match_id = str(match_meta["match_id"])
    rows = validate_state(state, match_id)
    if rows.empty:
        return 0
    kickoff = pd.Timestamp(match_meta["kickoff_utc"])
    if kickoff.tzinfo is not None:
        kickoff = kickoff.tz_convert("UTC").tz_localize(None)
    result_row = pd.DataFrame(
        [
            {
                "match_id": match_id,
                "home_team": str(match_meta.get("home_team") or "unknown"),
                "away_team": str(match_meta.get("away_team") or "unknown"),
                "kickoff_utc": kickoff,
                "final_score": str(match_meta.get("final_score") or "0-0"),
                "state": str(match_meta.get("state") or "post"),
                "archived_at": pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None),
            }
        ]
    )
    # Seed from the committed Parquet mirrors first: runners are ephemeral,
    # so the DuckDB file starts empty and the Parquet is the durable store.
    prior_rows = load_archive(archive_path=archive_path)
    if not prior_rows.empty:
        prior_rows = prior_rows.loc[prior_rows["match_id"] != match_id]
        for column in ARCHIVE_COLUMNS:
            if column not in prior_rows.columns:
                prior_rows[column] = (
                    "unknown" if column == "situation" else 0.0
                )
        prior_rows = prior_rows[ARCHIVE_COLUMNS]
    all_rows = (
        pd.concat([prior_rows, rows], ignore_index=True) if not prior_rows.empty else rows
    )
    prior_results = load_results(results_path=results_path)
    if not prior_results.empty:
        prior_results = prior_results.loc[prior_results["match_id"] != match_id]
    all_results = (
        pd.concat([prior_results, result_row], ignore_index=True)
        if not prior_results.empty
        else result_row
    )
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute("DROP TABLE IF EXISTS match_archive")
        connection.execute("DROP TABLE IF EXISTS match_results")
        connection.execute(ARCHIVE_DDL)
        connection.execute(RESULTS_DDL)
        connection.register("archive_rows", all_rows)
        connection.register("result_row", all_results)
        connection.execute("INSERT INTO match_archive SELECT * FROM archive_rows")
        connection.execute("INSERT INTO match_results SELECT * FROM result_row")
        connection.execute(
            "COPY (SELECT * FROM match_archive ORDER BY match_id, minute) TO ? "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(archive_path)],
        )
        connection.execute(
            "COPY (SELECT * FROM match_results ORDER BY match_id) TO ? "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(results_path)],
        )
    finally:
        connection.close()
    return int(len(rows))


def load_archive(
    match_id: str | None = None, archive_path: Path = ARCHIVE_PARQUET
) -> pd.DataFrame:
    """Read archived minute-rows (optionally one match) from Parquet."""
    if not Path(archive_path).exists():
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    try:
        frame = pd.read_parquet(archive_path)
    except (OSError, ValueError):
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    if match_id is not None:
        frame = frame.loc[frame["match_id"] == str(match_id)]
    return frame.reset_index(drop=True)


def load_results(results_path: Path = RESULTS_PARQUET) -> pd.DataFrame:
    """Read the per-match results/metadata table from Parquet."""
    if not Path(results_path).exists():
        return pd.DataFrame(
            columns=[
                "match_id",
                "home_team",
                "away_team",
                "kickoff_utc",
                "final_score",
                "state",
                "archived_at",
            ]
        )
    try:
        return pd.read_parquet(results_path)
    except (OSError, ValueError):
        return pd.DataFrame()
