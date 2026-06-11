"""Advanced stats enrichment via the soccerdata package (FBref).

Pulls World Cup schedule results and (once FBref publishes them mid-
tournament) team-level advanced stats - xG, possession - and condenses
them into a per-team priors table committed as data/team_priors.parquet.

FBref scraping is slow (politeness rate limits) and blocked from some
datacenter IPs, so the refresh runs only in the nightly flywheel as a
best-effort step; the dashboard and pipeline read the committed parquet
and never import soccerdata. The priors give the live readings context:
"is this team's current threat level normal for them or an anomaly?"

CLI: python src/advanced.py refresh
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIORS_PARQUET = ROOT / "data" / "team_priors.parquet"

PRIOR_COLUMNS = [
    "team",
    "matches_played",
    "goals_for_per_match",
    "goals_against_per_match",
    "xg_for_per_match",
    "possession_pct",
    "updated_at",
]


def build_priors(
    schedule: pd.DataFrame, team_stats: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Condense an FBref schedule (+optional team stats) into team priors.

    schedule needs columns home_team, away_team, score (e.g. "2-0" or
    "2–0", NA for unplayed). team_stats, when available, contributes xG
    per match and possession; otherwise those fields are NaN and the
    dashboard simply omits them.
    """
    records: dict[str, dict[str, float]] = {}
    played = schedule.dropna(subset=["score"]) if "score" in schedule.columns else schedule.iloc[0:0]
    for _, row in played.iterrows():
        raw = str(row["score"]).replace("–", "-").replace("—", "-")
        parts = raw.split("-")
        if len(parts) != 2:
            continue
        try:
            home_goals, away_goals = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            continue
        for team, scored, conceded in (
            (str(row["home_team"]).strip(), home_goals, away_goals),
            (str(row["away_team"]).strip(), away_goals, home_goals),
        ):
            entry = records.setdefault(team, {"played": 0, "gf": 0, "ga": 0})
            entry["played"] += 1
            entry["gf"] += scored
            entry["ga"] += conceded
    rows = [
        {
            "team": team,
            "matches_played": entry["played"],
            "goals_for_per_match": entry["gf"] / entry["played"],
            "goals_against_per_match": entry["ga"] / entry["played"],
            "xg_for_per_match": np.nan,
            "possession_pct": np.nan,
        }
        for team, entry in records.items()
        if entry["played"] > 0
    ]
    priors = pd.DataFrame(rows, columns=PRIOR_COLUMNS[:-1])
    if team_stats is not None and not team_stats.empty:
        stats = team_stats.reset_index()
        stats.columns = ["_".join(map(str, c)).strip("_") if isinstance(c, tuple) else str(c) for c in stats.columns]
        team_col = next((c for c in stats.columns if c.lower() == "team"), None)
        xg_col = next((c for c in stats.columns if "xg" in c.lower() and "per" in c.lower()), None)
        xg_col = xg_col or next((c for c in stats.columns if c.lower().endswith("xg")), None)
        poss_col = next((c for c in stats.columns if "poss" in c.lower()), None)
        if team_col is not None:
            extra = stats[[team_col]].copy()
            extra.columns = ["team"]
            played = pd.to_numeric(stats.get("MP", stats.get("Playing Time_MP")), errors="coerce")
            if xg_col is not None:
                xg = pd.to_numeric(stats[xg_col], errors="coerce")
                extra["xg_for_per_match"] = np.where(
                    (played > 0) & xg.notna(), xg / played.replace(0, np.nan), xg
                )
            if poss_col is not None:
                extra["possession_pct"] = pd.to_numeric(stats[poss_col], errors="coerce")
            if priors.empty:
                priors = extra.reindex(columns=PRIOR_COLUMNS[:-1])
            else:
                priors = priors.drop(
                    columns=[c for c in ("xg_for_per_match", "possession_pct") if c in priors]
                ).merge(extra, on="team", how="left")
    priors = priors.reindex(columns=PRIOR_COLUMNS[:-1])
    priors["updated_at"] = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    return priors.reset_index(drop=True)


def refresh_priors(parquet_path: Path = PRIORS_PARQUET) -> int:
    """Scrape FBref via soccerdata and write the priors parquet.

    Returns the number of teams written; 0 when scraping fails entirely
    (the previous parquet is left untouched in that case).
    """
    try:
        import soccerdata as sd

        fbref = sd.FBref(leagues="INT-World Cup", seasons=datetime.now().year)
        schedule = fbref.read_schedule().reset_index()
    except Exception as exc:
        print(f"[priors] schedule scrape failed: {type(exc).__name__}: {exc}")
        return 0
    team_stats: pd.DataFrame | None = None
    try:
        team_stats = fbref.read_team_season_stats(stat_type="standard")
    except Exception:
        print("[priors] team season stats not yet published; using schedule only")
    priors = build_priors(schedule, team_stats)
    if priors.empty:
        print("[priors] no completed matches yet; nothing to write")
        return 0
    priors.to_parquet(parquet_path, compression="zstd")
    print(f"[priors] wrote {len(priors)} team priors to {parquet_path.name}")
    return int(len(priors))


def load_priors(parquet_path: Path = PRIORS_PARQUET) -> pd.DataFrame:
    """Committed priors table; empty frame when not yet generated."""
    if not Path(parquet_path).exists():
        return pd.DataFrame(columns=PRIOR_COLUMNS)
    try:
        return pd.read_parquet(parquet_path)
    except (OSError, ValueError):
        return pd.DataFrame(columns=PRIOR_COLUMNS)


def fixture_prior(
    home_team: str, away_team: str, priors: pd.DataFrame | None = None
) -> dict[str, object] | None:
    """Context for one fixture: both teams' priors + a scoring-edge value.

    Edge > 0 means the home side's goal difference per match has been
    better so far. Returns None when neither team has priors yet.
    """
    table = load_priors() if priors is None else priors
    if table.empty:
        return None

    def lookup(name: str) -> pd.Series | None:
        hits = table.loc[
            table["team"].astype(str).str.lower() == str(name).lower().strip()
        ]
        return hits.iloc[0] if not hits.empty else None

    home, away = lookup(home_team), lookup(away_team)
    if home is None and away is None:
        return None
    edge = None
    if home is not None and away is not None:
        home_gd = float(home["goals_for_per_match"]) - float(home["goals_against_per_match"])
        away_gd = float(away["goals_for_per_match"]) - float(away["goals_against_per_match"])
        edge = round(home_gd - away_gd, 2)
    return {"home": home, "away": away, "edge": edge}


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "refresh"
    if command == "refresh":
        refresh_priors()
    else:
        print(f"unknown command: {command}")


if __name__ == "__main__":
    main()
