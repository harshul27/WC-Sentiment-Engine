"""Advanced match metrics: ESPN boxscore backbone + optional Sofascore.

ESPN's public summary endpoint carries a live team-statistics block
(possession, shots, shots on target, corners, saves) that works keyless
from any environment - this is the always-available backbone used for the
match control index and the takeaway generator.

When ScraperFC is installed and ENABLE_SOFASCORE=1 (typically a local
machine - Sofascore blocks most datacenter IPs), the layer additionally
pulls Sofascore's attack-momentum graph and shot xG for the same fixture.
Every fetcher returns empty data on failure; nothing here can take the
product down.
"""
from __future__ import annotations

import os
import re

import pandas as pd
import requests

ESPN_LEAGUE = os.environ.get("ESPN_LEAGUE", "fifa.world")
ESPN_BASE = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN_LEAGUE}"
USER_AGENT = {"User-Agent": "wc-sentiment-engine/0.1"}

KEY_STATS: tuple[str, ...] = (
    "possessionPct",
    "totalShots",
    "shotsOnTarget",
    "wonCorners",
    "saves",
    "foulsCommitted",
)


def parse_boxscore(payload: dict) -> dict[str, dict[str, str]]:
    """ESPN summary boxscore -> {team_name: {stat: value}} for key stats."""
    result: dict[str, dict[str, str]] = {}
    for team in payload.get("boxscore", {}).get("teams", []) or []:
        name = str((team.get("team") or {}).get("displayName", "")).strip()
        if not name:
            continue
        stats = {
            str(s.get("name")): str(s.get("displayValue", ""))
            for s in team.get("statistics", []) or []
            if s.get("name") in KEY_STATS
        }
        if stats:
            result[name] = stats
    return result


def fetch_boxscore(event_id: str, timeout: float = 15.0) -> dict[str, dict[str, str]]:
    """Live team statistics for one fixture; empty dict on any failure."""
    try:
        response = requests.get(
            f"{ESPN_BASE}/summary",
            params={"event": event_id},
            timeout=timeout,
            headers=USER_AGENT,
        )
        response.raise_for_status()
        return parse_boxscore(response.json())
    except (requests.RequestException, ValueError):
        return {}


def _stat_share(stats: dict[str, dict[str, str]], key: str) -> float | None:
    """Home-perspective share of a numeric stat across the two teams."""
    values: list[float] = []
    for team_stats in stats.values():
        raw = re.sub(r"[^\d.]", "", str(team_stats.get(key, "")))
        if raw == "":
            return None
        values.append(float(raw))
    if len(values) != 2 or sum(values) == 0:
        return None
    return values[0] / (values[0] + values[1])


def control_index(stats: dict[str, dict[str, str]]) -> float | None:
    """Composite match-control share in [0, 1] from the first team's view.

    Blend of possession, shots-on-target, and total-shot shares. 0.5 means
    an even contest; values near the extremes mean one side dominates.
    Returns None when the boxscore lacks the needed numbers.
    """
    weights = {"possessionPct": 0.45, "shotsOnTarget": 0.35, "totalShots": 0.20}
    total_weight = 0.0
    blended = 0.0
    for key, weight in weights.items():
        share = _stat_share(stats, key)
        if share is None:
            continue
        blended += weight * share
        total_weight += weight
    if total_weight == 0.0:
        return None
    return blended / total_weight


def sofascore_enabled() -> bool:
    return os.environ.get("ENABLE_SOFASCORE", "") == "1"


def fetch_sofascore_momentum(home_team: str, away_team: str) -> pd.DataFrame:
    """Sofascore attack-momentum series via ScraperFC (optional enrichment).

    Returns columns (minute, momentum) where momentum > 0 favours the home
    side. Requires ScraperFC installed, ENABLE_SOFASCORE=1, and a network
    position Sofascore accepts; otherwise returns an empty frame.
    """
    empty = pd.DataFrame(columns=["minute", "momentum"]).astype(
        {"minute": "int64", "momentum": "float64"}
    )
    if not sofascore_enabled():
        return empty
    try:
        from ScraperFC import Sofascore

        scraper = Sofascore()
        year = pd.Timestamp.utcnow().year
        matches = scraper.get_match_dicts(str(year), "FIFA World Cup")
        target = None
        for match in matches:
            home = str(match.get("homeTeam", {}).get("name", "")).lower()
            away = str(match.get("awayTeam", {}).get("name", "")).lower()
            if home_team.lower() in home and away_team.lower() in away:
                target = match
                break
        if target is None:
            return empty
        momentum = scraper.scrape_match_momentum(str(target["id"]))
        if momentum is None or momentum.empty:
            return empty
        frame = momentum.rename(columns={"value": "momentum"})
        frame["minute"] = frame["minute"].astype("float64").round().astype("int64")
        frame["momentum"] = frame["momentum"].astype("float64") / 100.0
        return frame[["minute", "momentum"]]
    except Exception:
        return empty


def match_context(
    event_id: str, home_team: str = "", away_team: str = ""
) -> dict[str, object]:
    """Everything the takeaway generator and UI need about the match state."""
    stats = fetch_boxscore(event_id)
    context: dict[str, object] = {
        "stats": stats,
        "control_index": control_index(stats),
        "momentum": fetch_sofascore_momentum(home_team, away_team)
        if home_team and away_team
        else pd.DataFrame(columns=["minute", "momentum"]),
    }
    return context
