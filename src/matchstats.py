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

# Second tier shown in an expander: passing/defending depth from the same
# ESPN boxscore payload (verified live: 28 stats are published per team).
ADVANCED_STATS: tuple[str, ...] = (
    "accuratePasses",
    "passPct",
    "totalTackles",
    "interceptions",
    "effectiveClearance",
    "blockedShots",
    "accurateCrosses",
)

# Per-player stat names worth carrying (from rosters[].roster[].stats).
PLAYER_STATS: tuple[str, ...] = (
    "totalGoals",
    "goalAssists",
    "totalShots",
    "shotsOnTarget",
    "saves",
    "shotsFaced",
    "goalsConceded",
    "foulsCommitted",
    "yellowCards",
    "redCards",
)


def parse_boxscore(payload: dict) -> dict[str, dict[str, str]]:
    """ESPN summary boxscore -> {team_name: {stat: value}} for key + advanced stats."""
    wanted = set(KEY_STATS) | set(ADVANCED_STATS)
    result: dict[str, dict[str, str]] = {}
    for team in payload.get("boxscore", {}).get("teams", []) or []:
        name = str((team.get("team") or {}).get("displayName", "")).strip()
        if not name:
            continue
        stats = {
            str(s.get("name")): str(s.get("displayValue", ""))
            for s in team.get("statistics", []) or []
            if s.get("name") in wanted
        }
        if stats:
            result[name] = stats
    return result


def parse_player_stats(payload: dict) -> dict[str, list[dict[str, object]]]:
    """ESPN summary rosters -> {team_name: [player rows]}.

    Each row: name, position, starter, plus the PLAYER_STATS values (floats).
    Players with no stats block (unused subs) are skipped.
    """
    result: dict[str, list[dict[str, object]]] = {}
    for side in payload.get("rosters", []) or []:
        team = str((side.get("team") or {}).get("displayName", "")).strip()
        if not team:
            continue
        rows: list[dict[str, object]] = []
        for entry in side.get("roster", []) or []:
            stats = {
                str(s.get("name")): float(s.get("value", 0.0) or 0.0)
                for s in entry.get("stats", []) or []
            }
            if not stats:
                continue
            rows.append(
                {
                    "name": str((entry.get("athlete") or {}).get("displayName", "")),
                    "position": str(
                        (entry.get("position") or {}).get("abbreviation", "")
                    ),
                    "starter": bool(entry.get("starter", False)),
                    **{k: stats.get(k, 0.0) for k in PLAYER_STATS},
                }
            )
        if rows:
            result[team] = rows
    return result


def parse_leaders(payload: dict) -> dict[str, dict[str, str]]:
    """ESPN summary leaders -> {team_name: {category: "Player (value)"}}."""
    result: dict[str, dict[str, str]] = {}
    for block in payload.get("leaders", []) or []:
        team = str((block.get("team") or {}).get("displayName", "")).strip()
        if not team:
            continue
        categories: dict[str, str] = {}
        for cat in block.get("leaders", []) or []:
            top = (cat.get("leaders") or [{}])[0]
            player = str((top.get("athlete") or {}).get("displayName", ""))
            value = str(top.get("displayValue", ""))
            if player:
                categories[str(cat.get("name", ""))] = f"{player} ({value})"
        if categories:
            result[team] = categories
    return result


def parse_key_events(payload: dict) -> list[dict[str, object]]:
    """ESPN summary keyEvents -> [{minute, type, team, text}] (typed events only)."""
    events: list[dict[str, object]] = []
    for event in payload.get("keyEvents", []) or []:
        kind = str((event.get("type") or {}).get("type", ""))
        if not kind:
            continue
        clock = event.get("clock") or {}
        seconds = float(clock.get("value", 0.0) or 0.0)  # ESPN clock is seconds
        events.append(
            {
                "minute": int(seconds // 60),
                "type": kind,
                "team": str((event.get("team") or {}).get("displayName", "")),
                "text": str(event.get("text", "")),
            }
        )
    return events


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


def fetch_match_detail(event_id: str, timeout: float = 15.0) -> dict[str, object]:
    """Everything the live panel needs from ONE summary request.

    Returns {stats, players, leaders, key_events}; every part degrades to
    empty on failure so the panel can render whatever arrived.
    """
    empty: dict[str, object] = {
        "stats": {},
        "players": {},
        "leaders": {},
        "key_events": [],
    }
    try:
        response = requests.get(
            f"{ESPN_BASE}/summary",
            params={"event": event_id},
            timeout=timeout,
            headers=USER_AGENT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return empty
    return {
        "stats": parse_boxscore(payload),
        "players": parse_player_stats(payload),
        "leaders": parse_leaders(payload),
        "key_events": parse_key_events(payload),
    }


def keeper_pressure(
    players: dict[str, list[dict[str, object]]],
) -> dict[str, dict[str, float]]:
    """Per-team goalkeeper workload: {team: {keeper, saves, shots_faced, conceded}}.

    "Fans furious while the keeper is making save after save" is a classic
    mood-vs-game conflict; this makes that context available to the takeaway
    and consistency layers.
    """
    result: dict[str, dict[str, float]] = {}
    for team, rows in (players or {}).items():
        keepers = [r for r in rows if str(r.get("position")) == "G"]
        if not keepers:
            continue
        keeper = max(keepers, key=lambda r: float(r.get("saves", 0.0) or 0.0))
        result[team] = {
            "keeper": str(keeper.get("name", "")),
            "saves": float(keeper.get("saves", 0.0) or 0.0),
            "shots_faced": float(keeper.get("shotsFaced", 0.0) or 0.0),
            "conceded": float(keeper.get("goalsConceded", 0.0) or 0.0),
        }
    return result


def top_performers(leaders: dict[str, dict[str, str]]) -> dict[str, str]:
    """One readable line per team from the ESPN leader categories."""
    labels = {
        "totalShots": "shots",
        "accuratePasses": "passes",
        "defensiveInterventions": "defensive actions",
        "saves": "saves",
    }
    lines: dict[str, str] = {}
    for team, categories in (leaders or {}).items():
        parts = [
            f"{categories[key]} {label}"
            for key, label in labels.items()
            if key in categories
        ]
        if parts:
            lines[team] = "; ".join(parts)
    return lines


def goal_scorers(key_events: list[dict[str, object]]) -> list[str]:
    """Readable goal lines ("23' Mexico — <text>") from parsed key events."""
    lines: list[str] = []
    for event in key_events or []:
        if str(event.get("type", "")) not in ("goal", "penalty--scored", "own-goal"):
            continue
        team = str(event.get("team", ""))
        text = str(event.get("text", "")) or str(event.get("type", ""))
        lines.append(f"{int(event.get('minute', 0))}' {team} — {text}".strip())
    return lines


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
