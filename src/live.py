"""Free live data connectors for real World Cup matches.

Two zero-cost, no-key public sources feed the agents during live matches:

  ESPN site API  -> scoreboard (fixtures, state, clock) and per-match
                    play-by-play commentary for the Match Progression Agent
  Bluesky search -> real-time fan posts mentioning the teams, mapped onto
                    match minutes for the Social Listening Agent

Every fetcher degrades gracefully: any network or schema failure returns an
empty frame/series so callers can fall back to the deterministic simulator.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

ESPN_LEAGUE = os.environ.get("ESPN_LEAGUE", "fifa.world")
ESPN_BASE = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN_LEAGUE}"
BLUESKY_SEARCH = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
USER_AGENT = {"User-Agent": "wc-sentiment-engine/0.1"}

_MINUTE_PREFIX = re.compile(r"(\d+)")

SCOREBOARD_COLUMNS = [
    "event_id",
    "name",
    "short_name",
    "kickoff_utc",
    "state",
    "clock_minute",
    "home_team",
    "away_team",
    "score",
]


def _empty_scoreboard() -> pd.DataFrame:
    return pd.DataFrame(columns=SCOREBOARD_COLUMNS)


def _parse_kickoff(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def parse_scoreboard(payload: dict) -> pd.DataFrame:
    """Normalize an ESPN scoreboard payload into one row per fixture."""
    rows: list[dict[str, object]] = []
    for event in payload.get("events", []) or []:
        competition = (event.get("competitions") or [{}])[0]
        status = competition.get("status") or event.get("status") or {}
        clock_match = _MINUTE_PREFIX.search(str(status.get("displayClock", "")))
        competitors = competition.get("competitors") or []
        home = next(
            (c for c in competitors if c.get("homeAway") == "home"), {}
        )
        away = next(
            (c for c in competitors if c.get("homeAway") == "away"), {}
        )
        rows.append(
            {
                "event_id": str(event.get("id", "")),
                "name": str(event.get("name", "")),
                "short_name": str(event.get("shortName", "")),
                "kickoff_utc": _parse_kickoff(str(event.get("date", ""))),
                "state": str((status.get("type") or {}).get("state", "")),
                "clock_minute": int(clock_match.group(1)) if clock_match else 0,
                "home_team": str((home.get("team") or {}).get("displayName", "")),
                "away_team": str((away.get("team") or {}).get("displayName", "")),
                "score": f"{home.get('score', '0')}-{away.get('score', '0')}",
            }
        )
    if not rows:
        return _empty_scoreboard()
    return pd.DataFrame(rows, columns=SCOREBOARD_COLUMNS)


def fetch_scoreboard(timeout: float = 15.0) -> pd.DataFrame:
    """Today's World Cup fixtures with live state (pre / in / post)."""
    try:
        response = requests.get(
            f"{ESPN_BASE}/scoreboard", timeout=timeout, headers=USER_AGENT
        )
        response.raise_for_status()
        return parse_scoreboard(response.json())
    except (requests.RequestException, ValueError):
        return _empty_scoreboard()


def parse_commentary_payload(payload: dict) -> pd.Series:
    """Convert ESPN summary commentary into the engine's minute-line format.

    Output lines look like "23' shot on target ..." which the model-layer
    commentary regex already understands. Entries without a numeric clock
    (half-time notes, lineups) are dropped.
    """
    lines: list[str] = []
    for entry in payload.get("commentary", []) or []:
        text = str(entry.get("text", "")).strip()
        display = str((entry.get("time") or {}).get("displayValue", ""))
        clock = _MINUTE_PREFIX.search(display)
        if not text or clock is None:
            continue
        lines.append(f"{clock.group(1)}' {text}")
    return pd.Series(lines, dtype="str", name="line")


def fetch_match_commentary(event_id: str, timeout: float = 15.0) -> pd.Series:
    """Live play-by-play text for one fixture, oldest first."""
    try:
        response = requests.get(
            f"{ESPN_BASE}/summary",
            params={"event": event_id},
            timeout=timeout,
            headers=USER_AGENT,
        )
        response.raise_for_status()
        lines = parse_commentary_payload(response.json())
    except (requests.RequestException, ValueError):
        return pd.Series(dtype="str", name="line")
    return lines.iloc[::-1].reset_index(drop=True)


def fetch_crowd_posts(
    terms: list[str], limit: int = 50, timeout: float = 15.0
) -> pd.DataFrame:
    """Latest Bluesky posts mentioning any search term, deduplicated.

    Returns columns: created_utc (tz-aware datetime), message (str).
    """
    rows: list[dict[str, object]] = []
    for term in terms:
        try:
            response = requests.get(
                BLUESKY_SEARCH,
                params={"q": term, "limit": limit, "sort": "latest"},
                timeout=timeout,
                headers=USER_AGENT,
            )
            response.raise_for_status()
            posts = response.json().get("posts", []) or []
        except (requests.RequestException, ValueError):
            continue
        for post in posts:
            record = post.get("record") or {}
            created = _parse_kickoff(str(record.get("createdAt", "")))
            text = str(record.get("text", "")).strip()
            if created is None or not text:
                continue
            rows.append(
                {"uri": str(post.get("uri", "")), "created_utc": created, "message": text}
            )
    if not rows:
        return pd.DataFrame(columns=["created_utc", "message"])
    frame = pd.DataFrame(rows).drop_duplicates(subset="uri")
    return frame[["created_utc", "message"]].sort_values("created_utc").reset_index(drop=True)


def posts_to_chat(
    posts: pd.DataFrame, kickoff_utc: datetime, max_minute: int = 130
) -> pd.DataFrame:
    """Map timestamped posts onto match minutes for the listening agent.

    Posts up to 15 minutes pre-kickoff are clamped to minute 0; anything
    older is discarded as unrelated pre-match chatter.
    """
    if posts.empty or kickoff_utc is None:
        return pd.DataFrame(columns=["minute", "message"]).astype(
            {"minute": "int64", "message": "str"}
        )
    kickoff = pd.Timestamp(kickoff_utc)
    stamps = pd.to_datetime(posts["created_utc"], utc=True, errors="coerce")
    recent = posts.loc[stamps.notna() & (stamps >= kickoff - timedelta(minutes=15))].copy()
    if recent.empty:
        return pd.DataFrame(columns=["minute", "message"]).astype(
            {"minute": "int64", "message": "str"}
        )
    elapsed = (stamps.loc[recent.index] - kickoff).dt.total_seconds() // 60
    recent["minute"] = elapsed.clip(lower=0, upper=max_minute).astype("int64")
    columns = ["minute", "message"] + (["source"] if "source" in recent.columns else [])
    return recent[columns].reset_index(drop=True)


POST_GRACE = timedelta(minutes=15)
MATCH_MAX_DURATION = timedelta(minutes=180)


def capture_phase(
    state: str,
    kickoff_utc: datetime,
    now: datetime | None = None,
    post_first_seen: datetime | None = None,
) -> str:
    """Start/stop filter for data collection around one fixture.

    Returns one of:
      pre         - before kickoff, nothing to collect yet
      live        - match in progress, full collection
      post-window - finished within the 15-minute grace window: keep
                    collecting to capture the post-match emotional settle
      frozen      - grace window elapsed (or match long over): stop all
                    fetching and serve archived data only
    """
    current = utc_now() if now is None else now
    if state == "pre":
        return "pre"
    if state == "in":
        return "live"
    if post_first_seen is not None and current - post_first_seen >= POST_GRACE:
        return "frozen"
    if kickoff_utc is not None and current >= kickoff_utc + MATCH_MAX_DURATION:
        return "frozen"
    return "post-window"


def current_live_match(scoreboard: pd.DataFrame | None = None) -> pd.Series | None:
    """The first in-progress fixture today, or None outside match windows."""
    board = fetch_scoreboard() if scoreboard is None else scoreboard
    if board.empty:
        return None
    live = board.loc[board["state"] == "in"]
    if live.empty:
        return None
    return live.iloc[0]


def current_capture_match(
    scoreboard: pd.DataFrame | None = None, now: datetime | None = None
) -> pd.Series | None:
    """The fixture the pipeline should collect right now, if any.

    Prefers an in-progress match; otherwise a finished match still inside
    its capture window (so the scheduled runs archive full-time data),
    otherwise None.
    """
    board = fetch_scoreboard() if scoreboard is None else scoreboard
    if board.empty:
        return None
    live = board.loc[board["state"] == "in"]
    if not live.empty:
        return live.iloc[0]
    current = utc_now() if now is None else now
    finished = board.loc[board["state"] == "post"]
    for _, row in finished.iterrows():
        kickoff = row["kickoff_utc"]
        if kickoff is not None and current < kickoff + MATCH_MAX_DURATION:
            return row
    return None


def live_streams(match: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch both live inputs (multi-source fan chat, commentary).

    The returned chat carries a `team` column (home|away|both|neither) so the
    dashboard can break the crowd mood down by team.
    """
    import sources
    import teams

    commentary = fetch_match_commentary(str(match["event_id"]))
    home_team = str(match.get("home_team") or "")
    away_team = str(match.get("away_team") or "")
    terms = [t for t in (home_team, away_team) if t]
    posts = sources.gather_reactions(terms)
    chat = posts_to_chat(posts, match["kickoff_utc"])
    if not chat.empty:
        chat["team"] = teams.tag_reactions(chat["message"], home_team, away_team)
    return chat, commentary


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
