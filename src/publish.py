"""Publish the engine's genuine insights to Bluesky, and read organic engagement.

Deliberately NOT an influence tool: posts are honest analytics from the engine,
clearly labelled as automated, sent from the user's own account - manually, or
via an explicit opt-in auto-post of the data-backed underdog case when an
overreaction moment fires (rate-limited to one per 15 minutes). Engagement is
read descriptively - replies/reposts/likes on your own posts, never framed as
proof the crowd was moved. No thread targeting/injection: hashtags are the
standard fixture tags only.

Posting is key-gated on BLUESKY_HANDLE + BLUESKY_APP_PASSWORD (an app password
from Bluesky Settings, never the account password). Reading engagement uses the
keyless public AppView.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests

PDS = "https://bsky.social/xrpc"
PUBLIC = "https://public.api.bsky.app/xrpc"
LABEL = "🤖 automated analytics from the WC Crowd Mood Engine"
MAX_CHARS = 290  # ponytail: grapheme-approx of Bluesky's 300 limit; truncate under it


def enabled() -> bool:
    return bool(os.environ.get("BLUESKY_HANDLE") and os.environ.get("BLUESKY_APP_PASSWORD"))


def draft_post(state: pd.DataFrame, headline: str = "") -> str:
    """Compose an honest, labelled insight post from the latest scored minute."""
    if state is None or state.empty:
        return f"Waiting for enough crowd data to post an insight.\n{LABEL}"
    latest = state.iloc[-1]
    match = str(latest.get("match_id", "")).replace("ESPN-", "match ")
    minute = int(latest.get("minute", 0))
    mood = str(latest.get("dominant_emotion", "neutral")).title()
    gap = float(latest.get("arbitrage_index", 0.0))
    situation = str(latest.get("situation", "")).replace("_", " ")
    lines = [
        f"⚽ WC Crowd Mood — {match}, {minute}'",
        f"Loudest fan emotion: {mood}",
        f"Hype-vs-Reality gap: {gap:.2f} ({situation})",
    ]
    if headline:
        lines.append(headline)
    lines.append(LABEL)
    text = "\n".join(lines)
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1].rstrip() + "…"


def match_hashtags(short_name: str) -> str:
    """Standard fixture hashtags from ESPN's "MAR @ FRA" short name.

    Honest discoverability only (the tags every sports account uses) - no
    scanning or targeting of active fan threads.
    """
    parts = [p.strip() for p in str(short_name or "").split("@")]
    if len(parts) == 2 and all(parts):
        away, home = parts
        return f"#{home}{away} #FIFAWorldCup"
    return "#FIFAWorldCup"


def _stat_pair(match_stats: dict, key: str, team: str) -> str | None:
    value = (match_stats or {}).get(team, {}).get(key)
    return None if value in (None, "") else str(value)


def underdog_case(
    match_row: pd.Series,
    state: pd.DataFrame,
    match_stats: dict | None = None,
    keeper: dict | None = None,
    market: dict | None = None,
) -> str | None:
    """Data-backed case for the side least likely to win right now, or None.

    Underdog = the trailing side (or the market's least-likely side when the
    score is level and odds exist). The post cites ONLY real numbers that
    genuinely support the case (shots on target, attacking threat, keeper
    workload); if the data offers no support, returns None rather than
    fabricating belief - honesty over output.
    """
    home = str(match_row.get("home_team") or "")
    away = str(match_row.get("away_team") or "")
    score = str(match_row.get("score") or "")
    parts = score.replace("–", "-").split("-")
    try:
        home_goals, away_goals = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None
    if home_goals < away_goals:
        underdog, opponent = home, away
    elif away_goals < home_goals:
        underdog, opponent = away, home
    elif market and market.get("home_prob") and market.get("away_prob"):
        underdog, opponent = (
            (home, away)
            if float(market["home_prob"]) < float(market["away_prob"])
            else (away, home)
        )
    else:
        return None  # level game, no market: no clear underdog, no post
    minute = int(state.iloc[-1].get("minute", 0)) if state is not None and not state.empty else 0
    evidence: list[str] = []
    on_target = _stat_pair(match_stats, "shotsOnTarget", underdog)
    if on_target and on_target not in ("0", "0.0"):
        evidence.append(f"{on_target} shots on target")
    threat = (
        float(state.iloc[-1].get("delta_xg_10min", 0.0))
        if state is not None and not state.empty
        else 0.0
    )
    if threat >= 0.3:
        evidence.append(f"live attacking threat {threat:.2f}")
    opp_keeper = (keeper or {}).get(opponent, {})
    saves = float(opp_keeper.get("saves", 0.0) or 0.0)
    if saves >= 3:
        evidence.append(
            f"{opponent}'s keeper has needed {saves:.0f} saves to keep this scoreline"
        )
    if not evidence:
        return None  # the data does not support a belief case; do not invent one
    # The hashtags and automation label must always survive truncation - the
    # transparency label is the point; the evidence line absorbs any cut.
    tail = "\n".join([match_hashtags(str(match_row.get("short_name") or "")), LABEL])
    head = "\n".join(
        [
            f"⚽ {underdog} aren't done yet — the data at {minute}': "
            + "; ".join(evidence)
            + ".",
            f"Score {score} vs {opponent}, but the underlying numbers say this "
            "match is still alive.",
        ]
    )
    budget = MAX_CHARS - len(tail) - 1
    if len(head) > budget:
        head = head[: budget - 1].rstrip() + "…"
    return f"{head}\n{tail}"


def should_autopost(last_posted_iso: str, now: datetime | None = None, min_gap_minutes: int = 15) -> bool:
    """Rate limit for trigger-based posting: at most one post per gap."""
    if not last_posted_iso:
        return True
    try:
        last = datetime.fromisoformat(last_posted_iso)
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    moment = now or datetime.now(timezone.utc)
    return (moment - last).total_seconds() >= min_gap_minutes * 60


def post_insight(text: str, timeout: float = 15.0) -> str | None:
    """Post `text` from the configured account; returns the post URI or None."""
    handle = os.environ.get("BLUESKY_HANDLE", "")
    password = os.environ.get("BLUESKY_APP_PASSWORD", "")
    if not handle or not password or not text.strip():
        return None
    try:
        session = requests.post(
            f"{PDS}/com.atproto.server.createSession",
            json={"identifier": handle, "password": password},
            timeout=timeout,
        )
        session.raise_for_status()
        auth = session.json()
        record = requests.post(
            f"{PDS}/com.atproto.repo.createRecord",
            headers={"Authorization": f"Bearer {auth['accessJwt']}"},
            json={
                "repo": auth["did"],
                "collection": "app.bsky.feed.post",
                "record": {
                    "$type": "app.bsky.feed.post",
                    "text": text,
                    "createdAt": datetime.now(timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z"),
                },
            },
            timeout=timeout,
        )
        record.raise_for_status()
        return str(record.json().get("uri", "")) or None
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"[publish] post failed ({type(exc).__name__}: {exc})")
        return None


def recent_engagement(handle: str = "", limit: int = 15, timeout: float = 15.0) -> pd.DataFrame:
    """Your account's recent posts with organic engagement counts (keyless read)."""
    actor = handle or os.environ.get("BLUESKY_HANDLE", "")
    columns = ["posted_at", "text", "likes", "reposts", "replies"]
    if not actor:
        return pd.DataFrame(columns=columns)
    try:
        response = requests.get(
            f"{PUBLIC}/app.bsky.feed.getAuthorFeed",
            params={"actor": actor, "limit": limit, "filter": "posts_no_replies"},
            timeout=timeout,
        )
        response.raise_for_status()
        feed = response.json().get("feed", []) or []
    except (requests.RequestException, ValueError):
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "posted_at": (p.get("record") or {}).get("createdAt", ""),
            "text": (p.get("record") or {}).get("text", ""),
            "likes": int(p.get("likeCount", 0) or 0),
            "reposts": int(p.get("repostCount", 0) or 0),
            "replies": int(p.get("replyCount", 0) or 0),
        }
        for item in feed
        if (p := item.get("post"))
    ]
    return pd.DataFrame(rows, columns=columns)
