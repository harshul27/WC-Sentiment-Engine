"""Publish the engine's genuine insights to Bluesky, and read organic engagement.

Deliberately NOT an influence tool: posts are honest analytics from the engine,
clearly labelled as automated, sent from the user's own account only when they
click (no autonomous posting), and engagement is read descriptively - replies/
reposts/likes on your own posts, not an experiment on the crowd.

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
