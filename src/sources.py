"""Multi-platform crowd reaction aggregator with a 200-comment window.

Free sources, deepest coverage first:

  Bluesky   - public search API, no key required (always on)
  Mastodon  - public hashtag timelines, no key required (always on)
  Reddit    - r/soccer + r/worldcup match-thread comments via the free
              official OAuth app flow; activates when REDDIT_CLIENT_ID and
              REDDIT_CLIENT_SECRET are set
  YouTube   - live stream chat via the free YouTube Data API quota;
              activates when YOUTUBE_API_KEY is set

Every connector returns the unified schema (created_utc, message, source)
and degrades to an empty frame on any failure, so the aggregate is always
usable regardless of which credentials exist in the environment.
"""
from __future__ import annotations

import os
import re

import pandas as pd
import requests

COMMENT_WINDOW = 200
USER_AGENT = {"User-Agent": "wc-sentiment-engine/0.1"}
BLUESKY_SEARCH = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
MASTODON_INSTANCE = os.environ.get("MASTODON_INSTANCE", "https://mastodon.social")

_HTML_TAG = re.compile(r"<[^>]+>")

REACTION_COLUMNS = ["created_utc", "message", "source"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=REACTION_COLUMNS)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty()
    frame = pd.DataFrame(rows)
    frame["created_utc"] = pd.to_datetime(frame["created_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created_utc"])
    frame["message"] = frame["message"].astype(str).str.strip()
    return frame.loc[frame["message"] != "", REACTION_COLUMNS]


def fetch_bluesky(terms: list[str], limit: int = 50, timeout: float = 15.0) -> pd.DataFrame:
    """Latest Bluesky posts mentioning any search term."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
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
            uri = str(post.get("uri", ""))
            record = post.get("record") or {}
            if uri in seen:
                continue
            seen.add(uri)
            rows.append(
                {
                    "created_utc": record.get("createdAt"),
                    "message": record.get("text", ""),
                    "source": "bluesky",
                }
            )
    return _frame(rows)


def fetch_mastodon(tags: list[str], limit: int = 40, timeout: float = 15.0) -> pd.DataFrame:
    """Public hashtag timeline posts from a Mastodon instance (no key)."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for tag in tags:
        clean = re.sub(r"[^A-Za-z0-9]", "", tag)
        if not clean:
            continue
        try:
            response = requests.get(
                f"{MASTODON_INSTANCE}/api/v1/timelines/tag/{clean}",
                params={"limit": min(limit, 40)},
                timeout=timeout,
                headers=USER_AGENT,
            )
            response.raise_for_status()
            posts = response.json() or []
        except (requests.RequestException, ValueError):
            continue
        for post in posts:
            post_id = str(post.get("id", ""))
            if post_id in seen:
                continue
            seen.add(post_id)
            rows.append(
                {
                    "created_utc": post.get("created_at"),
                    "message": _HTML_TAG.sub(" ", str(post.get("content", ""))),
                    "source": "mastodon",
                }
            )
    return _frame(rows)


def _reddit_token(timeout: float = 15.0) -> str:
    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return ""
    try:
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            timeout=timeout,
            headers=USER_AGENT,
        )
        response.raise_for_status()
        return str(response.json().get("access_token", ""))
    except (requests.RequestException, ValueError):
        return ""


def fetch_reddit(team_terms: list[str], limit: int = 100, timeout: float = 15.0) -> pd.DataFrame:
    """Newest comments from the most recent r/soccer match thread.

    Requires REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET (free script app from
    reddit.com/prefs/apps); silently skipped otherwise.
    """
    token = _reddit_token(timeout)
    if not token:
        return _empty()
    auth_headers = {**USER_AGENT, "Authorization": f"Bearer {token}"}
    query = "Match Thread " + " ".join(team_terms)
    try:
        search = requests.get(
            "https://oauth.reddit.com/r/soccer+worldcup/search",
            params={"q": query, "sort": "new", "restrict_sr": 1, "t": "day", "limit": 3},
            timeout=timeout,
            headers=auth_headers,
        )
        search.raise_for_status()
        posts = [c["data"] for c in search.json().get("data", {}).get("children", [])]
    except (requests.RequestException, ValueError, KeyError):
        return _empty()
    threads = [p for p in posts if "match thread" in str(p.get("title", "")).lower()]
    if not threads:
        return _empty()
    rows: list[dict[str, object]] = []
    try:
        comments = requests.get(
            f"https://oauth.reddit.com/comments/{threads[0]['id']}",
            params={"sort": "new", "limit": limit},
            timeout=timeout,
            headers=auth_headers,
        )
        comments.raise_for_status()
        listing = comments.json()[1]["data"]["children"]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return _empty()
    for item in listing:
        data = item.get("data", {})
        if item.get("kind") != "t1":
            continue
        rows.append(
            {
                "created_utc": pd.Timestamp(float(data.get("created_utc", 0)), unit="s", tz="UTC"),
                "message": data.get("body", ""),
                "source": "reddit",
            }
        )
    return _frame(rows)


def fetch_youtube(query: str, limit: int = 100, timeout: float = 15.0) -> pd.DataFrame:
    """Live chat messages from the top YouTube live stream for the query.

    Requires YOUTUBE_API_KEY (free Data API quota); silently skipped
    otherwise. Costs ~105 quota units per poll, well within the free
    10,000/day allowance at a 60s refresh cadence.
    """
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        return _empty()
    base = "https://www.googleapis.com/youtube/v3"
    try:
        search = requests.get(
            f"{base}/search",
            params={
                "part": "snippet",
                "eventType": "live",
                "type": "video",
                "q": query,
                "maxResults": 1,
                "key": key,
            },
            timeout=timeout,
            headers=USER_AGENT,
        )
        search.raise_for_status()
        items = search.json().get("items", [])
        if not items:
            return _empty()
        video_id = items[0]["id"]["videoId"]
        video = requests.get(
            f"{base}/videos",
            params={"part": "liveStreamingDetails", "id": video_id, "key": key},
            timeout=timeout,
            headers=USER_AGENT,
        )
        video.raise_for_status()
        details = video.json().get("items", [{}])[0].get("liveStreamingDetails", {})
        chat_id = details.get("activeLiveChatId", "")
        if not chat_id:
            return _empty()
        chat = requests.get(
            f"{base}/liveChat/messages",
            params={
                "liveChatId": chat_id,
                "part": "snippet",
                "maxResults": min(limit, 200),
                "key": key,
            },
            timeout=timeout,
            headers=USER_AGENT,
        )
        chat.raise_for_status()
        messages = chat.json().get("items", [])
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return _empty()
    rows = [
        {
            "created_utc": m.get("snippet", {}).get("publishedAt"),
            "message": m.get("snippet", {}).get("displayMessage", ""),
            "source": "youtube",
        }
        for m in messages
    ]
    return _frame(rows)


def gather_reactions(
    team_terms: list[str], window: int = COMMENT_WINDOW
) -> pd.DataFrame:
    """All available sources merged, deduplicated, newest `window` comments."""
    frames = [
        fetch_bluesky(team_terms),
        fetch_mastodon([*team_terms, "worldcup"]),
        fetch_reddit(team_terms),
        fetch_youtube(" vs ".join(team_terms[:2]) + " live"),
    ]
    merged = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else _empty()
    if merged.empty:
        return merged
    merged = merged.drop_duplicates(subset="message")
    return (
        merged.sort_values("created_utc")
        .tail(window)
        .reset_index(drop=True)
    )
