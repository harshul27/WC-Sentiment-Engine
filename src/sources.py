"""Multi-platform crowd reaction aggregator with a 200-comment window.

Sources, deepest coverage first:

  Bluesky   - public search API, no key required (always on)
  Mastodon  - public hashtag timelines, no key required (always on)
  Reddit    - r/soccer + r/worldcup match-thread comments via the free
              official OAuth app flow; activates when REDDIT_CLIENT_ID and
              REDDIT_CLIENT_SECRET are set
  YouTube   - live stream chat via the free YouTube Data API quota;
              activates when YOUTUBE_API_KEY is set
  X         - recent posts via your own xAI/Grok key (the X Search agent
              tool); activates when XAI_API_KEY is set. X has no free public
              firehose, so this is the only first-party way to read X content
              and it bills against your personal xAI account.

Every connector returns the unified schema (created_utc, message, source)
and degrades to an empty frame on any failure, so the aggregate is always
usable regardless of which credentials exist in the environment.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd
import requests

COMMENT_WINDOW = 1000
# Per-author flood cap and minimum readable length keep one spammer or a wall
# of emotes from dominating a minute.
MAX_PER_AUTHOR = 5
MIN_WORDS = 3
USER_AGENT = {"User-Agent": "wc-sentiment-engine/0.1"}
BLUESKY_SEARCH = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
MASTODON_INSTANCE = os.environ.get("MASTODON_INSTANCE", "https://mastodon.social")
# xAI Agent Tools API (the deprecated search_parameters Live Search was
# retired 2026-01-12); the X Search tool is grounded in live X posts.
XAI_RESPONSES = "https://api.x.ai/v1/responses"
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4.3")

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"[@#]\w+")
_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)  # word-ish token, any language

# Automod/chat bots whose posts are not fan reactions.
_BOT_AUTHORS = {
    "automoderator",
    "nightbot",
    "streamelements",
    "moderator",
    "wadu",
    "fossabot",
    "soccerbot",
    "sports_bot",
    "botrickbateman",
}

# Football vocabulary (multi-lingual) used as a lenient relevance gate so a
# generic on-topic reaction ("what a goal") survives even without a team name.
_FOOTBALL_KEYWORDS = {
    "goal", "gol", "golazo", "golaco", "but", "penalty", "penal", "ref", "var",
    "offside", "keeper", "save", "shot", "corner", "foul", "header", "miss",
    "score", "draw", "win", "lose", "red card", "yellow", "match", "game",
    "tournament", "world cup", "worldcup",
}

REACTION_COLUMNS = ["created_utc", "message", "source", "author"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=REACTION_COLUMNS)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty()
    frame = pd.DataFrame(rows)
    if "author" not in frame.columns:
        frame["author"] = ""
    frame["author"] = frame["author"].fillna("").astype(str)
    frame["created_utc"] = pd.to_datetime(frame["created_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created_utc"])
    frame["message"] = frame["message"].astype(str).str.strip()
    return frame.loc[frame["message"] != "", REACTION_COLUMNS]


def fetch_bluesky(terms: list[str], limit: int = 100, timeout: float = 15.0) -> pd.DataFrame:
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
                    "author": str((post.get("author") or {}).get("handle", "")),
                }
            )
    return _frame(rows)


def fetch_bluesky_window(
    terms: list[str],
    since: str,
    until: str,
    max_pages: int = 4,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """Historical Bluesky posts for a time window (ISO8601 since/until).

    Used by the reactions backfill: Bluesky's searchPosts supports since/until
    with cursor pagination, so past match windows remain queryable. Pages are
    capped per term to stay polite.
    """
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for term in terms:
        cursor = ""
        for _ in range(max_pages):
            params = {
                "q": term,
                "limit": 100,
                "sort": "latest",
                "since": since,
                "until": until,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                response = requests.get(
                    BLUESKY_SEARCH, params=params, timeout=timeout, headers=USER_AGENT
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError):
                break
            for post in payload.get("posts", []) or []:
                uri = str(post.get("uri", ""))
                if uri in seen:
                    continue
                seen.add(uri)
                record = post.get("record") or {}
                rows.append(
                    {
                        "created_utc": record.get("createdAt"),
                        "message": record.get("text", ""),
                        "source": "bluesky",
                        "author": str((post.get("author") or {}).get("handle", "")),
                    }
                )
            cursor = str(payload.get("cursor", ""))
            if not cursor:
                break
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
                    "author": str((post.get("account") or {}).get("acct", "")),
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


def fetch_reddit(
    team_terms: list[str],
    limit: int = 100,
    timeout: float = 15.0,
    time_filter: str = "day",
) -> pd.DataFrame:
    """Newest comments from the most recent r/soccer match thread.

    Requires REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET (free script app from
    reddit.com/prefs/apps); silently skipped otherwise. time_filter widens the
    thread search window ("day" live; "year" for the historical backfill).
    """
    token = _reddit_token(timeout)
    if not token:
        return _empty()
    auth_headers = {**USER_AGENT, "Authorization": f"Bearer {token}"}
    query = "Match Thread " + " ".join(team_terms)
    try:
        search = requests.get(
            "https://oauth.reddit.com/r/soccer+worldcup/search",
            params={
                "q": query,
                "sort": "new",
                "restrict_sr": 1,
                "t": time_filter,
                "limit": 3,
            },
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
                "author": str(data.get("author", "")),
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
                "part": "snippet,authorDetails",
                "maxResults": min(limit, 500),
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
            "author": str(m.get("authorDetails", {}).get("displayName", "")),
        }
        for m in messages
    ]
    return _frame(rows)


def _extract_response_text(payload: dict) -> str:
    """Pull the assistant message text out of an xAI /responses payload.

    Tolerant of both the convenience ``output_text`` field and the raw
    ``output`` array of message items so a minor schema shift can't break it.
    """
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
    return "\n".join(chunks)


def _parse_post_json(text: str) -> list[dict[str, object]]:
    """Extract the JSON array of posts the model was asked to return."""
    if not text:
        return []
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        return []
    try:
        data = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def fetch_x(team_terms: list[str], limit: int = 50, timeout: float = 30.0) -> pd.DataFrame:
    """Recent X posts about the fixture via xAI's Grok X Search tool.

    Requires XAI_API_KEY (your own xAI/Grok key); silently skipped otherwise.
    The X Search tool grounds Grok in live X posts; Grok is asked to return
    them as a JSON array, which is mapped to the unified reaction schema.
    Posts without a usable timestamp fall back to the current time so recent
    reactions are still counted on the live tick.
    """
    key = os.environ.get("XAI_API_KEY", "")
    terms = [str(t) for t in team_terms if t]
    if not key or not terms:
        return _empty()
    focus = " vs ".join(terms[:2]) if len(terms) >= 2 else terms[0]
    instruction = (
        "Use X Search to find the most recent fan posts reacting to the live "
        f"football match {focus}. Return ONLY a compact JSON array (no prose, no "
        f"code fences) of up to {min(limit, 50)} objects, each with keys "
        '"text" (the verbatim post text) and "created_at" (the post\'s ISO 8601 '
        "timestamp). Exclude retweets, advertisements, and media-only posts."
    )
    payload = {
        "model": XAI_MODEL,
        "input": [{"role": "user", "content": instruction}],
        "tools": [{"type": "x_search"}],
    }
    try:
        response = requests.post(
            XAI_RESPONSES,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        posts = _parse_post_json(_extract_response_text(response.json()))
    except (requests.RequestException, ValueError):
        return _empty()
    now = pd.Timestamp.now(tz="UTC")
    rows = [
        {
            "created_utc": post.get("created_at") or now,
            "message": post.get("text", ""),
            "source": "x",
            "author": str(post.get("handle") or post.get("author") or ""),
        }
        for post in posts
    ]
    return _frame(rows)


def _normalise(message: str) -> str:
    """Lowercased, URL/mention-stripped, repeated-char-collapsed key for dedup."""
    text = _URL.sub(" ", str(message).lower())
    text = _MENTION.sub(" ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # gooooal -> gooal
    return re.sub(r"\s+", " ", text).strip()


def _is_readable(message: str) -> bool:
    """Keep only messages with enough real words (not pure emoji/links/punct)."""
    stripped = _MENTION.sub(" ", _URL.sub(" ", str(message)))
    return len(_WORD.findall(stripped)) >= MIN_WORDS


def _is_relevant(message: str, terms: list[str]) -> bool:
    """Lenient on-topic gate: mentions a team or any football keyword."""
    if not terms:
        return True
    low = str(message).lower()
    if any(t.lower() in low for t in terms if t):
        return True
    return any(kw in low for kw in _FOOTBALL_KEYWORDS)


def clean_reactions(
    frame: pd.DataFrame, team_terms: list[str] | None = None
) -> pd.DataFrame:
    """Drop bots, low-content, off-topic, and per-author floods.

    Removes the jargon/trash that erodes the mood signal: automod/chat bots,
    emoji/link-only or too-short posts, off-topic chatter, near-duplicate
    copypasta, and any single author posting more than MAX_PER_AUTHOR times.
    """
    if frame is None or frame.empty:
        return _empty()
    work = frame.copy()
    if "author" not in work.columns:
        work["author"] = ""
    author = work["author"].fillna("").astype(str).str.lower()
    work = work.loc[~author.isin(_BOT_AUTHORS) & ~author.str.endswith("bot")]
    work = work.loc[work["message"].map(_is_readable)]
    if team_terms is not None:
        work = work.loc[work["message"].map(lambda m: _is_relevant(m, team_terms))]
    if work.empty:
        return _empty()
    work = work.loc[~work["message"].map(_normalise).duplicated()]
    nonblank = work["author"].astype(str).str.strip() != ""
    rank = work.groupby("author").cumcount()
    work = work.loc[~nonblank | (rank < MAX_PER_AUTHOR)]
    return work.reset_index(drop=True)


def merge_window(
    prior: pd.DataFrame | None, new: pd.DataFrame, window: int = COMMENT_WINDOW
) -> pd.DataFrame:
    """Accumulate reactions across ticks: union, de-dup, keep newest `window`.

    Lets the live view build a rolling buffer (up to `window` reactions) over a
    match instead of replacing it every refresh.
    """
    parts = [f for f in (prior, new) if f is not None and not f.empty]
    if not parts:
        return _empty()
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.loc[~merged["message"].map(_normalise).duplicated()]
    return (
        merged.sort_values("created_utc").tail(window).reset_index(drop=True)
    )


def gather_reactions(
    team_terms: list[str], window: int = COMMENT_WINDOW
) -> pd.DataFrame:
    """All sources merged, cleaned of jargon/bots, newest `window` comments."""
    frames = [
        fetch_bluesky(team_terms),
        fetch_mastodon([*team_terms, "worldcup"]),
        fetch_reddit(team_terms, limit=150),
        fetch_youtube(" vs ".join(team_terms[:2]) + " live", limit=300),
        fetch_x(team_terms),
    ]
    merged = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else _empty()
    if merged.empty:
        return merged
    cleaned = clean_reactions(merged, team_terms)
    if cleaned.empty:
        return cleaned
    return (
        cleaned.sort_values("created_utc")
        .tail(window)
        .reset_index(drop=True)
    )
