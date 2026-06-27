-- WC Crowd Mood Engine — Supabase schema
-- Run this once in the Supabase SQL editor (Dashboard → SQL → New query).
-- The engine upserts into these tables on each run via PostgREST
-- (Prefer: resolution=merge-duplicates), keyed on the primary keys below.
-- The committed Parquet files remain the keyless source of truth; Supabase is
-- a queryable mirror, so these tables can be dropped/recreated safely.

create table if not exists match_results (
    match_id     text primary key,
    home_team    text,
    away_team    text,
    kickoff_utc  timestamptz,
    final_score  text,
    state        text,
    archived_at  timestamptz
);

create table if not exists match_archive (
    match_id             text not null,
    minute               integer not null,
    crowd_panic_score    double precision,
    emo_panic            double precision,
    emo_anger            double precision,
    emo_joy              double precision,
    emo_confidence       double precision,
    emo_despair          double precision,
    emo_surprise         double precision,
    dominant_emotion     text,
    emotional_volatility double precision,
    comment_volume       integer,
    rolling_xg           double precision,
    delta_xg_10min       double precision,
    arbitrage_index      double precision,
    flagged              boolean,
    situation            text,
    situation_confidence double precision,
    archived_at          timestamptz,
    primary key (match_id, minute)
);

-- Latest scored state shown on the dashboard (same shape as the archive minus
-- archived_at; keyed on match_id+minute so re-runs upsert cleanly).
create table if not exists engine_state (
    match_id             text not null,
    minute               integer not null,
    crowd_panic_score    double precision,
    emo_panic            double precision,
    emo_anger            double precision,
    emo_joy              double precision,
    emo_confidence       double precision,
    emo_despair          double precision,
    emo_surprise         double precision,
    comment_volume       integer,
    dominant_emotion     text,
    emotional_volatility double precision,
    rolling_xg           double precision,
    delta_xg_10min       double precision,
    arbitrage_index      double precision,
    flagged              boolean,
    situation            text,
    situation_confidence double precision,
    primary key (match_id, minute)
);

create table if not exists run_status (
    match_id      text primary key,
    last_run_utc  timestamptz,
    source        text,
    live          boolean,
    n_reactions   integer,
    n_commentary  integer,
    fetch_ok      boolean
);

-- Individual cleaned fan reactions per match. Keyed on (match_id,
-- message_hash) so the same reaction seen on a later flywheel run upserts
-- rather than duplicating; the set accumulates over the match here instead of
-- in git. message_hash = first 24 hex chars of sha256(message).
create table if not exists reactions (
    match_id     text not null,
    message_hash text not null,
    minute       integer,
    message      text,
    source       text,
    team         text,
    author       text,
    primary key (match_id, message_hash)
);
