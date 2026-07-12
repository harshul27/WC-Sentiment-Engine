# 🧠 Project Memory Bank: WC Sentiment Arbitrage Engine

## 🎯 System Intent & Core Niche
This system is an autonomous, 100% free-to-run, self-correcting MLOps pipeline. It tracks real-time crowd sentiment spikes across social media (X/YouTube Live Chat streams) and maps them against live pitch event data during the FIFA World Cup. It flags **Market Arbitrage Moments**—instances where public sentiment drastically decouples from actual underlying match efficiency (e.g., extreme crowd panic despite stable match control).

---

## 🏗️ Free Infrastructure Architecture
To stay completely zero-cost, the system strictly implements this stateless serverless configuration:
* **Orchestration:** GitHub Actions (Event-driven & Midnight Cron loops).
* **Storage Engine:** DuckDB (OLAP relational processing engine reading/writing directly to local states).
* **Data Serialization:** Compressed `.parquet` database files saved to the Git tree or Hugging Face Dataset buckets.
* **Frontend UI:** Streamlit Web Application deployed via Streamlit Community Cloud (streaming parquet values directly via public repository Raw URLs).
* **LLM Inference / Listeners:** Free OpenRouter endpoint mappings (Llama 3 / Mistral 7B) or Groq API free allowances.

---

## 🤖 AI Agentic & Mathematical Workflows
The application bypasses heavy framework overheads (like LangChain) and uses direct, vectorized Python Multi-Agent modules:

```text
 [Social Media Stream/Scrapes] ──> [Social Listening Agent] ──> Computed "Crowd Panic Score"
                                                                         │
 [Match Commentary Text RSS]   ──> [Match Progression Agent] ──> Rolling Expected Goals (xG)
                                                                         │
                                                                         ▼
                                   [Arbitrage Selector]      ──> Flags Disconnect Delta ($Arbitrage_{Index}$)
```

### Core Arbitrage Equation
The divergence index is dynamically formulated using:

$Arbitrage_{Index} = |Crowd\ Panic\ Score| \times (1.0 - \Delta xG_{10min})$

Where Crowd Panic Score is bounded between [-1.0, 1.0] and $\Delta xG$ represents the real-time offensive threat stability index.

---

## 📂 File Registry & Token Guardrails

```text
WC-sentiment-Engine/
├── .github/workflows/
│   ├── engine_flywheel.yml    # Daily cron executor & parameter optimization (test-gated)
│   ├── ci.yml                 # Push/PR gate: ruff, bandit, pip-audit, pytest, smoke run
│   └── codeql.yml             # Weekly + PR CodeQL static security analysis
├── src/
│   ├── model.py               # Vectorized sentiment math, agents, & arbitrage calculations
│   ├── emotion.py             # Emotion scoring: hybrid trained model + football lexicon; per-team summary, headline
│   ├── textmodel.py           # Dependency-free (numpy) TF-IDF + linear scorer for the exported models
│   ├── glossary.py            # Plain-language display labels + tooltips + "what am I looking at" guide
│   ├── teams.py               # Per-team reaction attribution (home|away|both|neither) + coverage
│   ├── sources.py             # Multi-source aggregator (Bluesky/Mastodon keyless; Reddit/YouTube/X key-gated); 1000-reaction rolling window + bot/length/relevance/flood cleaning
│   ├── odds.py                # The Odds API bookmaker consensus (key-gated, read-only) + market divergence
│   ├── warehouse.py           # Supabase mirror via PostgREST (key-gated, best-effort; Parquet stays source of truth)
│   ├── publish.py             # Bluesky insight publisher (app-password, manual post only) + keyless engagement read
│   ├── matchstats.py          # ESPN match detail: key+advanced team stats, player stats, leaders, key events, keeper pressure; optional Sofascore momentum
│   ├── consistency.py         # Mood-vs-game consistency: per-team verdict/explanation + clarity score
│   ├── live.py                # ESPN connectors, minute mapping, capture_phase start/stop lifecycle
│   ├── archive.py             # NOT NULL match archive: match_archive + match_results (DuckDB + Parquet mirror)
│   ├── situation.py           # Real-time nearest-centroid match-situation classifier + metrics playbook
│   ├── advanced.py            # soccerdata/FBref team priors (nightly best-effort refresh -> team_priors.parquet)
│   ├── health.py              # Run heartbeat + LIVE/STALE/DEGRADED/NO-DATA freshness (run_status.json)
│   ├── pipeline.py            # Live-first DuckDB ingestion, parsing, Parquet pipeline, corpus self-correction
│   └── app.py                 # Streamlit frontend: live mode (60s auto-refresh), simulator, committed state
├── scripts/
│   └── train_emotion.py       # Offline trainer/CV/benchmark -> exports data/models/*.npz|json (needs requirements-train.txt)
├── tests/
│   ├── conftest.py            # src path setup + offline/no-key isolation fixture
│   ├── test_model.py          # Unit tests: sentiment math, parsing, equation, grid search
│   ├── test_textmodel.py      # Numpy scorer parity, shipped-model sanity, benchmark guard (>=0.80)
│   ├── test_odds.py           # Odds de-vig, consensus, fixture market, divergence, key-gated skip
│   ├── test_live.py           # Offline fixture tests for ESPN/Bluesky parsers & failure paths
│   ├── test_pipeline.py       # Integration tests: DuckDB, Parquet, source selection, corpus self-correction
│   ├── test_health.py         # Heartbeat + freshness-badge classification
│   └── test_app.py            # Streamlit AppTest smoke tests across display modes
├── data/
│   ├── database.duckdb        # Local testing instance (Git-ignored)
│   ├── model_config.json      # Dynamic hyperparameters & log-loss backpropagation history
│   ├── run_status.json        # Engine heartbeat (source/freshness) for the dashboard badge
│   ├── models/                # Committed trained-model artifacts (emotion_model, sentiment_model: npz+json)
│   ├── benchmarks/            # emotion_benchmark.json: lexicon vs trained accuracy/F1/CV record
│   └── state.parquet          # Compressed operational UI dataset source
├── requirements.txt           # Explicitly pinned runtime library weights (CVE-audited)
├── requirements-dev.txt       # Pinned test & security tooling (pytest/ruff/bandit/pip-audit)
├── requirements-train.txt     # Offline training only (scikit-learn) - never runtime/CI
└── CLAUDE.md                  # Context persistence engine (This File)
```

---

## ⛔ Coding Constraints (Do Not Violate)
1. **No Token Bloat:** Do not implement custom HTML/CSS wrappers inside Streamlit UI files.
2. **Resource Integrity:** Every operation calling DuckDB connections must utilize `try...finally` resource closure statements to guarantee zero file locking errors on GitHub Action virtual machines.
3. **No Code Stubs:** Every script must be 100% operational. Never commit files containing `# Insert logic here` placeholders.

---

## ⏳ Project Progress & Evolution Logs
**Status:** Engine fully operational (v0.1.0).

**Completed (2026-06-10):**
1. Workspace shell + `requirements.txt` pinned to verified local versions.
2. `src/model.py` — vectorized lexicon sentiment, commentary event parser, rolling xG stability, arbitrage equation, log-loss grid search, three agent dataclasses, optional OpenRouter/Groq LLM refinement (graceful fallback when no API key).
3. `src/pipeline.py` — deterministic stream simulator (date-seeded), optional live commentary scraping (`COMMENTARY_FEED_URL` + BeautifulSoup), DuckDB table management with `try...finally` closure, zstd Parquet export via DuckDB `COPY`, nightly `optimize` self-correction writing back to `model_config.json`.
4. `src/app.py` — wide-layout Streamlit dashboard: divergence line chart, Live Stream Feed Simulator button with animated ticker, `st.metric` panic badges. Verified via headless boot + `AppTest` (zero exceptions, button path included).
5. `.github/workflows/engine_flywheel.yml` — midnight cron + manual dispatch, runs `pipeline.py all`, commits refreshed `state.parquet`/`model_config.json`.

**Completed (2026-06-10, CI/CD hardening pass):**
6. `tests/` — 24-test suite (unit + integration + Streamlit AppTest), all passing locally; autouse fixture strips LLM keys so tests never make network calls.
7. `.github/workflows/ci.yml` — push/PR gate: ruff lint, bandit static security scan, pip-audit CVE check on pinned deps, pytest, and a full pipeline smoke run. Least-privilege `contents: read`, actions pinned to commit SHAs, `persist-credentials: false`, concurrency cancellation.
8. `.github/workflows/codeql.yml` — CodeQL `security-and-quality` analysis on push/PR/weekly cron.
9. `engine_flywheel.yml` hardened — actions pinned to SHAs, top-level `contents: read` with job-level `write`, concurrency lock, 30-min timeout, and the nightly run now gates on the test suite before touching state.
10. Dependency CVE remediation: pip-audit flagged streamlit 1.46.1 (CVE-2026-33682) and pyarrow 20.0.0 (PYSEC-2026-113); both upgraded and re-pinned (streamlit 1.54.0, pyarrow 23.0.1), full suite re-verified green, audit now clean.

**Completed (2026-06-10, real-time live-match pass):**
11. `src/live.py` — zero-cost, no-API-key live connectors: ESPN public site API (`fifa.world` scoreboard + per-match play-by-play commentary, league overridable via `ESPN_LEAGUE`) for Agent B, and Bluesky public post search for Agent A's real crowd sentiment. All fetchers verified against live endpoints (real 2026 WC fixtures returned; 95 real posts scored through the sentiment agent). Reddit JSON was evaluated and rejected (403 for unauthenticated clients).
12. `pipeline.py` `gather_streams()` — live-first source selection: in-progress ESPN match → `COMMENTARY_FEED_URL` scrape → deterministic simulator. `match_id` prefixes: `ESPN-` / `FEED-` / `SIM-`.
13. `app.py` three modes — **Live match** (fixture picker + `st.fragment(run_every=60)` panel that refetches both streams and recomputes agents every 60s while the page is open, with score/clock/status badges and a crowd-post expander), Simulator, Committed state.
14. `engine_flywheel.yml` dual cron — `*/20 * * * *` live refresh (commit step no-ops when state unchanged, so off-match ticks produce no commits) + `0 0 * * *` nightly test-gated self-correction.
15. Real-data regression fixed: Bluesky mixed-precision timestamps coerced via `pd.to_datetime(utc=True)` in `posts_to_chat`. Suite now 37 tests, all offline/deterministic.

**Real-time latency profile (honest):** *Live mode* (browser open, app fetches ESPN directly) ≈ 60s, bounded by ESPN responsiveness. *Committed-state mode* is NOT 60s end-to-end: it is bounded by the `*/20` flywheel cron **plus** `raw.githubusercontent.com` CDN caching (~5 min), so realistic staleness is ~5–25 min. The dashboard now shows a LIVE/STALE/DEGRADED/NO-DATA badge (see entry 32) so users see the true freshness. Sentiment sources are Bluesky + Mastodon keyless (free X/YouTube firehoses do not exist).

**Deployed (2026-06-10):** Repository live at https://github.com/harshul27/WC-Sentiment-Engine. First production validation complete: CI green on GitHub runners, manual flywheel run passed the test gate, ingested, self-corrected, and autonomously committed `data/state.parquet` + `model_config.json` back to main (`0daaa4e`). Dependabot immediately opened 3 update PRs, each gated by CI — the security loop is functioning.

**Completed (2026-06-11, emotion intelligence & product pass):**
16. `src/emotion.py` — custom vectorized 6-emotion classifier (panic/anger/joy/confidence/despair/surprise) with per-minute distributions, dominant-emotion tracking, mood-volatility index, emotion-derived panic score, and a rule-based takeaway generator that relates crowd emotion to match state in plain language.
17. `src/sources.py` — multi-platform reaction aggregator, unified schema, 200-comment rolling window (`COMMENT_WINDOW`). Keyless: Bluesky + Mastodon. Key-gated (free tiers): Reddit match-thread comments (`REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`), YouTube live chat (`YOUTUBE_API_KEY`, ~105 units/poll). Verified live: 200/200 window filled (116 Mastodon + 84 Bluesky).
18. `src/matchstats.py` — ESPN boxscore (possession/shots/on-target/corners/saves, keyless, cloud-safe, verified vs 2022 final) → weighted match-control index; Sofascore attack momentum via ScraperFC behind `ENABLE_SOFASCORE=1` (Sofascore 403s plain HTTP and blocked botasaurus during testing — best-effort local enrichment only, NOT in requirements.txt).
19. Dashboard v2 — emotion area chart, crowd-mood badges, "What This Means Right Now" takeaway panel, source-labelled reaction window. State schema extended with emo_* shares, dominant_emotion, emotional_volatility, comment_volume.
20. Suite now 64 tests, all offline. First REAL live ingestion achieved during the MEX–RSA opener: `match_id=ESPN-760415`, 115 minutes scored, 2 arbitrage flags.

**Completed (2026-06-11, capture lifecycle & match archive pass):**
21. `live.capture_phase` start/stop filter — pre → live → post-window (15-minute grace after FT to capture the emotional settle, `POST_GRACE`) → frozen (all fetching stops; 180-min `MATCH_MAX_DURATION` guard freezes long-finished fixtures immediately). `current_capture_match` extends pipeline source selection to finished matches still inside their window.
22. Dashboard lifecycle — live panel shows a countdown banner during the post-window, snapshots the final state into session, and when frozen serves the snapshot or the committed archive with zero network fetches.
23. `src/archive.py` — strict schema: `match_archive` (PK match_id+minute, every column NOT NULL with CHECK range constraints on scores/shares/indexes) + `match_results` (fixture metadata, final score = ground truth for the self-correction loop). Idempotent delete+insert upserts; DuckDB tables mirrored to committed `data/match_archive.parquet` / `match_results.parquet` so the corpus survives ephemeral runners. Flywheel commit step now `git add -A data/`.
24. First real archive captured: ESPN-760415 Mexico 2-0 South Africa, 131 validated minute-rows, zero nulls. Suite: 74 tests.

**Completed (2026-06-12, situation classifier & soccerdata pass):**
25. `src/situation.py` — interpretable nearest-centroid classifier running on every scored minute: 7 situations (cruise_control, balanced_contest, panic_divergence, genuine_crisis, late_drama, emotional_chaos, dead_rubber) from 5 weighted features (panic, xG stability, mood volatility, arbitrage index, match phase) with softmax confidence. Each situation carries a playbook: the metrics that matter right now (rendered dynamically in the dashboard) + a one-line product read. Validated on the archived opener (131 minutes classified sensibly).
26. `src/advanced.py` — soccerdata 1.9.0 (FBref `INT-World Cup`) team priors: schedule-derived gf/ga per match, plus xG/possession once FBref publishes mid-tournament team stats. Committed to `data/team_priors.parquet` by a nightly best-effort flywheel step (`continue-on-error`; FBref is slow/cache-lagged and sometimes blocks datacenter IPs — never fetched from the app or 20-min ticks). NOTE: soccerdata requires `rich>=14` while streamlit pins `rich<14`, so soccerdata is deliberately NOT in requirements.txt — it is pip-installed ad hoc inside the nightly workflow step only. Dashboard shows fixture priors context when available.
27. Archive hardening: **fixed ephemeral-runner data-loss bug** (tables are now seeded from the committed Parquet mirrors before every upsert — previously a fresh runner's empty DuckDB would clobber the archive to a single match). Schema extended with `situation`/`situation_confidence` (NOT NULL + CHECK), with automatic migration of pre-classifier archives (defaults 'unknown'/0.0). Both covered by regression tests.
28. State/archive rows now carry the classification; dashboard adds a "Match Situation" panel (situation badge, confidence, dynamic metrics-that-matter row, playbook read, priors note). Suite: 90 tests.

**Completed (2026-06-14, second-order hardening pass):** A `/second-order-thinker` review surfaced criticalities; all resolved:
29. **Self-correction no longer trains on fiction.** `run_optimize` now trains on the committed `match_archive` corpus (real ESPN fixtures only — the simulator is structurally excluded since it is never archived), with guards: it **skips** (true no-op, config untouched) until ≥ `MIN_TRAIN_MINUTES` (180) of real minutes spanning **both** outcome classes accumulate, so one match or an idle night can never dictate the production threshold. New `derive_corpus_truth` derives outcome labels per match from the *forward* rising-edge of archived rolling xG (panic that the pitch never vindicated) — forward-looking, so not circular with the current-minute arbitrage index. Old `derive_overreaction_truth` retained for its unit test. Corpus path derives from `DB_PATH.parent` for test isolation.
30. **No more fake data in production state.** `run_ingest(allow_simulator=...)` + new `NONE-` sentinel in `gather_streams`; the live flywheel tick runs `pipeline.py run --live-only`, so between matches it writes **nothing** (no committed simulator state, no idle commits). Local dev/`all` keep the simulator default.
31. **Push can't silently drop a tick.** Flywheel commit step is now rebase-and-retry (5 attempts, `git pull --rebase -X theirs origin main` keeps our freshly regenerated Parquet) instead of a bare `git push`.
32. **Outages are now visible.** New `src/health.py` writes a `data/run_status.json` heartbeat each persisted run (source/live/fetch_ok/reaction counts/last_run_utc); dashboard `render_status_badge` turns it into LIVE/STALE/DEGRADED/NO-DATA via `health.freshness`, and live mode distinguishes "source unreachable mid-match" from "pre-match/quiet". A broken pipeline can no longer masquerade as a calm crowd.
33. **Emotion model de-biased for a global crowd.** `emotion.py` lexicons extended with high-volume es/pt/fr football terms + reaction emoji; new `scored_share` exposes lexicon coverage (surfaced in the reactions panel) so the English-only bias is measurable.
34. **soccerdata isolated.** Nightly priors step installs soccerdata into a throwaway `.soccerdata-venv` (gitignored), never mutating the streamlit-pinned runner env before the commit step.
35. **Honesty framing.** Dashboard caption clarifies the Arbitrage Index is a **sentiment–pitch divergence proxy**, not a live betting market and not financial advice (a true odds feed is a deliberate non-goal for cost/safety). Suite: **101 tests**, ruff + bandit clean.

**Known follow-ups (require user action / deliberate non-goals):**
- **Git binary bloat:** `*/20` Parquet commits during the tournament grow `main`'s history (binary blobs don't delta-compress). `--live-only` removed idle-commit churn; the full fix is an **orphan `data` branch** (or GH Releases) with Streamlit's `STATE_PARQUET_URL` repointed — deferred because it requires changing the deployed secret (avoided mid-tournament). 
- Reboot Streamlit Cloud app (Python 3.13 + `STATE_PARQUET_URL` secret). Optional secrets: `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`, `YOUTUBE_API_KEY`, `XAI_API_KEY`, `ODDS_API_KEY`.
- Consider learning situation prototypes from the archive once more matchdays accumulate.
- **Longitudinal mispricing:** persist per-minute market-implied probabilities alongside crowd panic so divergence can be *measured over time* (does crowd panic lead/lag market repricing?), upgrading the odds reference from a snapshot to a validated signal.
- Optional accuracy lift: a multilingual word+char union or distilled model could push sentiment past ~52%, but only a transformer would approach 80% (breaks the free/lightweight constraint).

**Completed (2026-06-16, X / Grok source):**
36. `src/sources.py` `fetch_x` — recent X posts via the user's own xAI/Grok key, using the **X Search agent tool** (`POST https://api.x.ai/v1/responses`, model `grok-4.3`, `tools:[{"type":"x_search"}]`; the deprecated `search_parameters` Live Search was retired 2026-01-12). Grok is asked to return found posts as a JSON array; `_extract_response_text`/`_parse_post_json` tolerantly parse the `/responses` payload (both `output_text` and `output[]` message shapes), mapping to the unified `(created_utc, message, source="x")` schema with a now() timestamp fallback. Key-gated on `XAI_API_KEY` (overridable model via `XAI_MODEL`), broad-except → empty like every other connector, wired into `gather_reactions`. Tests: parse path, missing-timestamp fallback, malformed-JSON resilience, key-gated skip. Suite: **104 tests**. NOTE: X has no free public firehose — this bills against the user's personal xAI account, so it stays opt-in.

**Completed (2026-06-17, model accuracy + market reference):**
37. **Emotion model trained & validated against real data.** `scripts/train_emotion.py` (run offline, `requirements-train.txt`) measured the OLD lexicon at **4.3% accuracy** on `dair-ai/emotion` (held-out 2k test) — far below the 80% bar — and trained a lightweight TF-IDF + logistic-regression replacement: **91.2% test accuracy, macro-F1 0.86, stratified 5-fold CV 91.1%±0.5%**. Models are exported to `data/models/*.npz|json` and scored at runtime by `src/textmodel.py` in **pure numpy** (shared analyzers guarantee numpy↔sklearn parity to 4e-16) — **no new runtime dependency**, free deploy preserved. `emotion.py` now uses the trained model as the primary scorer with the lexicon retained as a transparent fallback and as the source for the `confidence` emotion (no public dataset labels it). Full benchmark in `data/benchmarks/emotion_benchmark.json`; a test guards `test_accuracy >= 0.80`.
38. **Multilingual panic coverage.** A char-n-gram sentiment model trained on `cardiffnlp/tweet_sentiment_multilingual` (8 languages: en/es/pt/fr/it/de/hi/ar) drives the crowd-panic *direction*, blended with the emotion model. Lexicon multilingual sentiment was ~33% (chance); the trained model is ~52% across 8 languages (a hard 3-class tweet task — lightweight ML ceiling; 80% there needs a transformer, deliberately avoided). This genuinely moves the panic score for non-English posts, closing the "partial crowd signal" gap; honest per-language numbers are recorded in the benchmark.
39. **Market reference (odds).** `src/odds.py` — key-gated (`ODDS_API_KEY`) read-only **The Odds API** consensus: de-vigs bookmaker decimal odds into implied home/draw/away probabilities, derives a market `certainty`, and computes a `sentiment_market_divergence` (|crowd panic| × certainty). Dashboard shows a "Market vs Crowd" panel in live mode when a key is present. Strictly read-only — never wagers, not betting advice. Closes the "no market to compare against" gap with a real consensus reference. **Honest limit:** our crowd signal is aggregate (not team-aligned) and this is a single-snapshot divergence; true longitudinal mispricing measurement (storing market probs per minute and correlating with panic) is the documented next step. Suite: **118 tests**, ruff + bandit clean.

**Completed (2026-06-25, showcase-ready pass — plain language, per-team, validation, Supabase):**
40. **Plain-language UI.** New `src/glossary.py` is the single source of display labels + tooltips; dashboard drops "Panic Score"/"Arbitrage" for **Fan Mood / Attacking Threat / Hype-vs-Reality Gap / Overreaction Moments / Mood-vs-Odds Gap**, retitled "World Cup Crowd Mood Engine", added a "❓ What am I looking at?" guide and `help=` tooltips on every metric. Internal column names unchanged (model/archive don't churn).
41. **Per-team emotion attribution.** New `src/teams.py` `tag_reactions` labels each reaction home|away|both|neither (team-name tokens + alias map); `live.live_streams` adds a `team` column; `emotion.team_emotion_summary` produces per-team dominant mood + volume + reading coverage; dashboard "👥 Mood by Team" panel ("🏠 {home}: anxious | 🛫 {away}: joyful") so a singular emotion is never ambiguous. Live-mode feature (committed/sim are pooled).
42. **Domain-hybrid emotion fix (important).** The 91% model is trained on *general* tweets and mislabelled football phrasing ("we are done", "disgrace") as **joy**. `emotion.model_comment_scores` now blends the trained model with the football+emoji lexicon, weighting the lexicon by how strongly it fires (`_LEX_HALF`/`_LEX_MAX`) — fixes the embarrassing cases while keeping multilingual reach. Regression test `test_hybrid_corrects_football_phrasing` locks panic/anger/joy/confidence/despair on real football phrases.
43. **Validation visible.** `render_validation` surfaces `data/benchmarks/emotion_benchmark.json` in-app (91% vs 4%, CV±std, per-language sentiment, datasets, numpy parity) + live reading-coverage, and is honest that football phrasing is lexicon-anchored.
44. **Per-team live stats + real-time outcome.** `render_match_stats` shows a clear per-team ESPN boxscore table (possession/shots/on-target/corners/saves/fouls) + control; Sofascore attack-momentum chart when local (cloud IPs blocked — stated in-UI). `emotion.headline_outcome` renders one plain "Right now: …" sentence at the top of every mode.
45. **Supabase data platform.** New `src/warehouse.py` mirrors `engine_state`/`match_archive`/`match_results`/`run_status` into hosted Supabase Postgres via PostgREST upsert using **plain requests** (no new dependency), key-gated (`SUPABASE_URL`/`SUPABASE_KEY`), best-effort — committed Parquet stays the keyless source of truth. `pipeline.run_ingest` syncs after each run when enabled; `supabase/schema.sql` + `supabase/README.md` for one-time setup; `python -m warehouse --dry-run` previews payloads. Flywheel now forwards `XAI_API_KEY`/`ODDS_API_KEY`/`REDDIT_*`/`YOUTUBE_API_KEY`/`SUPABASE_URL`/`SUPABASE_KEY` to the ingestion step (also fixes the earlier "flywheel doesn't forward X key" gap). New secrets: `SUPABASE_URL`, `SUPABASE_KEY` (service_role; GitHub Actions + local only — never Streamlit). Suite: **131 tests**, ruff + bandit clean.

**Completed (2026-06-26, larger + cleaner reaction window):**
46. **1,000-reaction rolling window + quality cleaning.** `COMMENT_WINDOW` 200→**1000**. `sources.py` adds an `author` field to the reaction schema (captured from Bluesky/Mastodon/Reddit/YouTube where available) and `clean_reactions` — drops automod/chat bots (`_BOT_AUTHORS` + `*bot` handles), emoji/link/mention-only or <3-word posts, off-topic chatter (lenient team-name **or** football-keyword relevance gate), near-duplicate copypasta (`_normalise` collapses repeated chars so "goooal"=="gooal"), and per-author floods (`MAX_PER_AUTHOR=5`). Per-source fetch limits raised modestly (Bluesky 100, Reddit 150, YouTube 300). New `merge_window` accumulates reactions across refreshes (union → de-dup → newest `window`).
47. **Rolling buffer in live mode.** `live.gather_live_reactions` returns cleaned raw reactions tagged by team; `live.live_streams_buffered(match, prior, window)` merges each 60s refresh into a session-state buffer per fixture so the crowd window *grows over the match* (up to 1000) instead of resetting each tick; `posts_to_chat` now carries `source`/`team`/`author`. `app.py` live panel keeps `st.session_state["reaction_buffer"][event_id]`; reactions expander notes the rolling window + per-team tag counts. The stateless pipeline keeps single-fetch `live_streams` (unchanged contract). Suite: **135 tests**, ruff + bandit clean.
**Completed (2026-07-11, Publish Insights tab):**
52. **`src/publish.py` + "📤 Publish Insights" tab.** Drafts an honest, clearly-labelled (`🤖 automated analytics`) insight post from the latest state (`draft_post`, ≤290 char Bluesky limit), posts to the user's own account **only on button click** via app-password auth (`post_insight`: createSession→createRecord; key-gated on BLUESKY_HANDLE + BLUESKY_APP_PASSWORD), and reads organic engagement (likes/reposts/replies) on recent posts via the keyless public AppView (`recent_engagement`). **Deliberately NOT an influence tool** — the user's original ask (posts crafted to "influence crowd mood" + measuring the manipulation + SEO-injection into the most active fan threads) was declined twice; this is transparent data-sharing with descriptive reach only and standard fixture hashtags (also methodologically: live crowd mood is driven by the match, not by posts, so manipulation-attribution would be spurious).
53. **Underdog-case trigger (the accepted reframe).** `publish.underdog_case(match_row, state, match_stats, keeper, market)` — when an overreaction moment fires in live mode, drafts a data-backed case for the side least likely to win (trailing side, or market's least-likely when level; **returns None when the data offers no genuine support — belief is never fabricated**). Evidence cited only from real numbers: shots on target, live threat ≥0.3, opponent keeper ≥3 saves. `match_hashtags` from ESPN short_name ("FRA @ MAR" → `#MARFRA #FIFAWorldCup`); hashtags + `🤖` label always survive truncation (evidence line absorbs the cut). Draft queues to the Publish tab; optional **opt-in** sidebar toggle auto-posts, rate-limited via `should_autopost` (1 per 15 min). New secrets: `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`. Suite: **163 tests**, ruff + bandit clean.

**Completed (2026-07-09, game-aware mood + player stats pass):**
49. **Mood-vs-game consistency (`src/consistency.py`).** Each team's crowd mood is now checked against the live game situation: `game_context` (scoreline→leading/trailing/level + threat + keeper workload) → `mood_consistency` → per-team verdict `consistent|conflict` with a plain explanation, and a `clarity` score in [0,1] (0.5·emotion-share margin + 0.3·reading coverage + 0.2·volume, `VOLUME_FULL=30`). Rules: trailing+joy/confidence → **conflict**; leading+panic with idle keeper (≤2 saves) → **conflict**; leading+panic with busy keeper → consistent (lead under pressure); trailing+panic/despair → consistent. Conflicts are flagged and explained, never suppressed (consolation-goal joy is legitimate — the explanation says so). Dashboard: Mood by Team cards gain a **Mood Clarity** metric + conflict warning chips.
50. **Overreaction Moments explicitly defined + two-sided.** UI states the definition verbatim ("minutes where fan mood conflicts with the game situation — panic/anger while stable, or celebration/confidence while losing and creating nothing"; in glossary GUIDE + flags panel). Live mode promotes current mood-vs-game conflicts into the moments panel with reasons (`positive-while-losing`, `panic-while-ahead`) alongside the flagged minutes (reason: panic-vs-stable). Archive schema unchanged (reasons are display-level; persisting to Supabase = noted follow-up).
51. **Player + advanced team stats from the same ESPN request.** `matchstats.fetch_match_detail(event_id)` (ONE summary GET) → `stats` (KEY_STATS + new ADVANCED_STATS tier: accuratePasses/passPct/tackles/interceptions/clearances/blockedShots/crosses), `players` (per-player PLAYER_STATS from `rosters[]`; verified live: 23-26 players/team), `leaders` (top shots/passes/defensiveInterventions/saves), `key_events` (typed, **clock value is SECONDS** → minute = value//60; verified 2700s=45'). Derived: `keeper_pressure` (per-team GK saves/shots-faced/conceded), `top_performers`, `goal_scorers`. Dashboard: "⭐ Key Performers" strip + advanced-stats expander; `generate_takeaways` gains a keeper-context rule ("fans anxious but keeper has N saves"); keeper workload feeds the consistency verdicts. Suite: **150 tests**, ruff + bandit clean.

48. **Reactions persisted to Supabase (not git).** New `reactions` table (PK `match_id`+`message_hash` = first 24 hex of sha256(message)); `warehouse.push_reactions(chat, match_id)` upserts the cleaned per-reaction rows (minute/source/team/author/message) so the full per-match set accumulates across flywheel runs in Supabase without bloating the repo. `pipeline.run_ingest` pushes them for **real fixtures only** (skips simulator/feed) when Supabase is enabled. The live UI's rolling 1000 stays in session memory; this is the durable store. Suite: **137 tests**, ruff + bandit clean. NOTE on Twitch (evaluated, deferred): Helix is free but exposes **no chat** — reactions need an IRC/EventSub streaming listener that breaks the stateless model, and Twitch chat is the noisiest/least-representative source; Reddit+YouTube (already integrated, key-gated) are the high-value free adds.
