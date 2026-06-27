# Supabase mirror (optional, free tier)

The engine works fully without this — committed Parquet in `data/` is the
durable, keyless source of truth. Supabase adds a hosted, queryable Postgres
(SQL editor, REST API, table viewer) that's nice for inspecting and sharing the
data. It's strictly a **mirror**, written best-effort; if the keys are missing
or a push fails, the pipeline is unaffected.

## One-time setup

1. Create a free project at https://supabase.com (free tier is plenty for this
   data volume — a few thousand rows).
2. In the dashboard: **SQL → New query**, paste [`schema.sql`](schema.sql), run it.
   This creates `match_results`, `match_archive`, `engine_state`, `run_status`,
   and `reactions` (the individual cleaned fan reactions per match — stored here
   instead of git, accumulating over the match via upsert).
3. Get your credentials from **Project Settings → API**:
   - `SUPABASE_URL` = the Project URL (e.g. `https://abcd1234.supabase.co`)
   - `SUPABASE_KEY` = the **service_role** key (server-side; it can write).
     Keep it secret — it bypasses row-level security.

## Where to put the keys

- **GitHub Actions** (so the flywheel mirrors automatically): repo
  **Settings → Secrets and variables → Actions → New repository secret**, add
  `SUPABASE_URL` and `SUPABASE_KEY`. The flywheel already forwards them to the
  ingestion step.
- **Local runs**: set them in your shell environment before
  `python src/pipeline.py run`.

> Do **not** put the `service_role` key in Streamlit Cloud secrets — the public
> app only *reads* committed Parquet and never needs write access.

## Verify

```bash
# preview the payloads with no network call:
python -m warehouse --dry-run

# with the env vars set, mirror the committed data files:
python -m warehouse
```

Then open the Supabase **Table editor** (or run `select count(*) from
match_archive;` in the SQL editor) to see the rows.
