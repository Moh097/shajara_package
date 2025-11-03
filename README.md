# SHAJARA — Deployment Package (GitHub Actions + Collectors + Supabase)

This package contains the ready-to-upload files and step-by-step instructions to recreate the environment (GitHub repo, Actions workflow, Supabase schema, and collectors) after your GitHub account reset.

## Contents
- `collectors/telegram_collector.py` — Telegram collector (Telethon) ready to run on GitHub Actions.
- `collectors/facebook_collector.py` — Facebook collector (facebook-scraper) with cookie support and diagnostics.
- `.github/workflows/collectors.yml` — GitHub Actions workflow (daily schedule, secrets validation, debug find).
- `utils/supabase_client.py` — Simple Supabase REST helper used by collectors.
- `collectors/__init__.py`, `utils/__init__.py`
- `gen_session.py` — Local helper to generate TELEGRAM_STRING_SESSION.
- `supabase_schema.sql` — SQL to create `posts` table and indexes.
- `README.md` — this file.
