-- supabase_schema.sql
-- Schema for SHAJARA collectors + Streamlit app

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS public.posts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    platform TEXT NOT NULL DEFAULT 'telegram',
    source_name TEXT NOT NULL,
    source_id TEXT,
    datetime_utc TIMESTAMPTZ NOT NULL,
    author TEXT,
    text TEXT,
    likes INTEGER NOT NULL DEFAULT 0,
    shares INTEGER NOT NULL DEFAULT 0,
    comments INTEGER NOT NULL DEFAULT 0,
    raw JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_posts_source_datetime
    ON public.posts (source_name, datetime_utc DESC);

CREATE INDEX IF NOT EXISTS idx_posts_datetime
    ON public.posts (datetime_utc DESC);

CREATE INDEX IF NOT EXISTS idx_posts_text_trgm
    ON public.posts USING GIN (text gin_trgm_ops);
