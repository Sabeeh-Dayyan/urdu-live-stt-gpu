-- 001: core schema
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS channels (
    id           text PRIMARY KEY,
    name         text NOT NULL DEFAULT '',
    url          text NOT NULL,
    audio_track  jsonb NOT NULL DEFAULT '{}'::jsonb,
    language     text NOT NULL DEFAULT 'ur',
    glossary     text,
    enabled      boolean NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- Range-partitioned by month on start_utc. The partition key must be part of every unique
-- constraint, hence the upsert key (channel_id, segment_id, start_utc).
CREATE TABLE IF NOT EXISTS asr_segments (
    id                uuid NOT NULL DEFAULT gen_random_uuid(),
    channel_id        text NOT NULL,
    segment_id        text NOT NULL,
    start_utc         timestamptz NOT NULL,
    end_utc           timestamptz NOT NULL,
    pass              text NOT NULL CHECK (pass IN ('fast', 'refined')),
    model             text,
    text_raw          text NOT NULL DEFAULT '',
    text_norm         text NOT NULL DEFAULT '',
    words             jsonb NOT NULL DEFAULT '[]'::jsonb,
    avg_logprob       real,
    no_speech_prob    real,
    flags             text[] NOT NULL DEFAULT '{}',
    latency_s         real,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (channel_id, segment_id, start_utc)
) PARTITION BY RANGE (start_utc);

-- Catch-all so inserts never fail if a monthly partition is missing.
CREATE TABLE IF NOT EXISTS asr_segments_default PARTITION OF asr_segments DEFAULT;

CREATE INDEX IF NOT EXISTS asr_segments_channel_start ON asr_segments (channel_id, start_utc);
CREATE INDEX IF NOT EXISTS asr_segments_start_brin ON asr_segments USING brin (start_utc);
CREATE INDEX IF NOT EXISTS asr_segments_text_trgm ON asr_segments USING gin (text_norm gin_trgm_ops);

CREATE TABLE IF NOT EXISTS stream_events (
    id          bigserial PRIMARY KEY,
    channel_id  text NOT NULL,
    type        text NOT NULL,
    at_utc      timestamptz NOT NULL,
    details     jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS stream_events_channel_at ON stream_events (channel_id, at_utc);

-- Create the monthly partition containing ts (idempotent). Rows already sitting in the default
-- partition for that month are moved so the attach does not fail.
CREATE OR REPLACE FUNCTION ensure_asr_partition(ts timestamptz) RETURNS text
LANGUAGE plpgsql AS $$
DECLARE
    m_start timestamptz := date_trunc('month', ts AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
    m_end   timestamptz := m_start + interval '1 month';
    pname   text := 'asr_segments_' || to_char(m_start AT TIME ZONE 'UTC', 'YYYY_MM');
BEGIN
    IF to_regclass(pname) IS NOT NULL THEN
        RETURN pname;
    END IF;
    PERFORM pg_advisory_xact_lock(hashtext(pname));
    IF to_regclass(pname) IS NOT NULL THEN
        RETURN pname;
    END IF;
    EXECUTE format('CREATE TABLE %I (LIKE asr_segments INCLUDING DEFAULTS INCLUDING CONSTRAINTS)', pname);
    EXECUTE format('INSERT INTO %I SELECT * FROM asr_segments_default WHERE start_utc >= %L AND start_utc < %L',
                   pname, m_start, m_end);
    EXECUTE format('DELETE FROM asr_segments_default WHERE start_utc >= %L AND start_utc < %L', m_start, m_end);
    EXECUTE format('ALTER TABLE asr_segments ATTACH PARTITION %I FOR VALUES FROM (%L) TO (%L)',
                   pname, m_start, m_end);
    RETURN pname;
END $$;
