-- 003: uploaded media files transcribed by the `uploads` process. Their transcripts live in
-- asr_segments under channel_id = uploads.id ("up-…"); base_utc = media time 0.
CREATE TABLE IF NOT EXISTS uploads (
    id             text PRIMARY KEY,
    filename       text NOT NULL,
    size_bytes     bigint NOT NULL DEFAULT 0,
    language       text NOT NULL DEFAULT 'ur',
    status         text NOT NULL DEFAULT 'queued'
                   CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    error          text,
    source_file    text NOT NULL,          -- relative to uploads.dir
    media_file     text,                   -- browser-playable copy (relative), if any
    has_video      boolean NOT NULL DEFAULT false,
    duration_s     double precision,
    decoded_s      double precision NOT NULL DEFAULT 0,
    segments_sent  integer NOT NULL DEFAULT 0,
    segments_done  integer NOT NULL DEFAULT 0,
    base_utc       timestamptz NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    started_at     timestamptz,
    finished_at    timestamptz
);
CREATE INDEX IF NOT EXISTS uploads_status ON uploads (status, created_at);
