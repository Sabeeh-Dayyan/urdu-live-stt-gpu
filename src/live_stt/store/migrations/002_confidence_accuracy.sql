-- Segment confidence (mean word probability, 0–1; model confidence, not accuracy) for filtering
-- and dashboards, plus measured accuracy results (bench / calibration against references).
ALTER TABLE asr_segments ADD COLUMN IF NOT EXISTS confidence real;

UPDATE asr_segments s SET confidence = c.v
FROM (SELECT channel_id, segment_id, start_utc, avg((w->>'p')::real) AS v
      FROM asr_segments, jsonb_array_elements(words) w
      WHERE jsonb_typeof(words) = 'array' AND (w->>'p') IS NOT NULL
      GROUP BY channel_id, segment_id, start_utc) c
WHERE s.channel_id = c.channel_id AND s.segment_id = c.segment_id AND s.start_utc = c.start_utc
  AND s.confidence IS NULL;

CREATE TABLE IF NOT EXISTS accuracy_results (
    id          bigserial PRIMARY KEY,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    source_file text NOT NULL,           -- result JSON it was loaded from
    dataset     text NOT NULL,           -- e.g. fleurs_ur, broadcast ch1_a
    kind        text NOT NULL,           -- bench | calibration_word | calibration_segment
    config      text NOT NULL,           -- model/decoder label
    metric      text NOT NULL,           -- wer, cer, rtf, precision, accuracy, kept_share
    threshold   real,                    -- confidence threshold (calibration rows)
    value       double precision,
    n           integer,                 -- words or segments behind the value
    UNIQUE (source_file, dataset, kind, config, metric, threshold)
);
