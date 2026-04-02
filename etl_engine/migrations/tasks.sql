-- Task queue table for the pull-based worker pipeline.
--
-- SELECT … FOR UPDATE SKIP LOCKED lets multiple workers poll the same
-- table concurrently without ever claiming the same row twice.

CREATE TABLE IF NOT EXISTS tasks (
    id          SERIAL PRIMARY KEY,
    item_id     TEXT        NOT NULL,
    stage       TEXT        NOT NULL CHECK (stage IN ('download', 'predict', 'postprocess')),
    status      TEXT        NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'in_progress', 'done', 'failed')),
    payload     JSONB       NOT NULL DEFAULT '{}',
    result      JSONB,
    error       TEXT,
    worker_id   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

-- Fast claim queries: workers filter by (stage, status)
CREATE INDEX IF NOT EXISTS idx_tasks_stage_status
    ON tasks (stage, status, created_at ASC);

-- Trigger to keep updated_at current
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$;

DROP TRIGGER IF EXISTS tasks_updated_at ON tasks;
CREATE TRIGGER tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
