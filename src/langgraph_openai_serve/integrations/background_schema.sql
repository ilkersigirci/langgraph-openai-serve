-- Background Response store schema.
CREATE TABLE IF NOT EXISTS lgos_background_responses (
    response_id text PRIMARY KEY,
    owner_scope text NOT NULL,
    model text NOT NULL,
    idempotency_digest varchar(64),
    request_fingerprint text NOT NULL,
    status text NOT NULL CHECK (
        status IN (
            'queued',
            'in_progress',
            'completed',
            'incomplete',
            'failed',
            'cancelled'
        )
    ),
    workflow_run_id text,
    terminal_at timestamptz,
    result_expires_at timestamptz,
    idempotency_expires_at timestamptz,
    cancellation_pending boolean NOT NULL,
    cleanup_pending boolean NOT NULL,
    updated_at timestamptz NOT NULL,
    record jsonb NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS lgos_background_idempotency_digest
    ON lgos_background_responses (idempotency_digest)
    WHERE idempotency_digest IS NOT NULL;

CREATE INDEX IF NOT EXISTS lgos_background_active
    ON lgos_background_responses (status)
    WHERE status IN ('queued', 'in_progress');

CREATE INDEX IF NOT EXISTS lgos_background_cancellation
    ON lgos_background_responses (updated_at, response_id)
    WHERE cancellation_pending;

CREATE INDEX IF NOT EXISTS lgos_background_cleanup
    ON lgos_background_responses (updated_at, response_id)
    WHERE cleanup_pending;

CREATE INDEX IF NOT EXISTS lgos_background_expiry
    ON lgos_background_responses (result_expires_at)
    WHERE terminal_at IS NOT NULL;
