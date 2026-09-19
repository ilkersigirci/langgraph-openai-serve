CREATE TABLE IF NOT EXISTS lgos_background_responses (
    run_id text PRIMARY KEY,
    response_id text NOT NULL UNIQUE,
    owner_scope text NOT NULL,
    model text NOT NULL,
    idempotency_key uuid,
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
    recovery_cleaned boolean NOT NULL,
    updated_at timestamptz NOT NULL,
    version bigint NOT NULL,
    record jsonb NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS lgos_background_idempotency_key
    ON lgos_background_responses (owner_scope, model, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS lgos_background_active
    ON lgos_background_responses (status)
    WHERE status IN ('queued', 'in_progress');

CREATE INDEX IF NOT EXISTS lgos_background_cancellation
    ON lgos_background_responses (updated_at, run_id)
    WHERE cancellation_pending;

CREATE INDEX IF NOT EXISTS lgos_background_cleanup
    ON lgos_background_responses (updated_at, run_id)
    WHERE cleanup_pending AND NOT recovery_cleaned;

CREATE INDEX IF NOT EXISTS lgos_background_expiry
    ON lgos_background_responses (result_expires_at)
    WHERE terminal_at IS NOT NULL;
