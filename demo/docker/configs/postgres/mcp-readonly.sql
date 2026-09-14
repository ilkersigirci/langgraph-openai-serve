\set ON_ERROR_STOP on
\getenv mcp_password LGOS_MCP_DB_PASSWORD

BEGIN;

SELECT format(
  'CREATE ROLE lgos_mcp LOGIN PASSWORD %L',
  :'mcp_password'
)
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'lgos_mcp')
\gexec

SELECT format('ALTER ROLE lgos_mcp PASSWORD %L', :'mcp_password')
\gexec

ALTER ROLE lgos_mcp
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
GRANT CONNECT ON DATABASE lgos TO lgos_mcp;
SELECT 'REVOKE pg_read_all_data FROM lgos_mcp'
WHERE pg_has_role('lgos_mcp', 'pg_read_all_data', 'member')
\gexec

-- Keep the reporting login confined even if the source schemas gain objects.
REVOKE ALL PRIVILEGES ON SCHEMA public FROM lgos_mcp;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM lgos_mcp;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM lgos_mcp;
ALTER DEFAULT PRIVILEGES FOR ROLE lgos IN SCHEMA public
  REVOKE SELECT ON TABLES FROM lgos_mcp;
ALTER DEFAULT PRIVILEGES FOR ROLE lgos IN SCHEMA public
  REVOKE SELECT ON SEQUENCES FROM lgos_mcp;

CREATE SCHEMA IF NOT EXISTS mcp_demo AUTHORIZATION lgos;

REVOKE ALL PRIVILEGES ON SCHEMA mcp_demo FROM lgos_mcp;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA mcp_demo FROM lgos_mcp;

DROP VIEW IF EXISTS mcp_demo.chainlit_activity_summary;
DROP VIEW IF EXISTS mcp_demo.chainlit_profile_activity;
DROP VIEW IF EXISTS mcp_demo.lgos_interrupted_runs;
DROP VIEW IF EXISTS mcp_demo.chainlit_user_conversation_counts;

CREATE VIEW mcp_demo.chainlit_user_conversation_counts
WITH (security_barrier = true, security_invoker = false) AS
SELECT
  chainlit_user."identifier" AS user_identifier,
  chainlit_user."createdAt" AS user_created_at,
  count(conversation."id") FILTER (
    WHERE conversation."deletedAt" IS NULL
  ) AS current_conversation_count,
  count(conversation."id") FILTER (
    WHERE conversation."deletedAt" IS NOT NULL
  ) AS deleted_conversation_count,
  count(conversation."id") AS total_conversation_count,
  max(conversation."createdAt") FILTER (
    WHERE conversation."deletedAt" IS NULL
  ) AS last_current_conversation_at
FROM public."User" AS chainlit_user
LEFT JOIN public."Thread" AS conversation
  ON conversation."userId" = chainlit_user."id"
GROUP BY
  chainlit_user."id",
  chainlit_user."identifier",
  chainlit_user."createdAt";

CREATE VIEW mcp_demo.chainlit_profile_activity
WITH (security_barrier = true, security_invoker = false) AS
WITH conversation_steps AS (
  SELECT
    "threadId" AS conversation_id,
    count(*) FILTER (
      WHERE "type"::text = 'user_message'
    ) AS user_message_count,
    count(*) FILTER (
      WHERE "type"::text = 'assistant_message'
    ) AS assistant_message_count,
    count(*) FILTER (
      WHERE "isError"
    ) AS error_step_count,
    max("createdAt") AS last_step_at
  FROM public."Step"
  GROUP BY "threadId"
)
SELECT
  coalesce(
    nullif(btrim(conversation."metadata" ->> 'chat_profile'), ''),
    'unknown'
  ) AS chat_profile,
  count(DISTINCT conversation."userId") FILTER (
    WHERE conversation."deletedAt" IS NULL
  ) AS current_user_count,
  count(*) FILTER (
    WHERE conversation."deletedAt" IS NULL
  ) AS current_conversation_count,
  count(*) FILTER (
    WHERE conversation."deletedAt" IS NOT NULL
  ) AS deleted_conversation_count,
  count(*) FILTER (
    WHERE conversation."deletedAt" IS NULL
      AND coalesce(conversation_steps.user_message_count, 0) > 0
  ) AS active_conversation_count,
  count(*) FILTER (
    WHERE conversation."deletedAt" IS NULL
      AND coalesce(conversation_steps.user_message_count, 0) = 0
  ) AS empty_conversation_count,
  coalesce(sum(conversation_steps.user_message_count) FILTER (
    WHERE conversation."deletedAt" IS NULL
  ), 0)::bigint AS user_message_count,
  coalesce(sum(conversation_steps.assistant_message_count) FILTER (
    WHERE conversation."deletedAt" IS NULL
  ), 0)::bigint AS assistant_message_count,
  count(*) FILTER (
    WHERE conversation."deletedAt" IS NULL
      AND coalesce(conversation_steps.error_step_count, 0) > 0
  ) AS conversation_with_errors_count,
  coalesce(sum(conversation_steps.error_step_count) FILTER (
    WHERE conversation."deletedAt" IS NULL
  ), 0)::bigint AS error_step_count,
  max(greatest(
    conversation."createdAt",
    conversation_steps.last_step_at
  )) FILTER (
    WHERE conversation."deletedAt" IS NULL
  ) AS last_activity_at
FROM public."Thread" AS conversation
LEFT JOIN conversation_steps
  ON conversation_steps.conversation_id = conversation."id"
GROUP BY coalesce(
  nullif(btrim(conversation."metadata" ->> 'chat_profile'), ''),
  'unknown'
);

CREATE VIEW mcp_demo.chainlit_activity_summary
WITH (security_barrier = true, security_invoker = false) AS
SELECT
  count(*) FILTER (
    WHERE current_conversation_count > 0
  ) AS current_profile_count,
  coalesce(sum(current_conversation_count), 0)::bigint
    AS current_conversation_count,
  coalesce(sum(deleted_conversation_count), 0)::bigint
    AS deleted_conversation_count,
  coalesce(sum(active_conversation_count), 0)::bigint
    AS active_conversation_count,
  coalesce(sum(empty_conversation_count), 0)::bigint
    AS empty_conversation_count,
  coalesce(sum(user_message_count), 0)::bigint AS user_message_count,
  coalesce(sum(assistant_message_count), 0)::bigint AS assistant_message_count,
  coalesce(sum(conversation_with_errors_count), 0)::bigint
    AS conversation_with_errors_count,
  coalesce(sum(error_step_count), 0)::bigint AS error_step_count,
  max(last_activity_at) AS last_activity_at
FROM mcp_demo.chainlit_profile_activity;

CREATE VIEW mcp_demo.lgos_interrupted_runs
WITH (security_barrier = true, security_invoker = false) AS
WITH latest_heads AS (
  SELECT DISTINCT ON (thread_id, checkpoint_ns)
    thread_id,
    checkpoint_ns,
    checkpoint_id,
    checkpoint
  FROM public.checkpoints
  ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
),
pending_heads AS (
  SELECT
    head.thread_id,
    head.checkpoint_ns,
    (head.checkpoint ->> 'ts')::timestamptz AS interrupted_at
  FROM latest_heads AS head
  WHERE EXISTS (
    SELECT 1
    FROM public.checkpoint_writes AS interrupt_write
    WHERE interrupt_write.thread_id = head.thread_id
      AND interrupt_write.checkpoint_ns = head.checkpoint_ns
      AND interrupt_write.checkpoint_id = head.checkpoint_id
      AND interrupt_write.channel = '__interrupt__'
      AND NOT EXISTS (
        SELECT 1
        FROM public.checkpoint_writes AS resume_write
        WHERE resume_write.thread_id = interrupt_write.thread_id
          AND resume_write.checkpoint_ns = interrupt_write.checkpoint_ns
          AND resume_write.checkpoint_id = interrupt_write.checkpoint_id
          AND resume_write.task_id = interrupt_write.task_id
          AND resume_write.channel = '__resume__'
      )
  )
),
checkpoint_history AS (
  SELECT
    thread_id,
    count(*) AS checkpoint_count,
    min((checkpoint ->> 'ts')::timestamptz) AS first_checkpoint_at
  FROM public.checkpoints
  GROUP BY thread_id
),
run_metadata AS (
  SELECT
    thread_id,
    max(metadata ->> 'lgos.model') FILTER (
      WHERE jsonb_typeof(metadata -> 'lgos.model') = 'string'
    ) AS model_name,
    max(metadata ->> 'lgos.operation_id') FILTER (
      WHERE jsonb_typeof(metadata -> 'lgos.operation_id') = 'string'
    ) AS run_id
  FROM public.checkpoints
  GROUP BY thread_id
)
SELECT
  pending.thread_id AS checkpoint_thread_id,
  run_metadata.model_name,
  run_metadata.run_id,
  checkpoint_history.first_checkpoint_at,
  max(pending.interrupted_at) AS interrupted_at,
  checkpoint_history.checkpoint_count,
  count(*) AS pending_checkpoint_namespaces
FROM pending_heads AS pending
JOIN checkpoint_history USING (thread_id)
LEFT JOIN run_metadata USING (thread_id)
GROUP BY
  pending.thread_id,
  run_metadata.model_name,
  run_metadata.run_id,
  checkpoint_history.first_checkpoint_at,
  checkpoint_history.checkpoint_count;

COMMENT ON SCHEMA mcp_demo IS
  'Read-only reporting views over live LGOS and Chainlit data';
COMMENT ON VIEW mcp_demo.chainlit_user_conversation_counts IS
  'Current, deleted, and total Chainlit conversation counts per user';
COMMENT ON VIEW mcp_demo.chainlit_profile_activity IS
  'Content-free Chainlit conversation activity aggregated by chat profile';
COMMENT ON VIEW mcp_demo.chainlit_activity_summary IS
  'Content-free aggregate of current Chainlit conversation activity';
COMMENT ON VIEW mcp_demo.lgos_interrupted_runs IS
  'LGOS checkpoint threads whose latest state contains pending interrupts';

GRANT USAGE ON SCHEMA mcp_demo TO lgos_mcp;
GRANT SELECT ON
  mcp_demo.chainlit_user_conversation_counts,
  mcp_demo.chainlit_profile_activity,
  mcp_demo.chainlit_activity_summary,
  mcp_demo.lgos_interrupted_runs
TO lgos_mcp;
ALTER ROLE lgos_mcp IN DATABASE lgos SET default_transaction_read_only = on;

COMMIT;
