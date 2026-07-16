-- Required by the official Chainlit 2.1.0 migration guide:
-- https://docs.chainlit.io/guides/migration/2.1.0

ALTER TABLE "Step"
    ADD COLUMN IF NOT EXISTS "command" TEXT;
