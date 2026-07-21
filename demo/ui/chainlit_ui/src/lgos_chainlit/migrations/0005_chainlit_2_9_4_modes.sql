-- Required by the official Chainlit 2.9.4 migration guide:
-- https://docs.chainlit.io/guides/migration/2.9.4

ALTER TABLE "Step"
    ADD COLUMN IF NOT EXISTS "modes" JSONB;
