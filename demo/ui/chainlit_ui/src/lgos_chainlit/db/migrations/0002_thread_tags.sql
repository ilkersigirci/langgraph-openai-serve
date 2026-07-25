-- Official Chainlit data-layer migration 20250108095538_add_tags_to_thread.

ALTER TABLE "Thread"
    ADD COLUMN "tags" TEXT[] DEFAULT ARRAY[]::TEXT[];
