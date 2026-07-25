-- Based on the official Chainlit PostgreSQL data-layer migration at commit
-- 8844b1b318112961f8589b0614474a85530995df.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TYPE "StepType" AS ENUM (
    'assistant_message',
    'embedding',
    'llm',
    'retrieval',
    'rerank',
    'run',
    'system_message',
    'tool',
    'undefined',
    'user_message'
);

CREATE TABLE "Element" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "threadId" TEXT,
    "stepId" TEXT NOT NULL,
    "metadata" JSONB NOT NULL,
    "mime" TEXT,
    "name" TEXT NOT NULL,
    "objectKey" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "display" TEXT,
    "size" TEXT,
    "language" TEXT,
    "page" INTEGER,
    "props" JSONB,
    CONSTRAINT "Element_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "User" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "metadata" JSONB NOT NULL,
    "identifier" TEXT NOT NULL,
    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "Feedback" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "stepId" TEXT,
    "name" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "comment" TEXT,
    CONSTRAINT "Feedback_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "Step" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "parentId" TEXT,
    "threadId" TEXT,
    "input" TEXT,
    "metadata" JSONB NOT NULL,
    "name" TEXT,
    "output" TEXT,
    "type" "StepType" NOT NULL,
    "showInput" TEXT DEFAULT 'json',
    "isError" BOOLEAN DEFAULT false,
    "startTime" TIMESTAMP(3) NOT NULL,
    "endTime" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "Step_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "Thread" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "deletedAt" TIMESTAMP(3),
    "name" TEXT,
    "metadata" JSONB NOT NULL,
    "userId" TEXT,
    CONSTRAINT "Thread_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "Element_stepId_idx" ON "Element"("stepId");
CREATE INDEX "Element_threadId_idx" ON "Element"("threadId");
CREATE INDEX "User_identifier_idx" ON "User"("identifier");
CREATE UNIQUE INDEX "User_identifier_key" ON "User"("identifier");
CREATE INDEX "Feedback_createdAt_idx" ON "Feedback"("createdAt");
CREATE INDEX "Feedback_name_idx" ON "Feedback"("name");
CREATE INDEX "Feedback_stepId_idx" ON "Feedback"("stepId");
CREATE INDEX "Feedback_value_idx" ON "Feedback"("value");
CREATE INDEX "Feedback_name_value_idx" ON "Feedback"("name", "value");
CREATE INDEX "Step_createdAt_idx" ON "Step"("createdAt");
CREATE INDEX "Step_endTime_idx" ON "Step"("endTime");
CREATE INDEX "Step_parentId_idx" ON "Step"("parentId");
CREATE INDEX "Step_startTime_idx" ON "Step"("startTime");
CREATE INDEX "Step_threadId_idx" ON "Step"("threadId");
CREATE INDEX "Step_type_idx" ON "Step"("type");
CREATE INDEX "Step_name_idx" ON "Step"("name");
CREATE INDEX "Step_threadId_startTime_endTime_idx"
    ON "Step"("threadId", "startTime", "endTime");
CREATE INDEX "Thread_createdAt_idx" ON "Thread"("createdAt");
CREATE INDEX "Thread_name_idx" ON "Thread"("name");

ALTER TABLE "Element"
    ADD CONSTRAINT "Element_stepId_fkey"
    FOREIGN KEY ("stepId") REFERENCES "Step"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "Element"
    ADD CONSTRAINT "Element_threadId_fkey"
    FOREIGN KEY ("threadId") REFERENCES "Thread"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "Feedback"
    ADD CONSTRAINT "Feedback_stepId_fkey"
    FOREIGN KEY ("stepId") REFERENCES "Step"("id")
    ON DELETE SET NULL ON UPDATE CASCADE;
ALTER TABLE "Step"
    ADD CONSTRAINT "Step_parentId_fkey"
    FOREIGN KEY ("parentId") REFERENCES "Step"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "Step"
    ADD CONSTRAINT "Step_threadId_fkey"
    FOREIGN KEY ("threadId") REFERENCES "Thread"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "Thread"
    ADD CONSTRAINT "Thread_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "User"("id")
    ON DELETE SET NULL ON UPDATE CASCADE;
