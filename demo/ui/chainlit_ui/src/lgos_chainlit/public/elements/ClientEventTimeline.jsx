import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Activity, Check, FileCheck2, LoaderCircle } from "lucide-react"

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

function humanize(value) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function eventSummary(data) {
  if (typeof data.message === "string") return data.message
  if (typeof data.title === "string") return data.title

  const scalar = Object.entries(data).find(
    ([, value]) => ["string", "number", "boolean"].includes(typeof value),
  )
  if (!scalar) return null

  const [key, value] = scalar
  return `${humanize(key)}: ${value}`
}

export default function ClientEventTimeline() {
  const events = Array.isArray(props.events) ? props.events : []
  const latestEvent = events[events.length - 1]
  const workflowComplete = latestEvent?.type === "artifact"
  const latestProgress = [...events]
    .reverse()
    .find((event) => event?.type === "progress" && isObject(event.data))
  const completed = latestProgress?.data?.completed
  const total = latestProgress?.data?.total
  const progressValue =
    typeof completed === "number" &&
    Number.isFinite(completed) &&
    typeof total === "number" &&
    Number.isFinite(total) &&
    total > 0
      ? Math.min(100, Math.max(0, (completed / total) * 100))
      : null

  return (
    <Card className="relative w-full overflow-hidden border-border/60 bg-gradient-to-br from-card via-card to-muted/30 shadow-lg shadow-black/5">
      <div className="absolute -right-20 -top-20 h-48 w-48 rounded-full bg-primary/5 blur-3xl" />
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/70 to-transparent" />

      <CardHeader className="relative space-y-5 pb-5">
        <div className="flex items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/15">
              <Activity className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <CardTitle className="text-base font-semibold tracking-tight">
                Workflow activity
              </CardTitle>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Live updates from the graph
              </p>
            </div>
          </div>

          <Badge
            variant="outline"
            className={
              workflowComplete
                ? "gap-1.5 border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1 text-emerald-500"
                : "gap-1.5 border-primary/20 bg-primary/10 px-2.5 py-1 text-primary"
            }
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                workflowComplete ? "bg-emerald-500" : "animate-pulse bg-primary"
              }`}
            />
            {workflowComplete ? "Complete" : humanize(latestEvent?.type || "waiting")}
          </Badge>
        </div>

        {progressValue !== null && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-foreground">
                {completed} of {total} stages
              </span>
              <span className="tabular-nums text-muted-foreground">
                {Math.round(progressValue)}%
              </span>
            </div>
            <Progress value={progressValue} className="h-1.5" />
          </div>
        )}
      </CardHeader>

      <CardContent className="relative border-t border-border/50 pb-5 pt-5">
        <div className="relative">
          <div className="absolute bottom-4 left-4 top-4 w-px bg-border/70" />

          {events.map((event, index) => {
            const data = isObject(event?.data) ? event.data : {}
            const namespace = Array.isArray(event?.namespace)
              ? event.namespace.join(" / ")
              : ""
            const summary = eventSummary(data)
            const isArtifact = event?.type === "artifact"
            const isFinished = index < events.length - 1 || workflowComplete
            const Icon = isArtifact ? FileCheck2 : isFinished ? Check : LoaderCircle
            const detail = isArtifact
              ? [data.kind, data.id]
                  .filter((value) => typeof value === "string")
                  .join(" · ")
              : typeof data.stage === "string"
                ? humanize(data.stage)
                : ""

            return (
              <div
                key={`${index}-${event?.type || "event"}`}
                className="relative flex gap-3 pb-5 last:pb-0"
              >
                <div
                  className={`relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border shadow-sm ${
                    isArtifact
                      ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-500"
                      : isFinished
                        ? "border-primary/20 bg-background text-primary"
                        : "border-primary/25 bg-primary/10 text-primary"
                  }`}
                >
                  <Icon className={`h-4 w-4 ${!isFinished ? "animate-spin" : ""}`} />
                </div>

                <div className="min-w-0 flex-1 pt-0.5">
                  <p
                    className={`text-sm font-medium leading-5 ${
                      isArtifact ? "text-emerald-500" : "text-foreground"
                    }`}
                  >
                    {summary || humanize(event?.type || "event")}
                  </p>

                  {(event?.type || namespace || detail) && (
                    <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                      {event?.type && (
                        <span className="font-medium text-primary/80">
                          {humanize(event.type)}
                        </span>
                      )}
                      {event?.type && namespace && <span aria-hidden="true">·</span>}
                      {namespace && <span>{namespace}</span>}
                      {(event?.type || namespace) && detail && (
                        <span aria-hidden="true">·</span>
                      )}
                      {detail && <span>{detail}</span>}
                    </div>
                  )}

                  {!summary && (
                    <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded-lg border border-border/40 bg-muted/30 p-2.5 text-xs text-muted-foreground">
                      {JSON.stringify(event?.data ?? null, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
