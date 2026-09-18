import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useEffect, useState } from "react"

export default function InterruptReview() {
  const reviews = Array.isArray(props.reviews) ? props.reviews : []
  const control = props._chainlit_utils_hitl || {}
  const [responses, setResponses] = useState(() => reviews.map(() => ""))
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    setResponses(reviews.map(() => ""))
    setSubmitting(false)
    setError("")
  }, [control.revision])

  const setResponse = (index, value) => {
    setResponses((current) =>
      reviews.map((_, reviewIndex) =>
        reviewIndex === index ? value : current[reviewIndex] || ""
      )
    )
  }

  const ready =
    reviews.length > 0 &&
    responses.length === reviews.length &&
    responses.every((response) => response.trim()) &&
    typeof control.action === "string" &&
    typeof control.step_id === "string" &&
    typeof control.element_id === "string" &&
    typeof control.revision === "string"

  const submitResponses = async () => {
    if (!ready || submitting) return
    setSubmitting(true)
    setError("")
    try {
      const result = await callAction({
        name: control.action,
        payload: {
          step_id: control.step_id,
          element_id: control.element_id,
          revision: control.revision,
          outputs: responses.map((response) => response.trim()),
        },
      })
      if (result?.response?.ok !== true) {
        setError(
          result?.response?.error || "The human review could not be submitted."
        )
      }
    } catch (submissionError) {
      setError(String(submissionError))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card className="w-full max-w-xl">
      <CardHeader>
        <CardTitle className="text-base">Human review</CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {reviews.map((review, index) => {
          const choices = Array.isArray(review.choices) ? review.choices : []
          const allowOther = review.allow_other === true || choices.length === 0
          const prompt =
            typeof review.prompt === "string"
              ? review.prompt
              : "Human input required."
          const response = responses[index] || ""
          const inputId = `interrupt-response-${index}`

          return (
            <section key={index} className="space-y-3">
              <p className="whitespace-pre-wrap text-sm text-muted-foreground">
                {prompt}
              </p>

              {choices.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {choices.map((choice) => (
                    <Button
                      key={choice}
                      type="button"
                      variant={response === choice ? "default" : "outline"}
                      disabled={submitting}
                      onClick={() => setResponse(index, choice)}
                    >
                      {choice}
                    </Button>
                  ))}
                </div>
              )}

              {allowOther && (
                <div className="space-y-2">
                  <Label htmlFor={inputId}>Custom response</Label>
                  <Textarea
                    id={inputId}
                    placeholder="Write a response..."
                    rows={3}
                    value={response}
                    disabled={submitting}
                    onChange={(event) => setResponse(index, event.target.value)}
                  />
                </div>
              )}
            </section>
          )
        })}

        {error && <p className="text-sm text-destructive">{error}</p>}
      </CardContent>

      <CardFooter className="justify-end">
        <Button
          type="button"
          onClick={submitResponses}
          disabled={!ready || submitting}
        >
          {submitting ? "Submitting..." : "Submit"}
        </Button>
      </CardFooter>
    </Card>
  )
}
