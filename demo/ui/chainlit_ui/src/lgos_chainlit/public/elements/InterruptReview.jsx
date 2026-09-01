import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"

export default function InterruptReview() {
  const choices = Array.isArray(props.choices) ? props.choices : []
  const allowOther = props.allow_other === true || choices.length === 0
  const prompt =
    typeof props.prompt === "string" ? props.prompt : "Human input required."
  const [response, setResponse] = useState("")

  const submitResponse = () => {
    const value = response.trim()
    if (value) submitElement({ resume: value })
  }

  return (
    <Card className="w-full max-w-xl">
      <CardHeader>
        <CardTitle className="text-base">Human review</CardTitle>
        <p className="whitespace-pre-wrap text-sm text-muted-foreground">{prompt}</p>
      </CardHeader>

      <CardContent className="space-y-4">
        {choices.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {choices.map((choice) => (
              <Button
                key={choice}
                type="button"
                variant={choice.toLowerCase() === "reject" ? "destructive" : "default"}
                onClick={() => submitElement({ resume: choice })}
              >
                {choice}
              </Button>
            ))}
          </div>
        )}

        {allowOther && (
          <div className="space-y-2">
            <Label htmlFor="interrupt-response">Custom response</Label>
            <Textarea
              id="interrupt-response"
              placeholder="Write a response..."
              rows={3}
              value={response}
              onChange={(event) => setResponse(event.target.value)}
            />
          </div>
        )}
      </CardContent>

      <CardFooter className="justify-end gap-2">
        <Button type="button" variant="ghost" onClick={cancelElement}>
          Cancel
        </Button>
        {allowOther && (
          <Button type="button" onClick={submitResponse} disabled={!response.trim()}>
            Send
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
