from openai.types.responses import ResponseStreamEvent


def answer_deltas(events: list[ResponseStreamEvent]) -> list[str]:
    phases = {
        event.item.id: event.item.phase
        for event in events
        if event.type == "response.output_item.added" and event.item.type == "message"
    }
    return [
        event.delta
        for event in events
        if event.type == "response.output_text.delta"
        and phases.get(event.item_id) == "final_answer"
    ]
