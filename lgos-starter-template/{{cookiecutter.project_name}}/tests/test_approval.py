import pytest
from openai import AsyncOpenAI, BadRequestError, ConflictError


@pytest.mark.parametrize(
    ("decision", "expected"),
    [("approve", "Request approved."), ("reject", "Request rejected.")],
)
async def test_approval_resumes_once(
    openai_client: AsyncOpenAI, decision: str, expected: str
) -> None:
    paused = await openai_client.responses.create(
        model="approval", input="Publish the report", store=False
    )
    calls = [item for item in paused.output if item.type == "function_call"]
    assert len(calls) == 1
    assert calls[0].name == "lgos_interrupt"
    outputs = [
        {"type": "function_call_output", "call_id": call.call_id, "output": decision}
        for call in calls
    ]
    resumed = await openai_client.responses.create(
        model="approval", previous_response_id=paused.id, input=outputs, store=False
    )
    assert resumed.output_text == expected
    with pytest.raises(ConflictError):
        await openai_client.responses.create(
            model="approval", previous_response_id=paused.id, input=outputs, store=False
        )


async def test_approval_rejects_an_invalid_decision(openai_client: AsyncOpenAI) -> None:
    paused = await openai_client.responses.create(
        model="approval", input="Publish the report", store=False
    )
    (call,) = [item for item in paused.output if item.type == "function_call"]
    with pytest.raises(BadRequestError) as error:
        await openai_client.responses.create(
            model="approval",
            previous_response_id=paused.id,
            store=False,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "maybe",
                }
            ],
        )
    assert error.value.param == "input"
    assert error.value.code == "invalid_approval_decision"
