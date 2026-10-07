"""Pause and resume an approval; add --background to run it in the background."""

import argparse
import os
from uuid import uuid4

from background import wait_for_response
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", action="store_true")
    background = parser.parse_args().background
    with OpenAI(
        base_url=os.environ["CLIENT_BASE_URL"],
        api_key=os.environ["CLIENT_API_KEY"],
    ) as client:
        paused = client.responses.create(
            model="approval",
            input="Publish the reviewed report.",
            background=background,
            store=background,
            extra_headers={"Idempotency-Key": str(uuid4())},
        )
        if background:
            paused = wait_for_response(client, paused)
        calls = [
            item
            for item in paused.output
            if item.type == "function_call" and item.name == "lgos_interrupt"
        ]
        for call in calls:
            print(call.arguments)
        answer = input("Decision (approve/reject): ")
        resumed = client.responses.create(
            model="approval",
            previous_response_id=paused.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": answer,
                }
                for call in calls
            ],
            background=background,
            store=background,
            extra_headers={"Idempotency-Key": str(uuid4())},
        )
        if background:
            resumed = wait_for_response(client, resumed)
        print(resumed.output_text)


if __name__ == "__main__":
    main()
