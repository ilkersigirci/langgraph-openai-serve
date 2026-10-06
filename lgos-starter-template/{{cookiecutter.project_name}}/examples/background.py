"""Submit, poll, or cancel a background Response through the OpenAI SDK."""

import argparse
import os
import time
from uuid import uuid4

from openai import OpenAI
from openai.types.responses import Response


def wait_for_response(client: OpenAI, response: Response) -> Response:
    deadline = time.monotonic() + 120
    while response.status in {"queued", "in_progress"}:
        if time.monotonic() >= deadline:
            msg = f"Polling timed out; run {response.id} remains active."
            raise TimeoutError(msg)
        time.sleep(0.5)
        response = client.responses.retrieve(response.id)
    if response.status != "completed":
        msg = f"Response ended as {response.status}: {response.error}"
        raise RuntimeError(msg)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    with OpenAI(
        base_url=os.getenv("CLIENT_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("CLIENT_API_KEY", "DUMMY"),
    ) as client:
        response = client.responses.create(
            model="simple-graph",
            input="Explain durable background execution.",
            background=True,
            store=True,
            extra_headers={"Idempotency-Key": str(uuid4())},
        )
        print(response.id, response.status)
        if args.cancel:
            print(client.responses.cancel(response.id).status)
        else:
            print(wait_for_response(client, response).output_text)


if __name__ == "__main__":
    main()
