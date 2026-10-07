"""Call Responses with or without streaming."""

import argparse
import os

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true")
    arguments = parser.parse_args()
    with OpenAI(
        base_url=os.environ["CLIENT_BASE_URL"],
        api_key=os.environ["CLIENT_API_KEY"],
    ) as client:
        if arguments.stream:
            with client.responses.create(
                model="simple-graph",
                input="Explain LangGraph briefly.",
                store=False,
                stream=True,
            ) as events:
                for event in events:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                    elif event.type == "response.failed":
                        raise RuntimeError(event.response.error)
            print()
        else:
            response = client.responses.create(
                model="simple-graph", input="Explain LangGraph briefly.", store=False
            )
            print(response.output_text)


if __name__ == "__main__":
    main()
