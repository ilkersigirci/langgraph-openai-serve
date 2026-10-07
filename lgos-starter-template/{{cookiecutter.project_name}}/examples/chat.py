"""Call Chat Completions with or without streaming."""

import argparse
import os

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true")
    arguments = parser.parse_args()
    with OpenAI(
        base_url=os.getenv("CLIENT_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("CLIENT_API_KEY", "DUMMY"),
    ) as client:
        if arguments.stream:
            with client.chat.completions.create(
                model="simple-graph",
                messages=[{"role": "user", "content": "Explain LangGraph briefly."}],
                stream=True,
            ) as chunks:
                for chunk in chunks:
                    for choice in chunk.choices:
                        print(choice.delta.content or "", end="", flush=True)
            print()
        else:
            response = client.chat.completions.create(
                model="simple-graph",
                messages=[{"role": "user", "content": "Explain LangGraph briefly."}],
            )
            print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
