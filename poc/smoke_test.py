"""
Smoke test: confirm the Groq API key works AND that native (OpenAI-compatible)
tool-calling round-trips correctly. This is the same mechanism the real agent
loop uses in app.py, just stripped to one tool and one turn.

Run:
    pip install groq
    export GROQ_API_KEY=...        # or copy .env.example -> .env
    python poc/smoke_test.py

Expected: finish_reason == "tool_calls", and the model calls add(a=2, b=3).
"""

import json
import os
import sys

from groq import Groq

MODEL = "llama-3.3-70b-versatile"  # fallback: "llama-3.1-8b-instant"

# A trivial tool so we can prove the model chooses to call it.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def main() -> int:
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set. See poc/.env.example.")
        return 1

    client = Groq()  # reads GROQ_API_KEY from the env

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=200,
        tools=TOOLS,
        tool_choice="auto",
        messages=[
            {"role": "user", "content": "What is 2 + 3? Use the add tool."}
        ],
    )

    choice = resp.choices[0]
    print(f"finish_reason: {choice.finish_reason}")

    tool_called = False
    for call in choice.message.tool_calls or []:
        tool_called = True
        print(f"tool_call -> {call.function.name}({call.function.arguments})")

    if choice.finish_reason == "tool_calls" and tool_called:
        print("\nOK: key works and native tool-calling round-trips.")
        return 0

    print(f"\nWARNING: model did not call the tool. text -> {choice.message.content!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
