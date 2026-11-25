"""Streams one question from the deployed API and prints each agent step as it arrives.

Usage:
    uv run python scripts/ask_live.py "your question" [--url URL]

Set DOCQA_API_KEY to send an X-API-Key header and skip the rate limits.
Uses only the standard library, so it runs without the project installed.
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

DEFAULT_URL = "https://docqa-agent.onrender.com"
COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def stream(url: str, question: str, api_key: str | None) -> Iterator[tuple[str, dict[str, Any]]]:
    headers = {"content-type": "application/json", "accept": "text/event-stream"}
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(
        f"{url.rstrip('/')}/ask/stream",
        data=json.dumps({"question": question}).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        event = ""
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                yield event, json.loads(line.removeprefix("data:"))


def print_result(result: dict[str, Any]) -> None:
    # Chunk ids are UUIDs; number them in citation order so the answer stays readable.
    numbers = {c["chunk_id"]: i for i, c in enumerate(result["citations"], start=1)}
    answer = re.sub(
        r"\[([^\[\]\s]+)\]",
        lambda m: f"[{numbers[m.group(1)]}]" if m.group(1) in numbers else m.group(0),
        result["answer"],
    )
    revisions = result["revisions"]
    if result["approved"]:
        verdict = style("approved by the reviewer", "32")
    else:
        verdict = style("not fully supported", "33")
    print(f"\n{style('Answer', '1')} ({verdict}, {revisions} revision{'s' * (revisions != 1)})\n")
    for paragraph in answer.split("\n"):
        print(textwrap.fill(paragraph, width=88) if paragraph.strip() else "")
    if result["citations"]:
        print(f"\n{style('Sources', '1')}")
        for chunk_id, n in numbers.items():
            citation = next(c for c in result["citations"] if c["chunk_id"] == chunk_id)
            print(f"  [{n}] {citation['document_title']}, chunk {citation['chunk_index']}")
    for claim in result["unsupported_claims"]:
        print(style(f"  unsupported: {claim}", "33"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream one question from the deployed API.")
    parser.add_argument("question")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    print(f"{style('Q:', '1')} {args.question}\n")
    start = time.monotonic()
    try:
        for event, data in stream(args.url, args.question, os.environ.get("DOCQA_API_KEY")):
            elapsed = style(f"{time.monotonic() - start:5.1f}s", "2")
            if event == "progress":
                print(f"{elapsed}  {style(data['step'].ljust(9), '36')}  {data['message']}")
                for query in data["queries"] if data["step"] == "planner" else []:
                    print(f"{' ' * 19}{style('- ' + query, '2')}")
            elif event == "result":
                print_result(data)
            elif event == "error":
                sys.exit(style(data["message"], "31"))
    except urllib.error.HTTPError as error:
        detail = json.loads(error.read() or b"{}").get("detail", error.reason)
        sys.exit(style(f"HTTP {error.code}: {detail}", "31"))


if __name__ == "__main__":
    main()
