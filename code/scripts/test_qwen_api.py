"""Run a small connection test against the supervisor-hosted Qwen API."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI


DEFAULT_BASE_URL = "https://resolution-andreas-alerts-blah.trycloudflare.com/v1"
DEFAULT_API_KEY = "local-key"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


def _error_detail(exc: BaseException) -> str:
    """Include a distinct underlying transport cause when one is available."""
    details = [str(exc)]
    cause = exc.__cause__
    while cause is not None:
        cause_text = str(cause)
        if cause_text and cause_text not in details:
            details.append(cause_text)
        cause = cause.__cause__
    return ": ".join(details)


def run_connection_test() -> dict[str, Any]:
    """Call the Qwen endpoint and return its validated JSON response."""
    base_url = os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("QWEN_API_KEY", DEFAULT_API_KEY)
    model = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only valid JSON. Do not include Markdown or any "
                    "explanatory text."
                ),
            },
            {
                "role": "user",
                "content": 'Return exactly this JSON object: {"status":"working"}',
            },
        ],
        max_tokens=500,
        temperature=0,
        response_format={"type": "json_object"},
    )

    if not response.choices:
        raise ValueError("API response contained no completion choices")

    content = response.choices[0].message.content
    if not content:
        raise ValueError("API response contained no text")

    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("API response JSON was not an object")
    if parsed.get("status") != "working":
        raise ValueError(
            "API response had an unexpected status: "
            f"{parsed.get('status')!r}"
        )
    return parsed


def main() -> int:
    """Run the test and convert expected failures into useful exit messages."""
    try:
        result = run_connection_test()
    except APITimeoutError as exc:
        print(
            f"Qwen API test failed: request timed out: {_error_detail(exc)}",
            file=sys.stderr,
        )
        return 1
    except APIConnectionError as exc:
        print(
            f"Qwen API test failed: connection error: {_error_detail(exc)}",
            file=sys.stderr,
        )
        return 1
    except APIError as exc:
        print(f"Qwen API test failed: API error: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Qwen API test failed: invalid JSON response: {exc}", file=sys.stderr)
        return 1
    except (TypeError, ValueError, IndexError, AttributeError) as exc:
        print(f"Qwen API test failed: invalid response: {exc}", file=sys.stderr)
        return 1

    print(f"Qwen API connection successful. Returned JSON status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
